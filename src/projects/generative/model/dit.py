import typing

from flax import linen as nn
import jax
from jax import numpy as jnp


# ==============================================================================
# Helper Functions
@jax.jit
def modulate(
    inputs: jax.Array,
    shift: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    r"""Modulates the input tensor with given shift and scale.

    Args:
        inputs (jax.Array): Input tensor of shape (*, L, C).
        shift (jax.Array): Shift tensor of shape (*, C).
        scale (jax.Array): Scale tensor of shape (*, C).

    Returns:
        Modulated tensor of shape (*, L, C).
    """
    return inputs * (1 + scale[..., None, :]) + shift[..., None, :]


@jax.jit
def quick_gelu(x: jax.Array) -> jax.Array:
    r"""Applies the Quick GELU activation function.

    Args:
        x (jax.Array): Input tensor.

    Returns:
        Activated tensor.
    """
    return x * jax.nn.sigmoid(1.702 * x)


@jax.jit
def approx_gelu_tanh(x: jax.Array) -> jax.Array:
    r"""Applies the approximate GELU activation function using Tanh.

    Args:
        x (jax.Array): Input tensor.

    Returns:
        Activated tensor.
    """
    sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
    x_cubed = jnp.power(x, 3)
    return 0.5 * x * (1 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed)))


# ==============================================================================
# Modules
class SelfAttention(nn.Module):
    r"""Multi-head self-attention operator module.

    Args:
        features (int): Dimensionality of the feature space.
        num_heads (int, optional): Number of attention heads. Default is `8`.
        qkv_bias (bool, optional): Whether to add bias in query, key, value
            projections. Default is `False`.
        qk_norm (bool, optional): Whether to apply layer normalization to
            queries and keys. Default is `False`.
        attn_dropout_rate (float, optional): Dropout rate for attention weights.
            Default is `0.0`.
        proj_bias (bool, optional): Whether to add bias in output projection.
            Default is `False`.
        proj_dropout_rate (float, optional): Dropout rate for output projection.
            Default is `0.0`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout_rate: float = 0.0
    proj_bias: bool = False
    proj_dropout_rate: float = 0.0
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
        mask: typing.Optional[jax.Array] = None,
    ) -> jax.Array:
        r"""Forward pass multi-head scaled dot-product self-attention.

        Args:
            inputs (jax.Array): Input tensor of shape `(*, L, C)`.
            deterministic (bool, optional): Whether to apply dropout.
                It merges with the module-level attribute `deterministic`.
                Default is `None`.
            mask (Optional[jax.Array], optional): Optional attention mask of
                shape `(*, H, L, L)`, where `H` is number of heads.
                Default is `None`.

        Returns:
            Output tensor of shape `(*, L, C)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        qkv_proj = nn.Dense(
            features=3 * self.features,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="attn_proj",
        )
        q, k, v = jnp.split(qkv_proj(inputs.astype(self.dtype)), 3, axis=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.qk_norm:
            q = nn.LayerNorm(
                epsilon=1e-6,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="q_norm",
            )(q)
            k = nn.LayerNorm(
                epsilon=1e-6,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="k_norm",
            )(k)

        q_scaled = q / jnp.sqrt(jnp.sqrt(k.shape[-1]).astype(self.dtype))
        k_scaled = k / jnp.sqrt(jnp.sqrt(k.shape[-1]).astype(self.dtype))
        attn_weights = jnp.einsum(
            "...qhd,...khd ->...hqk",
            q_scaled,
            k_scaled,
            precision=self.precision,
        )
        if mask is not None:
            big_neg = jnp.finfo(self.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        if self.attn_dropout_rate > 0.0:
            attn_weights = nn.Dropout(
                rate=self.attn_dropout_rate,
                deterministic=m_deterministic,
                name="attn_dropout",
            )(attn_weights)

        out = jnp.einsum(
            "...hqk,...khd->...qhd",
            attn_weights,
            v,
            precision=self.precision,
        )
        out = nn.DenseGeneral(
            features=self.features,
            axis=(-2, -1),
            use_bias=self.proj_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="out_proj",
        )(out)
        if self.proj_dropout_rate > 0.0:
            out = nn.Dropout(
                rate=self.proj_dropout_rate,
                deterministic=m_deterministic,
                name="out_dropout",
            )(out)

        return out

    def _split_heads(self, x: jax.Array) -> jax.Array:
        r"""Split the last dimension into (num_heads, head_dim)."""
        *batch_dims, seq_len, features = x.shape
        head_dim = features // self.num_heads
        if head_dim * self.num_heads != features:
            raise ValueError(
                "Input dimension must be divisible by num_heads. "
                f"Got features={features}, num_heads={self.num_heads}."
            )
        out = x.reshape(*batch_dims, seq_len, self.num_heads, head_dim)
        return out


class FeedForwardNetwork(nn.Module):
    r"""Feed-forward network (FFN) in Transformer.

    Args:
        features (int): Dimensionality of the feature space.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        dropout_rate (float, optional): Dropout rate. Default is `0.0`.
        activation (Callable, optional): Activation function.
            Default is `approx_gelu_tanh`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
    """

    features: int
    ffn_ratio: int = 4
    dropout_rate: float = 0.0
    activation: typing.Callable[..., jax.Array] = approx_gelu_tanh
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the feed-forward network.

        Args:
            inputs (jax.Array): Input tensor of shape `(*, L, C)`.

        Returns:
            Output tensor of shape `(*, L, C)`.
        """
        hidden_dim = self.features * self.ffn_ratio
        out = nn.Dense(
            features=hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="fc_in",
        )(inputs.astype(self.dtype))
        out = self.activation(out)
        if self.dropout_rate > 0.0:
            m_deterministic = nn.merge_param(
                "deterministic",
                self.deterministic,
                None,
            )
            out = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=m_deterministic,
                name="dropout",
            )(out)
        out = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="fc_out",
        )(out)
        return out


class _DiTBlock(nn.Module):
    r"""General Diffusion Transformers (DiT) block.

    Args:
        features (int): Dimensionality of the feature space.
        num_heads (int, optional): Number of attention heads. Default is `8`.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    num_heads: int = 8
    ffn_ratio: int = 4
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @property
    def block_type(self) -> str:
        r"""str: One of 'in-context', 'cross-attention', or 'adaLN'."""
        ...

    def setup(self) -> None:
        """Sets up the DiT block components."""
        self.norm_1 = nn.LayerNorm(
            epsilon=1e-6,
            use_bias=not (self.block_type == "adaLN"),
            use_scale=not (self.block_type == "adaLN"),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attn_norm",
        )
        self.attn = SelfAttention(
            features=self.features,
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="attn",
        )
        self.norm_2 = nn.LayerNorm(
            epsilon=1e-6,
            use_bias=not (self.block_type == "adaLN"),
            use_scale=not (self.block_type == "adaLN"),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ffn_norm",
        )
        self.mlp = FeedForwardNetwork(
            features=self.features,
            ffn_ratio=self.ffn_ratio,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ffn",
        )
        if self.block_type == "adaLN":
            self.adaln_modulation = nn.Dense(
                features=6 * self.features,
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="adaln_modulation",
            )

    def __call__(
        self,
        inputs: jax.Array,
        cond: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the DiT block.

        Args:
            inputs (jax.Array): Input tensor of shape `(*, L, C)`.
            cond (jax.Array): Conditioning tensor of shape `(*, C)`.
            deterministic (bool, optional): Whether to apply dropout.
                Default is `None`.

        Returns:
            Output tensor of shape `(*, L, C)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        out = inputs.astype(self.dtype)
        cond = cond.astype(self.dtype)
        if self.block_type == "in-context":
            out = jnp.concatenate([out, cond[:, None, :]], axis=-2)
            skip = out
            out = self.norm_1(out)
            out = self.attn(out, deterministic=m_deterministic)
            out = out + skip

            skip = out
            out = self.norm_2(out)
            out = self.mlp(out)
            out = out + skip
        elif self.block_type == "cross-attention":
            raise NotImplementedError
        elif self.block_type == "adaLN":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_ffn,
                scale_ffn,
                gate_msa,
            ) = jnp.split(
                self.adaln_modulation(jax.nn.silu(cond)),
                indices_or_sections=6,
                axis=-1,
            )
            out = out + gate_msa[..., None, :] * self.attn(
                inputs=modulate(
                    self.norm_1(out),
                    shift=shift_msa,
                    scale=scale_msa,
                ),
                deterministic=m_deterministic,
            )
            out = out + gate_msa[..., None, :] * self.mlp(
                modulate(
                    self.norm_2(out),
                    shift=shift_ffn,
                    scale=scale_ffn,
                )
            )

        return out


class DiTConditioningBlock(_DiTBlock):
    r"""Diffusion Transformers (DiT) block with in-context conditioning.

    Args:
        features (int): Dimensionality of the feature space.
        num_heads (int, optional): Number of attention heads. Default is `8`.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    @property
    def block_type(self) -> str:
        return "in-context"


class DiTCrossAttentionBlock(_DiTBlock):
    r"""Diffusion Transformers (DiT) block with cross-attention.

    Args:
        features (int): Dimensionality of the feature space.
        num_heads (int, optional): Number of attention heads. Default is `8`.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    @property
    def block_type(self) -> str:
        return "cross-attention"


class DiTAdaLNBlock(_DiTBlock):
    r"""Diffusion Transformers (DiT) block with adaptive layer normalization.

    Args:
        features (int): Dimensionality of the feature space.
        num_heads (int, optional): Number of attention heads. Default is `8`.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    @property
    def block_type(self) -> str:
        return "adaLN"
