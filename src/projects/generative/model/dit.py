import functools
import typing

import chex
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


@functools.partial(jax.jit, static_argnames=("features",))
def sinusoidal_pos_enc(features: int, pos: jax.Array) -> jax.Array:
    r"""Returns 1D sinusoidal positional encodings.

    Args:
        features (int): Dimensionality of the embedding.
        pos (jax.Array): Positions tensor of shape `(*,)`.

    Returns:
        Positional encodings of shape `(flattened_shape, features)`.
    """
    assert features % 2 == 0, "Features must be even for sinusoidal encoding."
    pos = pos.reshape(-1)

    indx = jnp.arange(features // 2, dtype=jnp.float_)
    indx /= features / 2.0
    freqs = 1.0 / jnp.power(10_000.0, indx)

    emb = jnp.einsum("p,f->pf", pos, freqs)
    out = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    chex.assert_shape(out, (pos.shape[0], features))

    return out


def sinusoidal_patch_enc(
    features: int,
    grid_size: int,
    num_extra_tokens: int = 0,
) -> jax.Array:
    r"""Returns sinusoidal positional encodings for 2D patches.

    Args:
        features (int): Dimensionality of the positional embedding.
        grid_size (int): Size of the (squared) grid of patches.
        num_extra_tokens (int, optional): Number of extra tokens to include.
            Default is `0`.

    Returns:
        Positional encodings array with a shape of
            `(num_patches + num_extra_tokens, features)`.
    """
    if features % 4 != 0:
        raise ValueError(
            "Dimensions must be divisible by 4 for 2D positional encoding."
        )

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.stack(jnp.meshgrid(grid_w, grid_h), axis=0)  # (2, gs, gs)

    emb_h = sinusoidal_pos_enc(features // 2, grid[0:1])
    emb_w = sinusoidal_pos_enc(features // 2, grid[1:2])
    out = jnp.concatenate([emb_h, emb_w], axis=-1)

    if num_extra_tokens > 0:
        extra_emb = jnp.zeros([num_extra_tokens, features], dtype=out.dtype)
        out = jnp.concatenate([extra_emb, out], axis=0)

    return out


# ==============================================================================
# Modules
class Attention(nn.Module):
    r"""Multi-head attention operator module.

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
        query: jax.Array,
        key_value: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
        mask: typing.Optional[jax.Array] = None,
    ) -> jax.Array:
        r"""Forward pass multi-head scaled dot-product attention.

        Args:
            query (jax.Array): Query tensor of shape `(*, L, C)`.
            key_value (Optional[jax.Array], optional): Key and value tensor of
                shape `(*, S, C)`. If `None`, self-attention is performed.
                Default is `None`.
            deterministic (bool, optional): Whether to apply dropout.
                It merges with the module-level attribute `deterministic`.
                Default is `None`.
            mask (Optional[jax.Array], optional): Optional attention mask of
                shape `(*, H, L, S)`, where `H` is number of heads.
                Default is `None`.

        Returns:
            Output tensor of shape `(*, L, C)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        if key_value is None:
            qkv_proj = nn.Dense(
                features=3 * self.features,
                use_bias=self.qkv_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="attn_proj",
            )
            q, k, v = jnp.split(qkv_proj(query.astype(self.dtype)), 3, axis=-1)
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)
        else:
            q_proj = nn.Dense(
                features=self.features,
                use_bias=self.qkv_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="query_proj",
            )
            k_proj = nn.Dense(
                features=self.features,
                use_bias=self.qkv_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="key_proj",
            )
            v_proj = nn.Dense(
                features=self.features,
                use_bias=self.qkv_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="value_proj",
            )
            q = self._split_heads(q_proj(query.astype(self.dtype)))
            k = self._split_heads(k_proj(key_value.astype(self.dtype)))
            v = self._split_heads(v_proj(key_value.astype(self.dtype)))

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
        self.attn = Attention(
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
        if self.block_type == "cross-attention":
            self.norm_ca = nn.LayerNorm(
                epsilon=1e-6,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="cross_attn_norm",
            )
            self.cross_attn = Attention(
                features=self.features,
                num_heads=self.num_heads,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="cross_attn",
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
            skip = out
            out = self.norm_1(out)
            out = self.attn(
                query=out,
                key_value=cond,
                deterministic=m_deterministic,
            )
            out = out + skip

            skip = out
            out = self.norm_ca(out)
            out = self.cross_attn(
                query=out,
                key_value=cond[..., None, :],
                deterministic=m_deterministic,
            )
            out = out + skip

            skip = out
            out = self.norm_2(out)
            out = self.mlp(out)
            out = out + skip
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
                query=modulate(
                    self.norm_1(out),
                    shift=shift_msa,
                    scale=scale_msa,
                ),
                key_value=None,
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


class _Decoder(nn.Module):
    r"""Final linear decoder module.

    Args:
        features (int): Dimensionality of output feature map.
        patch_size (int): Patch size of the output feature map.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    patch_size: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @property
    def block_type(self) -> str:
        r"""str: Decoder block. Either "standard" or "adaLN"."""
        ...

    @nn.compact
    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        r"""Forward pass of the decoder block.

        Args:
            inputs (jax.Array): Input tensor of shape `(*, L, C)`.
            cond (jax.Array): Conditioning tensor of shape `(*, C)`.

        Returns:
            Output tensor of shape `(*, L, patch_size * patch_size * features)`.
        """
        out = inputs.astype(self.dtype)
        norm = nn.LayerNorm(
            epsilon=1e-6,
            use_bias=not (self.block_type == "adaLN"),
            use_scale=not (self.block_type == "adaLN"),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm_final",
        )
        if self.block_type == "adaLN":
            adaln_modulation = nn.Dense(
                features=2 * out.shape[-1],
                use_bias=True,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="adaln_modulation",
            )
            shift, scale = jnp.split(
                adaln_modulation(jax.nn.silu(cond)),
                indices_or_sections=2,
                axis=-1,
            )
            out = modulate(norm(out), shift=shift, scale=scale)
        else:
            out = norm(out)
        proj = nn.Dense(
            features=self.patch_size * self.patch_size * self.features,
            use_bias=True,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="linear",
        )
        out = proj(out)
        chex.assert_shape(
            out,
            (*inputs.shape[:-1], self.patch_size**2 * self.features),
        )

        return out


class StandardDecoder(_Decoder):
    r"""Final linear decoder module with standard layer normalization.

    Args:
        features (int): Dimensionality of output feature map.
        patch_size (int): Patch size of the output feature map.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    @property
    def block_type(self) -> str:
        return "standard"


class AdaLNDecoder(_Decoder):
    r"""Final linear decoder module with adaptive layer normalization.

    Args:
        features (int): Dimensionality of output feature map.
        patch_size (int): Patch size of the output feature map.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    @property
    def block_type(self) -> str:
        return "adaLN"


class PatchEmbed(nn.Module):
    r"""Embedding module that encode 2D images into patch tokens.

    Args:
        features (int): Dimensionality of the embedding.
        patch_size (int): Patch size for encoding 2D images.
        flatten (bool, optional): Whether to flatten the output patches.
            Default is `True`.
        padding (bool, optional): Whether to apply dynamic padding to input
            images. Default is `False`.
        use_bias (bool, optional): Whether to use bias in the convolutional
            projection. Default is `True`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    patch_size: int
    flatten: bool = True
    padding: bool = False
    use_bias: bool = True
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the patch embedding.

        Args:
            inputs (jax.Array): Input tensor of shape `(*, H, W, C)`.

        Returns:
            Output tensor of shape `(*, N, D)`, where `N` is number of patches
                and `D` is embedding dimension.
        """
        *batch_dims, height, width, channels = inputs.shape
        out = inputs.reshape(-1, height, width, channels).astype(self.dtype)

        # dynamic padding
        if self.padding:
            pad_h = jnp.remainder(
                self.patch_size - height % self.patch_size,
                self.patch_size,
            )
            pad_w = jnp.remainder(
                self.patch_size - width % self.patch_size,
                self.patch_size,
            )
            out = jnp.pad(
                out,
                ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            )
        else:
            assert height % self.patch_size == 0, (
                f"Input height {height} is not divisible by patch size "
                f"{self.patch_size}."
            )
            assert width % self.patch_size == 0, (
                f"Input width {width} is not divisible by patch size "
                f"{self.patch_size}."
            )

        # encoding and reshaping
        proj = nn.Conv(
            features=self.features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding=0,
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="proj",
        )
        out = proj(out)

        if self.flatten:
            out = out.reshape(*batch_dims, -1, self.features)
        else:
            out = jnp.reshape(
                out,
                (*batch_dims, out.shape[-3], out.shape[-2], self.features),
            )

        return out


class TimestampEmbed(nn.Module):
    r"""Embedding module to encode scalar diffusion timesteps.

    Args:
        features (int): Dimensionality of the embedding.
        frequency_embedding_size (int, optional): Dimensionality of the
            sinusoidal frequency embedding. Default is `256`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
    """

    features: int
    frequency_embedding_size: int = 256
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, timesteps: jax.Array) -> jax.Array:
        r"""Forward pass of the timestamp embedding.

        Args:
            timesteps (jax.Array): One-dimensional timestamps of shape `(B,)`.

        Returns:
            Positional embeddings of shape `(B, D)`, where `D` is the
                dimensionality of the embedding.
        """
        # sinusoidal positional embedding
        out = self.sinusoidal_embedding(
            timesteps=timesteps,
            features=self.frequency_embedding_size,
        )
        out = out.astype(self.dtype)

        # feed-forward network
        fc_proj = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="fc_proj",
        )
        out = jax.nn.silu(fc_proj(out))
        fc_out = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="fc_out",
        )
        out = fc_out(out)

        return out

    @staticmethod
    def sinusoidal_embedding(
        timesteps: jax.Array,
        features: int,
        max_length: int = 10_000,
    ) -> jax.Array:
        r"""Generates sinusoidal embeddings for the given timesteps.

        Args:
            timesteps (jax.Array): One-dimensional timestamps of shape `(B,)`.
            features (int): Dimensionality of the embedding.
            max_length (int, optional): Maximum length for frequency scaling.
                Default is `10_000`.

        Returns:
            Positional embeddings of shape `(B, features)`.
        """
        half_dim = features // 2
        freqs = jnp.exp(
            -jnp.log(max_length)
            * jnp.arange(start=0, stop=half_dim, dtype=jnp.float32)
            / half_dim
        )
        embed = timesteps[:, None] * freqs[None, :]
        out = jnp.concatenate([jnp.cos(embed), jnp.sin(embed)], axis=-1)

        if features % 2 == 1:
            out = jnp.pad(
                out,
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            )

        return out


class LabelEmbed(nn.Module):
    r"""Embedding module for discrete class labels with optional dropout.

    Args:
        features (int): Dimensionality of the embedding.
        num_classes (int): Number of discrete classes.
        dropout_rate (float, optional): Dropout rate for label embedding.
            Default is `0.0`.
        deterministic (bool, optional): Whether to apply dropout.
            Default is `None`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
    """

    features: int
    num_classes: int
    dropout_rate: float = 0.0
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        labels: jax.Array,
        deterministic: typing.Optional[bool] = False,
        force_drop_ids: typing.Optional[jax.Array] = None,
    ) -> jax.Array:
        r"""Forward pass of the label embedding module.

        Args:
            labels (jax.Array): Class labels of shape `(B,)`.
            deterministic (bool, optional): Whether to apply dropout.
                Default is `False`.
            force_drop_ids (Optional[jax.Array], optional): Optional binary
                mask of shape `(B,)` to force dropout on specific labels.
                Default is `None`.

        Returns:
            Label embeddings of shape `(B, D)`, where `D` is the
                dimensionality of the embedding.
        """
        use_cfg_embedding = self.dropout_rate > 0.0
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # apply random label dropout
        if (
            not m_deterministic and self.dropout_rate > 0.0
        ) or force_drop_ids is not None:
            if force_drop_ids is None:
                rng = self.make_rng("dropout")
                drop_mask = jnp.less(
                    jax.random.uniform(
                        rng,
                        shape=labels.shape,
                        minval=0.0,
                        maxval=1.0,
                    ),
                    self.dropout_rate,
                )
            else:
                chex.assert_shape(force_drop_ids, labels.shape)
                drop_mask = force_drop_ids == 1
            labels = jnp.where(
                drop_mask,
                self.num_classes,
                labels,
            )

        # forward pass the embedding
        embed = nn.Embed(
            num_embeddings=self.num_classes + int(use_cfg_embedding),
            features=self.features,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="embed_table",
        )
        out = embed(labels.astype(jnp.int32))

        return out


# ==============================================================================
# Network
class DiffusionTransformer(nn.Module):
    r"""Diffusion Transformer (DiT) network.

    Args:
        features (int): Dimensionality of the latent feature map.
        patch_size (int): Patch size for encoding 2D images.
        depth (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
        ffn_ratio (int, optional): Expansion ratio of the hidden layer.
            Default is `4`.
        num_classes (int, optional): Number of classes for class-conditional
            diffusion. Default is `1_000`.
        class_dropout_rate (float, optional): Dropout rate for class
            conditioning. Default is `0.1`.
        learn_sigma (bool, optional): Whether to learn predicting noise scale.
            Default is `True`.
        block_type (str, optional): Type of DiT block. One of
            `['in-context', 'cross-attention', 'adaLN']`. Default is `'adaLN'`.
        dtype (Any, optional): The data type of the computation.
        param_dtype (Any, optional): The data type of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    patch_size: int
    depth: int
    num_heads: int
    ffn_ratio: int = 4
    num_classes: int = 1_000
    class_dropout_rate: float = 0.1
    learn_sigma: bool = True
    block_type: str = "adaLN"
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        timestamp: jax.Array,
        labels: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Diffusion Transformer model."""

        # patch embedding
        patch_embed = PatchEmbed(
            features=self.features,
            patch_size=self.patch_size,
            flatten=True,
            padding=False,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="patch_embed",
        )
        out = patch_embed(inputs.astype(self.dtype))

        # positional encoding
        pos_emb = sinusoidal_patch_enc(
            features=self.features,
            grid_size=out.shape[-2] ** 0.5,
            num_extra_tokens=0,
        )
        out = out + pos_emb[None, :, :].astype(self.dtype)

        # timestamp embedding
        t_embed = TimestampEmbed(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="t_embed",
        )(timestamp)
        chex.assert_equal_shape([t_embed, out[..., 0, :]])

        # label embedding with dropout
        y_embed = LabelEmbed(
            features=self.features,
            num_classes=self.num_classes,
            dropout_rate=self.class_dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="y_embed",
        )(labels=labels, deterministic=deterministic)
        chex.assert_equal_shape([y_embed, out[..., 0, :]])

        # forward pass a cascade of DiT blocks
        cond = t_embed + y_embed
        for i in range(self.depth):
            if self.block_type == "in-context":
                block = DiTConditioningBlock(
                    features=self.features,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dit_block_{i}",
                )
            elif self.block_type == "cross-attention":
                block = DiTCrossAttentionBlock(
                    features=self.features,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dit_block_{i}",
                )
            elif self.block_type == "adaLN":
                block = DiTAdaLNBlock(
                    features=self.features,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dit_block_{i}",
                )
            else:
                raise ValueError(
                    f"Invalid block_type '{self.block_type}'. Must be one of "
                    "'in-context', 'cross-attention', or 'adaLN'."
                )
            out = block(
                inputs=out,
                cond=cond,
                deterministic=deterministic,
            )

        # final decoder
        if self.block_type == "adaLN":
            decoder = AdaLNDecoder(
                features=(
                    inputs.shape[-1] * 2
                    if self.learn_sigma
                    else inputs.shape[-1]
                ),
                patch_size=self.patch_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="decoder",
            )
        else:
            decoder = StandardDecoder(
                features=(
                    inputs.shape[-1] * 2
                    if self.learn_sigma
                    else inputs.shape[-1]
                ),
                patch_size=self.patch_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="decoder",
            )
        out = decoder(out, cond=cond)

        return out
