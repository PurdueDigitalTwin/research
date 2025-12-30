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

        q_scaled = q / jnp.sqrt(jnp.sqrt(q.shape[-1]).astype(self.dtype))
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


class _DiTConditioningBlock(nn.Module):
    r"""Diffusion Transformers (DiT) block with in-context conditioning.

    Args:
    """

    @nn.compact
    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        raise NotImplementedError


class _DiTCrossAttentionBlock(nn.Module):
    r"""Diffusion Transformers (DiT) block with cross-attention."""

    @nn.compact
    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        raise NotImplementedError


class _DiTAdaLNBlock(nn.Module):
    r"""Diffusion Transformers (DiT) block with adaptive layer normalization."""

    @nn.compact
    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        raise NotImplementedError
