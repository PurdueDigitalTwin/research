import functools
import typing

from flax import linen as nn
import jax
from jax import numpy as jnp
import jaxtyping

# Type aliases
PyTree = jaxtyping.PyTree


# ==============================================================================
# Helper functions
# ==============================================================================
@functools.partial(jax.jit, static_argnames=["features"])
def sinusoidal_embedding(timesteps: jax.Array, features: int) -> jax.Array:
    r"""Encode timesteps with sinusoidal embeddings.

    Args:
        timesteps (jax.Array): Time step array of shape `(*)`.
        features (int): Dimension of the output embeddings.

    Returns:
        jax.Array: Sinusoidal embeddings of shape `(*, features)`.
    """
    batch_dims = timesteps.shape
    timesteps = jnp.reshape(timesteps, (-1,))  # flatten to 1D

    half_dim = features // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = jnp.astype(timesteps, jnp.float32)[..., None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if features % 2 == 1:
        # NOTE: pad zero vector if number of features is odd
        emb = jnp.pad(emb, pad_width=((0, 0), (0, 1)), mode="constant")
    emb = jnp.reshape(emb, batch_dims + (features,))
    return emb
