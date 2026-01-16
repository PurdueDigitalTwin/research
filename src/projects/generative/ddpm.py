import functools
import typing

from flax import linen as nn
import jax
from jax import numpy as jnp
import jaxtyping

from src.projects.generative.model import unet

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


# ==============================================================================
# Main DDPM Modules
# ==============================================================================
class DDPMUNetModule(nn.Module):
    r"""Original Denoising Diffusion Probabilistic Model (DDPM) U-Net arch.

    Attributes:
        features (int): Base number of features for the latent representations.
        attn_resolutions (typing.Sequence[int]): Resolutions at which to apply
            attention following the residual convolution block.
        num_res_blocks (int): Number of residual blocks per level of the U-Net.
        resample_with_conv (bool): Whether to use convolutional layers for
            upsampling and downsampling operations.
        dropout_rate (float): Dropout rate for regularization.
        epsilon (float): Small constant for numerical stability in
            the group normalization layers.
        out_features (int, optional): Number of output features. If `None`,
            the number of input features is used. Default is `None`.
        deterministic (bool, optional): Whether to apply dropout.
        dtype (typing.Any, optional): Data type for computations.
        param_dtype (typing.Any, optional): Data type for parameters.
        precision (typing.Any, optional): Numerical precision for computations.
    """

    features: int
    attn_resolutions: typing.Sequence[int]
    num_res_blocks: int
    resample_with_conv: bool
    dropout_rate: float
    epsilon: float
    out_features: typing.Optional[int] = None
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        timestep: jax.Array,
        deterministic: typing.Optional[bool] = None,
        **kwargs,
    ) -> jax.Array:
        r"""Forward pass of the `DDPMUNetModule`.

        Args:
            inputs (jax.Array): Input data of shape `(*, H, W, C)`.
            timestep (jax.Array): Time step array of shape `(*)`.
            deterministic (typing.Optional[bool]): Whether to apply dropout.
                Merges with the module level attribute `deterministic`.

        Returns:
            Output data of shape `(*, H, W, C)`.
        """
        del kwargs  # unused
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # encode timestep embedding
        t_emb = sinusoidal_embedding(timestep, self.features)
        cond_in = nn.Dense(
            features=self.features * 4,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="cond_fc_1",
        )
        cond_out = nn.Dense(
            features=self.features * 4,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="cond_fc_2",
        )
        t_emb = cond_out(jax.nn.silu(cond_in(t_emb)))

        # forward pass through U-Net
        backbone = unet.HoNetwork(
            features=self.features,
            out_features=self.out_features,
            ch_mults=[1, 2, 2, 2],
            attn_resolutions=self.attn_resolutions,
            num_res_blocks=self.num_res_blocks,
            resample_with_conv=self.resample_with_conv,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="backbone",
        )
        output = backbone(inputs, cond=t_emb, deterministic=m_deterministic)

        return output
