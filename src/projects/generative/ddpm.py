import functools
import typing

from flax import linen as nn
import jax
from jax import numpy as jnp
import jaxtyping
import typing_extensions

from src.core import model as _model
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


class DDPMGaussianUNetModel(_model.Model):
    r"""Denoising Deep Probabilistic Models with U-Net and Gaussian noises.

    Args:
        in_channels (int): Number of input channels.
        image_size (int): Height and width of the (squared) input images.
        features (int): Base number of features for the latent representations.
        dropout_rate (float): Dropout rate for regularization.
        epsilon (float): Small constant for numerical stability in
            the group normalization layers.
        attn_resolutions (typing.Sequence[int]): Resolutions at which to apply
            attention following the residual convolution block.
        num_res_blocks (int): Number of residual blocks per level of the U-Net.
        resample_with_conv (bool): Whether to use convolutional layers for
            upsampling and downsampling operations. Default is `True`.
        predict_variance (bool): Whether the model predicts variance along with
            the mean. If `True`, the output channels will be
            `in_channels * 2`. Default is `False`.
        dtype (typing.Any, optional): Data type for computations.
        param_dtype (typing.Any, optional): Data type for parameters.
        precision (typing.Any, optional): Numerical precision for computations.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: int,
        dropout_rate: float,
        epsilon: float,
        attn_resolutions: typing.Sequence[int],
        num_res_blocks: int,
        resample_with_conv: bool = True,
        predict_variance: bool = False,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        """Initialize the DDPM U-Net model."""
        self.in_channels = in_channels
        self.image_size = image_size
        self.features = features
        self.dtype = dtype
        self.param_dtype = param_dtype

        self._network = DDPMUNetModule(
            features=features,
            out_features=in_channels * 2 if predict_variance else in_channels,
            resample_with_conv=resample_with_conv,
            attn_resolutions=attn_resolutions,
            num_res_blocks=num_res_blocks,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            name="unet",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    @property
    @typing_extensions.override
    def network(self) -> DDPMUNetModule:
        r"""DDPMUNetModule: The underlying neural network module."""
        return self._network

    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> PyTree:
        del batch, kwargs  # unused

        variables = self.network.init(
            rngs,
            inputs=jnp.zeros(
                (1, self.image_size, self.image_size, self.in_channels),
                dtype=self.dtype,
            ),
            timestep=jnp.zeros((1,), dtype=self.dtype),
            deterministic=True,
        )
        tabulate_fn = nn.summary.tabulate(
            self.network,
            depth=3,
            rngs=rngs,
            console_kwargs={"width": 120},
        )
        if tabulate_fn is not None:
            print(
                tabulate_fn(
                    inputs=jnp.zeros(
                        (
                            1,
                            self.image_size,
                            self.image_size,
                            self.in_channels,
                        ),
                        dtype=self.dtype,
                    ),
                    timestep=jnp.zeros((1,), dtype=self.dtype),
                    deterministic=True,
                )
            )

        return variables["params"]
