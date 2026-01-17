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
        beta_start (float, optional): Starting value of beta for the noise
            schedule. Default is `0.0001`.
        beta_end (float, optional): Ending value of beta for the noise schedule.
            Default is `0.02`.
        beta_schedule (str, optional): Type of beta schedule. One of
            `["linear", "quad", "const", "jsd"]`. Default is `"linear"`.
        num_diffusion_steps (int, optional): Number of diffusion steps.
            Default is `1,000`.
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
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        num_diffusion_steps: int = 1_000,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        """Initialize the DDPM U-Net model."""
        self.in_channels = in_channels
        self.image_size = image_size
        self.features = features
        self.num_diffusion_steps = num_diffusion_steps
        self.dtype = dtype
        self.param_dtype = param_dtype

        # U-Net backbone
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

        # initialize the beta schedule
        self.betas = self.get_betas(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_diffusion_steps=num_diffusion_steps,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_bars = jnp.cumprod(self.alphas, axis=0)
        self.alpha_bar_prevs = jnp.concatenate(
            [jnp.array([1.0], dtype=self.betas.dtype), self.alphas_bars[:-1]],
            axis=0,
        )

        # initialize posterior coefficients
        self.posterior_vars = (
            self.betas
            * (1.0 - self.alpha_bar_prevs)
            / (1.0 - self.alphas_bars)
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

    @typing_extensions.override
    def compute_loss(
        self,
        *,
        rngs: typing.Any,
        batch: PyTree,
        params: PyTree,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        del kwargs  # unused

        image = batch.get("image")
        if not isinstance(image, jax.Array):
            raise ValueError("Missing `image` in batch for training.")
        image = image.astype(self.dtype).reshape(-1, *image.shape[-3:])
        image = image * 2.0 - 1.0  # NOTE: scale to [-1, 1]

        rngs, t_rng, noise_rng, dropout_rng = jax.random.split(rngs, num=4)
        timestep = jax.random.randint(
            key=t_rng,
            shape=image.shape[:-3],
            minval=0,
            maxval=self.num_diffusion_steps,
        )

        # draw sample from the posterior `q(x_{t} | x_{0})`
        noise_true = jax.random.normal(
            key=noise_rng,
            shape=image.shape,
            dtype=image.dtype,
        )
        samples = jnp.add(
            jnp.sqrt(self.alphas_bars[timestep])[:, None, None, None] * image,
            jnp.sqrt(1.0 - self.alphas_bars[timestep])[:, None, None, None]
            * noise_true,
        )
        noise_pred = self.network.apply(
            variables={"params": params},
            inputs=samples,
            timestep=timestep,
            deterministic=deterministic,
            rngs={"dropout": dropout_rng},
        )
        loss = jnp.mean(
            jnp.square(noise_pred - jax.lax.stop_gradient(noise_true))
        )

        out = _model.StepOutputs(
            scalars={"loss": loss},
            histograms={
                "timestep": timestep,
                "alpha_bars": self.alphas_bars[timestep],
                "alphas": self.alphas[timestep],
                "betas": self.betas[timestep],
            },
        )

        return loss, out

    @typing_extensions.override
    def forward(
        self,
        *,
        rngs: typing.Any,
        params: PyTree,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        deterministic: bool = True,
        **kwargs,
    ) -> _model.StepOutputs:
        del kwargs  # unused

        def _scan_body(
            carry: jax.Array,
            x: jax.Array,
        ) -> typing.Tuple[jax.Array, jax.Array]:
            noise_pred = self.network.apply(
                variables={"params": params},
                inputs=carry,
                timestep=x,
                deterministic=deterministic,
            )
            _coef_0 = 1.0 / jnp.sqrt(self.alphas[x])
            _coef_1 = self.betas[x] / jnp.sqrt(1.0 - self.alphas_bars[x])
            x_t = _coef_0 * carry - _coef_1 * noise_pred

            # optionally adding noise
            noise = jnp.where(
                jnp.full(carry.shape, x > 0, dtype=jnp.bool_),
                jax.random.normal(
                    key=jax.random.fold_in(rngs, x),
                    shape=carry.shape,
                    dtype=carry.dtype,
                ),
                jnp.zeros_like(carry),
            )
            sigma_t = jnp.sqrt(self.posterior_vars[x])
            x_t += sigma_t * noise

            # scale images to [-1, 1]
            x_t = jnp.clip(x_t, -1.0, 1.0)

            return x_t, x_t

        init = jax.random.normal(
            key=rngs,
            shape=shape,
            dtype=self.dtype,
        )
        xs = jnp.arange(self.num_diffusion_steps - 1, -1, -1)
        _, samples = jax.lax.scan(
            f=_scan_body,
            init=init,
            xs=xs,
        )

        return _model.StepOutputs(output=samples[-1, ...])

    @staticmethod
    def get_betas(
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        num_diffusion_steps: int,
    ) -> jax.Array:
        if beta_schedule == "quad":
            return jnp.square(
                jnp.linspace(
                    jnp.sqrt(beta_start),
                    jnp.sqrt(beta_end),
                    num_diffusion_steps,
                )
            )
        elif beta_schedule == "linear":
            return jnp.linspace(
                beta_start,
                beta_end,
                num_diffusion_steps,
            )
        elif beta_schedule == "const":
            return jnp.full((num_diffusion_steps,), beta_end)
        elif beta_schedule == "jsd":
            return jnp.reciprocal(
                jnp.linspace(
                    num_diffusion_steps,
                    1,
                    num_diffusion_steps,
                )
            )
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
