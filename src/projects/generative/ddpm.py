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


def make_gaussian_sample_loop(
    network: nn.Module,
    alphas: jax.Array,
    alpha_bars: jax.Array,
    alpha_bar_prevs: jax.Array,
    log_posterior_vars: jax.Array,
    num_diffusion_steps: int,
    dtype: typing.Any,
) -> typing.Callable:
    r"""Wraps a JIT compiled sampling loop for Gaussian DDPM.

    Args:
        network (nn.Module): The diffusion model network used for denoising.
        alphas (jax.Array): Per-step :math:`\\alpha_t` values of shape ``(T,)``.
        alpha_bars (jax.Array): Cumulative products :math:`\\bar{\\alpha}_t`
            with a shape of ``(num_diffusion_steps,)``.
        alpha_bar_prevs (jax.Array): Previous cumulative products
            :math:`\\bar{\\alpha}_{t-1}` of shape ``(num_diffusion_steps,)``.
        log_posterior_vars (jax.Array): Log posterior variances
            :math:`\\log \\sigma_t^2` of shape ``(num_diffusion_steps,)``.
        num_diffusion_steps (int): Total number of diffusion steps.
        dtype (typing.Any): Data type to use for input and output.

    Returns:
        A JIT-compiled function that performs the sampling loop.
    """

    # pre-compute constants
    recip_sqrt_alpha_bars = jnp.sqrt(jnp.reciprocal(alpha_bars))
    recip_sqrtm1_alpha_bars = jnp.sqrt(jnp.reciprocal(alpha_bars) - 1.0)
    posterior_coef_t = jnp.true_divide(
        (1.0 - alpha_bar_prevs) * jnp.sqrt(alphas),
        1.0 - alpha_bars,
    )
    posterior_coef_start = jnp.true_divide(
        (1.0 - alphas) * jnp.sqrt(alpha_bar_prevs),
        1.0 - alpha_bars,
    )

    def _sample_loop(
        rngs: jax.Array,
        params: PyTree,
        shape: typing.Tuple[int, ...],
        deterministic: bool,
    ) -> jax.Array:
        def __body_fn(
            x_t: jax.Array,
            t: jax.Array,
        ) -> typing.Tuple[jax.Array, jax.Array]:
            noise_pred = network.apply(
                variables={"params": params},
                inputs=x_t,
                timestep=t,
                deterministic=deterministic,
            )
            x_0_rec = jnp.clip(
                recip_sqrt_alpha_bars[t] * x_t
                - recip_sqrtm1_alpha_bars[t] * noise_pred,
                -1.0,
                1.0,
            )
            x_prev = jnp.add(
                posterior_coef_t[t] * x_t,
                posterior_coef_start[t] * x_0_rec,
            )

            # optionally adding noise
            noise = jax.random.normal(
                key=jax.random.fold_in(rngs, t),
                shape=x_prev.shape,
                dtype=x_prev.dtype,
            )
            sigma_t = jnp.exp(0.5 * log_posterior_vars[t])
            mask = jnp.full(
                shape=x_prev.shape,
                fill_value=(1 - jnp.equal(t, 0)),
                dtype=noise.dtype,
            )
            x_prev = x_prev + mask * sigma_t * noise

            return x_prev, x_prev

        x_init = jax.random.normal(
            key=rngs,
            shape=shape,
            dtype=dtype,
        )
        xs = jnp.arange(0, num_diffusion_steps)
        _, out = jax.lax.scan(
            f=__body_fn,
            init=x_init,
            xs=xs,
            reverse=True,
        )

        return out

    return jax.jit(_sample_loop, static_argnames=["shape", "deterministic"])


# ==============================================================================
# Main DDPM Modules
# ==============================================================================
class DDPMUNetModule(nn.Module):
    r"""Original Denoising Diffusion Probabilistic Model (DDPM) U-Net arch.

    Attributes:
        features (int): Base number of features for the latent representations.
        ch_mults (typing.Sequence[int]): Multipliers of base number of channels
            for each level of the U-Net.
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
    ch_mults: typing.Sequence[int]
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
        t_emb = jax.nn.silu(cond_in(t_emb))
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
        t_emb = jax.nn.silu(cond_out(t_emb))

        # forward pass through U-Net
        backbone = unet.HoNetwork(
            features=self.features,
            out_features=self.out_features,
            ch_mults=self.ch_mults,
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
        ch_mults (typing.Sequence[int]): Multipliers of base number of channels
            for each level of the U-Net.
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
        model_var_type (str, optional): Type of the posterior variances. One of
            `["fixed_large", "fixed_small"]`. Default is `"fixed_large"`.
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
        ch_mults: typing.Sequence[int],
        dropout_rate: float,
        epsilon: float,
        attn_resolutions: typing.Sequence[int],
        num_res_blocks: int,
        resample_with_conv: bool = True,
        predict_variance: bool = False,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        model_var_type: str = "fixed_large",
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
            ch_mults=ch_mults,
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
        self.alpha_bars = jnp.cumprod(self.alphas, axis=0)
        self.alpha_bar_prevs = jnp.concatenate(
            [jnp.array([1.0], dtype=self.betas.dtype), self.alpha_bars[:-1]],
            axis=0,
        )
        self.recip_sqrt_alphas = jnp.sqrt(jnp.reciprocal(self.alphas))
        self.recip_sqrtm1_alphas = jnp.sqrt(jnp.reciprocal(self.alphas) - 1.0)

        # initialize posterior coefficients
        self.posterior_vars = (
            self.betas * (1.0 - self.alpha_bar_prevs) / (1.0 - self.alpha_bars)
        )
        if model_var_type == "fixed_large":
            # NOTE: discard scaling with cumulative products of alphas
            # see https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py#L142
            self.log_posterior_vars = jnp.log(
                jnp.append(self.posterior_vars[1], self.betas[1:])
            )
            self.posterior_vars = self.betas
        elif model_var_type == "fixed_small":
            self.log_posterior_vars = jnp.log(
                jnp.append(self.posterior_vars[1], self.posterior_vars[1:])
            )
        else:
            raise ValueError(f"Unsupported model_var_type: {model_var_type}")
        self.posterior_coef_t = jnp.true_divide(
            (1.0 - self.alpha_bar_prevs) * jnp.sqrt(self.alphas),
            1.0 - self.alpha_bars,
        )
        self.posterior_coef_start = jnp.true_divide(
            self.betas * jnp.sqrt(self.alpha_bar_prevs),
            1.0 - self.alpha_bars,
        )

        # initialize jit-compiled functions
        self._p_sample_loop = make_gaussian_sample_loop(
            network=self._network,
            alphas=self.alphas,
            alpha_bars=self.alpha_bars,
            alpha_bar_prevs=self.alpha_bar_prevs,
            log_posterior_vars=self.log_posterior_vars,
            num_diffusion_steps=num_diffusion_steps,
            dtype=dtype,
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

        if jax.process_index() == 0:
            tabulate_fn = nn.summary.tabulate(
                self.network,
                depth=3,
                rngs=rngs,
                console_kwargs={"width": 120},
            )
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
        flip_rng, t_rng, noise_rng, dropout_rng = jax.random.split(rngs, num=4)

        image = batch.get("image")
        if not isinstance(image, jax.Array):
            raise ValueError("Missing `image` in batch for training.")
        image = image.astype(self.dtype).reshape(-1, *image.shape[-3:])
        image = image * 2.0 - 1.0  # NOTE: scale to [-1, 1]
        # random horizontal flipping
        mask = jax.random.uniform(key=flip_rng, shape=image.shape[:-3]) < 0.5
        image = jnp.where(
            mask[..., None, None, None],
            jnp.flip(image, axis=-2),
            image,
        )

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
        w = jnp.broadcast_to(
            self.alpha_bars[timestep][:, None, None, None],
            image.shape,
        )
        samples = jnp.sqrt(w) * image + jnp.sqrt(1.0 - w) * noise_true
        noise_pred = self.network.apply(
            variables={"params": params},
            inputs=samples,
            timestep=timestep,
            deterministic=deterministic,
            rngs={"dropout": dropout_rng},
        )
        loss = jnp.mean(
            jnp.sum(
                jnp.square(noise_pred - jax.lax.stop_gradient(noise_true)),
                axis=(-1, -2, -3),
            )
        )

        out = _model.StepOutputs(
            scalars={"loss": loss},
            histograms={
                "timestep": timestep,
                "alpha_bars": self.alpha_bars[timestep],
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
        return_intermediates: bool = False,
        **kwargs,
    ) -> _model.StepOutputs:
        del kwargs  # unused

        out = self._p_sample_loop(
            rngs=rngs,
            params=params,
            shape=tuple(shape),
            deterministic=deterministic,
        )

        # scale images to [-1, 1]
        if return_intermediates:
            return _model.StepOutputs(output=out)
        else:
            # NOTE: with `reverse=True`, the final sample is at index 0
            return _model.StepOutputs(output=out[0])

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
