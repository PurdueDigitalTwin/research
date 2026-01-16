import typing

from flax import linen as nn
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax._src import typing as jax_typing
import jaxtyping
import typing_extensions

from src.core import model as _model
from src.projects.generative.model import unet
from src.projects.generative.pipeline import augment

# Type Aliases
PyTree = jaxtyping.PyTree


# ==============================================================================
# Helper functions
# ==============================================================================
def sample_t_r(
    *,
    key: jax.Array,
    shape: jax_typing.Shape,
    dtype: typing.Any,
    distribution: str,
    **kwargs,
) -> typing.Tuple[jax.Array, jax.Array]:
    """Samples begin and end timestamps randomly from a given distribution.

    Attributes:
        key (jax.Array): JAX random key.
        shape (jax.typing.Shape): The shape of the output arrays.
        dtype (dtype): The dtype of the output arrays.
        distribution (str): The distribution to sample from.
            One of `["uniform", "logit-normal"]`.
        **kwargs: Additional keyword arguments for the distribution.

    Returns:
        Tuple[jax.Array, jax.Array]: Sampled begin timestamps `r` and
            end timestamps `t`, each of shape `shape` and dtype `dtype`.
    """
    t_key = jax.random.fold_in(key, 0)
    r_key = jax.random.fold_in(key, 1)
    if distribution == "uniform":
        minval = kwargs.get("minval", 0.0)
        maxval = kwargs.get("maxval", 1.0)
        t = jax.random.uniform(
            key=t_key,
            shape=shape,
            dtype=dtype,
            minval=minval,
            maxval=maxval,
        )
        r = jax.random.uniform(
            key=r_key,
            shape=shape,
            dtype=dtype,
            minval=minval,
            maxval=maxval,
        )
    elif distribution == "logit-normal":

        def _logit_normal(
            key: jax.Array,
            shape: jax_typing.Shape,
            dtype: typing.Any,
            mean: float,
            stddev: float,
        ) -> jax.Array:
            z = jax.random.normal(key=key, shape=shape, dtype=dtype)
            return jax.nn.sigmoid(mean + stddev * z)

        mean = kwargs.get("mean", -0.4)
        stddev = kwargs.get("stddev", 1.0)
        t = _logit_normal(
            key=t_key,
            shape=shape,
            dtype=dtype,
            mean=mean,
            stddev=stddev,
        )
        r = _logit_normal(
            key=r_key,
            shape=shape,
            dtype=dtype,
            mean=mean,
            stddev=stddev,
        )
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            'Must be one of ["uniform", "logit-normal"].'
        )

    return jnp.clip(t, 0.0, 1.0), jnp.clip(r, 0.0, 1.0)


# ==============================================================================
# Helper modules
# ==============================================================================
class SinusoidalEmbed(nn.Module):
    r"""Sinusoidal positional embeddings.

    Args:
        features (int): Dimensionality of the output embeddings.
        max_indx (int): Maximum index value.
        endpoint (bool): Whether to include the endpoint frequency.
    """

    features: int
    max_indx: int = 10_000
    endpoint: bool = False

    def setup(self) -> None:
        """Instantiate a `SinusoidalEmbed` module."""
        half_dim = self.features >> 1
        freqs = jnp.arange(0, half_dim, dtype=jnp.float32)
        freqs = freqs / (half_dim - (1 if self.endpoint else 0))
        self.freqs = jnp.power(1.0 / self.max_indx, freqs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass and returns the sinusoidal embeddings.

        Args:
            inputs (jax.Array): Input indexes of shape `(*, )`.

        Returns:
            Sinusoidal embedding array of shape `(..., features)`.
        """
        out = jnp.outer(inputs[..., None], self.freqs)
        out = jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)
        return out


class TimestampEmbed(nn.Module):
    """Encode scalar timestamps to vectors.

    Attributes:
        features (int): Dimensionality of the output embeddings.
        frequency (int): Frequency of the sinusoidal embeddings.
        max_stamp (int): Maximum timestamp value.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
    """

    features: int
    """int: Dimensionality of the output embeddings."""
    frequency: int = 256
    """int: Frequency of the sinusoidal embeddings."""
    max_stamp: int = 10_000
    """int: Maximum timestamp value."""
    dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the computation."""
    param_dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the parameters."""

    def setup(self) -> None:
        """Instantiate a `TimestampEmbedding` module."""
        self.fc_in = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=0.02,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            name="fc_in",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.fc_out = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=0.02,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            name="fc_out",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    @staticmethod
    def _embed(
        t: jax.Array, frequency: int, max_stamp: int, dtype: typing.Any
    ) -> jax.Array:
        """Embeds timestamps using sinusoidal functions."""
        batch_dims = t.shape[:-1]
        half_dim = frequency // 2
        freqs = jnp.exp(
            -jnp.log(max_stamp)
            * jnp.arange(start=0, stop=half_dim, dtype=dtype)
            / half_dim
        )  # shape: (half_dim,)
        freqs = jnp.expand_dims(freqs, list(range(-len(batch_dims) - 1, -1)))
        embed = t[..., None] * freqs
        embed = jnp.concatenate((jnp.cos(embed), jnp.sin(embed)), axis=-1)
        if frequency % 2 == 1:
            # NOTE: zero pad if frequency is odd
            embed = jnp.concatenate(
                (embed, jnp.zeros_like(embed[..., :1])),
                axis=-1,
            )

        return embed

    def __call__(self, t: jax.Array) -> jax.Array:
        """Forward pass the timestamp encoder.

        Args:
            t (jax.Array): Scalar timestamps of shape `(*, 1)`.

        Returns:
            jax.Array: Timestamp embeddings of shape `(..., features)`.
        """
        embedding = self._embed(t, self.frequency, self.max_stamp, self.dtype)
        embedding = self.fc_in(embedding)
        embedding = jax.nn.silu(embedding)
        embedding = self.fc_out(embedding)
        return embedding


# ==============================================================================
# Main modules
# ==============================================================================
class MeanFlowUNetModule(nn.Module):
    """Generative model with a RefineNet backbone trained with `MeanFlow`.

    Attributes:
        features (int): Number of channels in the latent feature maps.
        dropout_rate (float): Dropout rate for the attention blocks.
        epsilon (float): Small constant for numerical stability in `GroupNorm`.
        skip_scale (float): Scaling factor for skip connections.
        resample_filter (Optional[Sequence[int]]): One-dimensional FIR
            filter for up/downsampling. Default is :math:`[1, 1]`.
        deterministic (Optional[bool]): Whether to run deterministically.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
    """

    features: int
    dropout_rate: float
    epsilon: float
    skip_scale: float
    resample_filter: typing.Sequence[int] = (1, 1)
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        image: jax.Array,
        timestamps: typing.Tuple[jax.Array],
        edm_cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass the `MeanFlowUNetModel`.

        Args:
            inputs (jax.Array): Input images of shape `(*, H, W, C)`.
            timestamps (Tuple[jax.Array, ...]): Timestamps of shape `(*, 1)`.
            edm_cond (jax.Array, optional): Conditioning embeddings for
                EDM data augmentation of shape `(*, 6)`.
            deterministic (bool, optional): Whether to run deterministically.

        Returns:
            The predicted average velocity of shape `(*, H, W, C)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # encode the conditions
        time_embed = SinusoidalEmbed(self.features * 2, endpoint=True)
        emb = [time_embed(time) for time in timestamps]
        cond = jnp.concatenate(emb, axis=-1)

        if edm_cond is not None:
            aug_embed = nn.Dense(
                features=cond.shape[-1],
                use_bias=False,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="aug_fc",
            )
            aug_cond = aug_embed(edm_cond)
            cond = cond + aug_cond

        # projects the conditioning embeddings
        cond_in = nn.Dense(
            features=self.features * 4,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cond_fc_1",
        )
        cond = jax.nn.silu(cond_in(cond))
        cond_out = nn.Dense(
            features=self.features * 4,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cond_fc_2",
        )
        cond = jax.nn.silu(cond_out(cond))

        # pass through the backbone U-Net
        backbone = unet.SongNetwork(
            features=self.features,
            ch_mults=[2, 2, 2],
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            skip_scale=self.skip_scale,
            resample_filter=self.resample_filter,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="backbone",
        )
        output = backbone(
            inputs=image,
            cond=cond,
            deterministic=m_deterministic,
        )

        return output


class MeanFlowUNetModel(_model.Model):
    r"""`MeanFlow` generative model with a U-Net backbone.

    Args:
        in_channels (int): Number of input image channels.
        image_size (int): Height and width of the input images.
        features (int): Dimensionality of the latent feature map.
        dropout_rate (float): Dropout rate for the classifier-free guidance.
        resample_filter (typing.Sequence[float | int]): One-dimensional FIR
            filter for up/downsampling. Default is :math:`[1, 1]`.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
        timestamp_cond (Literal): The type of timestamp conditioning.
            One of `["t_and_r", "t_and_t_minus_r",
            "t_and_r_and_t_minus_r", "t_minus_r"]`.
        timestamp_sampler (str): The distribution to sample timestamps from.
            One of `["uniform", "logit-normal"]`.
        timestamp_sampler_kwargs (Dict[str, Any]): Additional keyword arguments
            for the timestamp sampler.
        timestamp_overlap_rate (float): The minimum overlap rate between
            begin and end timestamps.
        adaptive_weight_power (float): The power for adaptive weight scaling.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: int,
        dropout_rate: float,
        epsilon: float = 1e-6,
        skip_scale: float = 1.0,
        resample_filter: typing.Sequence[int] = [1, 1],
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
        timestamp_cond: typing.Literal[
            "t_and_r",
            "t_and_t_minus_r",
            "t_and_r_and_t_minus_r",
            "t_minus_r",
        ] = "t_and_t_minus_r",
        timestamp_sampler: str = "logit-normal",
        timestamp_sampler_kwargs: typing.Dict[str, typing.Any] = {
            "mean": -0.4,
            "stddev": 1.0,
        },
        timestamp_overlap_rate: float = 0.75,
        adaptive_weight_power: float = 1.0,
    ) -> None:
        """Initializes the `MeanFlow` model."""
        self.in_channels = in_channels
        self.image_size = image_size
        self.features = features
        self.timestamp_cond = timestamp_cond
        self.timestamp_sampler = timestamp_sampler
        self.timestamp_sampler_kwargs = timestamp_sampler_kwargs
        self.timestamp_overlap_rate = timestamp_overlap_rate
        self.adaptive_weight_power = adaptive_weight_power
        self._augment = augment.EDMAugmentor(
            image_size=(image_size, image_size),
            p=0.12,
            xflip=1e8,
            yflip=0,
            scale=1,
            rotate_frac=0,
            aniso=1,
            translate_frac=1,
        )
        self._network = MeanFlowUNetModule(
            features=features,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            skip_scale=skip_scale,
            resample_filter=resample_filter,
            name="unet",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    @property
    @typing_extensions.override
    def network(self) -> MeanFlowUNetModule:
        r"""MeanFlowUNetModule: The U-Net neural network module."""
        return self._network

    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> PyTree:
        del batch  # unused

        # create dummy inputs
        if self.timestamp_cond in ["t_and_r", "t_and_t_minus_r"]:
            timestamps = (
                jnp.zeros((1,), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
            )
        elif self.timestamp_cond == "t_and_r_and_t_minus_r":
            timestamps = (
                jnp.zeros((1,), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
            )
        elif self.timestamp_cond == "t_minus_r":
            timestamps = (jnp.zeros((1,), dtype=jnp.float32),)
        else:
            raise ValueError(
                f"Unsupported timestamp conditioning: {self.timestamp_cond}."
            )

        dummy_inputs = {
            "image": jnp.zeros(
                (1, self.image_size, self.image_size, self.in_channels),
                dtype=jnp.float32,
            ),
            "timestamps": timestamps,
            "edm_cond": jnp.zeros((1, 6), dtype=jnp.float32),
        }
        variables = self.network.init(
            rngs=rngs,
            image=dummy_inputs["image"],
            timestamps=dummy_inputs["timestamps"],
            edm_cond=dummy_inputs["edm_cond"],
            deterministic=True,
        )
        _tabulate_fn = nn.summary.tabulate(
            self.network,
            depth=3,
            rngs=rngs,
            console_kwargs={"width": 120},
        )

        # log the model summary only on process 0
        if jax.process_index() == 0:
            print(_tabulate_fn(**dummy_inputs, deterministic=True))

        return variables["params"]

    @typing_extensions.override
    def compute_loss(
        self,
        *,
        rngs: typing.Any,
        batch: typing.Dict[str, typing.Any],
        params: frozen_dict.FrozenDict,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        r"""Computes the loss given parameters and model inputs.

        Args:
            rngs (Union[jax.random.KeyArray, Dict[str, jax.random.KeyArray]]):
                JAX random key or a dictionary of JAX random keys.
            batch (Dict[str, Any]): A batch of data containing:
                - image (jax.Array): Input images of shape `(*, H, W, C)`.
            params (frozen_dict.FrozenDict): The model parameters.
            deterministic (bool): Whether to run the model deterministically.
            **kwargs: additional keyword arguments.

        Returns:
            The computed loss and other outputs.
        """
        del kwargs  # unused

        # NOTE: enforce float32 for training stability using `jax.jvp`
        image = batch["image"].astype(jnp.float32)
        assert isinstance(image, jax.Array)
        batch_dims = image.shape[:-3]
        tr_rng, dropout_rng, a_rng, m_rng, e_rng = jax.random.split(rngs, 5)

        # pre-process the inputs
        image = image * 2.0 - 1.0
        if not deterministic:
            image, cond = self._augment.apply(
                variables={},
                images=image,
                rngs={"augment": a_rng},
            )
            assert isinstance(image, jax.Array)
            assert isinstance(cond, jax.Array)
        else:
            cond = None

        # NOTE: following the notation in Algorithm 1 of the source paper
        # sample begin timestep r and end timestep t.
        t, r = sample_t_r(
            key=tr_rng,
            shape=batch_dims,
            dtype=image.dtype,
            distribution=self.timestamp_sampler,
            **self.timestamp_sampler_kwargs,
        )
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        # ensure a portion of overlap between t and r
        # NOTE: the following code randomly mask by uniform samples
        r_eq_t_mask = jnp.less(
            jax.random.uniform(key=m_rng, shape=batch_dims, dtype=image.dtype),
            self.timestamp_overlap_rate,
        )
        r = jnp.where(r_eq_t_mask, t, r)

        # sample e ~ N(0, I)
        e = jax.random.normal(key=e_rng, shape=image.shape, dtype=image.dtype)

        # generate z_{t}
        z = jnp.add(
            (1 - t[..., None, None, None]) * image,
            t[..., None, None, None] * e,
        )
        v = e - image

        # applies Jacobian vector product
        def u_fn(
            z_t: jax.Array,
            r_in: jax.Array,
            t_in: jax.Array,
        ) -> jax.Array:
            if self.timestamp_cond == "t_and_r":
                timestamps = (t_in, r_in)
            elif self.timestamp_cond == "t_and_t_minus_r":
                timestamps = (t_in, t_in - r_in)
            elif self.timestamp_cond == "t_and_r_and_t_minus_r":
                timestamps = (t_in, r_in, t_in - r_in)
            elif self.timestamp_cond == "t_minus_r":
                timestamps = (t_in - r_in,)
            else:
                raise ValueError(
                    f"Unsupported timestamp conditioning: {self.timestamp_cond}."
                )

            out = self.network.apply(
                variables={"params": params},
                image=z_t,
                timestamps=timestamps,
                edm_cond=cond,
                deterministic=deterministic,
                rngs={"dropout": dropout_rng},
            )
            assert isinstance(out, jax.Array)

            return out

        # NOTE: following the original meanflow
        drdt = jnp.zeros_like(r)
        dtdt = jnp.ones_like(t)
        u, dudt = jax.jvp(u_fn, (z, r, t), (v, drdt, dtdt))
        u_target = v - (t - r)[..., None, None, None] * dudt

        # NOTE: following the symmetric meanflow
        # drdt = jnp.ones_like(r)
        # dtdt = jnp.negative(jnp.ones_like(t))
        # u, dudt = jax.jvp(u_fn, (z, r, t), (-v, drdt, dtdt))
        # u_target = jax.lax.stop_gradient(
        #     v
        #     - jnp.clip(t - r, a_min=0.0, a_max=1.0)[..., None, None, None]
        #     * dudt
        #     * 0.5
        # )

        # computes the target
        # NOTE: sum over all the pixels, following official implementation
        loss = jnp.sum(
            jnp.square(u - jax.lax.stop_gradient(u_target)),
            axis=(-1, -2, -3),
        )

        # applies adaptive weight power
        if self.adaptive_weight_power > 0.0:
            ada_wt = jnp.power(loss + 1e-3, self.adaptive_weight_power)
            loss = loss / jax.lax.stop_gradient(ada_wt)
        loss = jnp.mean(loss)

        # calculate velocity loss for monitoring
        velocity_loss = jnp.where(
            jnp.equal(t, r)[..., None, None, None],
            jnp.square(u - v),
            jnp.zeros_like(u),
        )
        velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

        out = _model.StepOutputs(
            scalars={"loss": loss, "velocity_loss": velocity_loss},
            histograms={"t": t, "r": r, "t - r": t - r},
        )

        return loss, out

    @typing_extensions.override
    def forward(
        self,
        *,
        rngs: jax.Array,
        params: frozen_dict.FrozenDict,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        deterministic: bool = True,
        **kwargs,
    ) -> _model.StepOutputs:
        r"""Forward sampling with average velocity prediction.

        Args:
            rngs (jax.Array): Random key for sampling.
            params (frozen_dict.FrozenDict): The model parameters.
            shape (typing.Sequence[typing.Union[int, typing.Any]]): The shape
                of the generated samples, including batch size.
            deterministic (bool): Whether to run the model deterministically.
            **kwargs: Additional keyword arguments.

        Returns:
            The output samples.
        """
        del kwargs  # unused

        z_1 = jax.random.normal(
            key=rngs,
            shape=shape,
            dtype=self.network.dtype,
        )
        r = jnp.zeros(z_1.shape[:-3], dtype=z_1.dtype)
        t = jnp.ones(z_1.shape[:-3], dtype=z_1.dtype)
        if self.timestamp_cond == "t_and_r":
            timestamps = (t, r)
        elif self.timestamp_cond == "t_and_t_minus_r":
            timestamps = (t, t - r)
        elif self.timestamp_cond == "t_and_r_and_t_minus_r":
            timestamps = (t, r, t - r)
        elif self.timestamp_cond == "t_minus_r":
            timestamps = (t - r,)
        else:
            raise ValueError(
                f"Unsupported timestamp conditioning: {self.timestamp_cond}."
            )

        out = z_1 - self.network.apply(
            variables={"params": params},
            image=z_1,
            timestamps=timestamps,
            edm_cond=None,
            deterministic=deterministic,
        )

        return _model.StepOutputs(output=out)
