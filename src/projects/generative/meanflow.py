import typing

from flax import linen as nn
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax._src import typing as jax_typing
import jaxtyping
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
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
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
        precision (Any): Numerical precision for the computation.
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
        inputs: jax.Array,
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
            inputs=inputs,
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
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
        precision (Any): Numerical precision for the computation.
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
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
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
        variables = self._network.init(
            rngs=rngs,
            inputs=dummy_inputs["image"],
            timestamps=dummy_inputs["timestamps"],
            edm_cond=dummy_inputs["edm_cond"],
            deterministic=True,
        )

        # log the model summary only on process 0
        if jax.process_index() == 0:
            _tabulate_fn = nn.summary.tabulate(
                self._network,
                depth=3,
                rngs=rngs,
                console_kwargs={"width": 120},
            )
            print(_tabulate_fn(**dummy_inputs, deterministic=True))

        params = variables.pop("params")

        return params, variables

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: typing.Any,
        state: _train_state.TrainState,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        # NOTE: enforce float32 for training stability using `jax.jvp`
        image = batch["image"].astype(jnp.float32)
        assert isinstance(image, jax.Array)
        batch_dims = image.shape[:-3]
        tr_rng, dropout_rng, a_rng, m_rng, e_rng = jax.random.split(rngs, 5)

        # pre-process the inputs
        image = image * 2.0 - 1.0
        image, cond = self._augment.apply(
            variables={},
            images=image,
            rngs={"augment": a_rng},
        )
        assert isinstance(image, jax.Array)
        assert isinstance(cond, jax.Array)

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

        def _loss_fn(params: PyTree) -> typing.Tuple[jax.Array, jax.Array]:
            # applies Jacobian vector product
            def u_fn(
                z_t: jax.Array,
                r_in: jax.Array,
                t_in: jax.Array,
            ) -> jax.Array:
                timestamps = self._make_timestamps(t_in=t_in, r_in=r_in)
                out = self._network.apply(
                    variables={"params": params},
                    inputs=z_t,
                    timestamps=timestamps,
                    edm_cond=cond,
                    deterministic=False,
                    rngs={"dropout": dropout_rng},
                    **kwargs,
                )
                assert isinstance(out, jax.Array)

                return out

            # NOTE: following the original meanflow
            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            v = e - image
            u, dudt = jax.jvp(u_fn, (z, r, t), (v, drdt, dtdt))
            u_target = v - (t - r)[..., None, None, None] * dudt

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
                jnp.square(u - (e - image)),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            return loss, velocity_loss

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, velocity_loss), grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(
            scalars={
                "loss": loss.mean(),
                "velocity_loss": velocity_loss.mean(),
            },
            histograms={"t": t, "r": r, "t - r": t - r},
        )

        return new_state, outputs

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
            dtype=self._network.dtype,
        )
        timestamps = self._make_timestamps(
            t_in=jnp.ones(z_1.shape[:-3], dtype=jnp.float32),
            r_in=jnp.zeros(z_1.shape[:-3], dtype=jnp.float32),
        )

        out = z_1 - self._network.apply(
            variables={"params": params},
            inputs=z_1,
            timestamps=timestamps,
            edm_cond=None,
            deterministic=deterministic,
        )

        return _model.StepOutputs(output=out)

    def _make_timestamps(
        self,
        t_in: jax.Array,
        r_in: jax.Array,
    ) -> typing.Tuple[jax.Array, ...]:
        """Constructs timestamp tuple from (t, r).

        Args:
            t_in (jax.Array): Terminal timesteps.
            r_in (jax.Array): Start timesteps.

        Returns:
            Tuple of timestamp arrays for the network.
        """
        if self.timestamp_cond == "t_and_r":
            return (t_in, r_in)
        elif self.timestamp_cond == "t_and_t_minus_r":
            return (t_in, t_in - r_in)
        elif self.timestamp_cond == "t_and_r_and_t_minus_r":
            return (t_in, r_in, t_in - r_in)
        elif self.timestamp_cond == "t_minus_r":
            return (t_in - r_in,)
        else:
            raise ValueError(
                "Unsupported timestamp conditioning: "
                f"{self.timestamp_cond}."
            )


class ImprovedMeanFlowUNetModel(MeanFlowUNetModel):
    r"""Implementation of improved MeanFlow model.

    .. note::

        This is a customized implementation of improved MeanFlow algorithm
        presented in ``https://arxiv.org/abs/2512.02012``.

    Args:
        in_channels (int): Number of input image channels.
        image_size (int): Height and width of the input images.
        features (int): Dimensionality of the latent feature map.
        dropout_rate (float): Dropout rate for the classifier-free guidance.
        resample_filter (typing.Sequence[float | int]): One-dimensional FIR
            filter for up/downsampling. Default is :math:`[1, 1]`.
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
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
        precision (Any): Numerical precision for the computation.
    """

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: typing.Any,
        state: _train_state.TrainState,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        # NOTE: enforce float32 for training stability using `jax.jvp`
        image = batch["image"].astype(jnp.float32)
        assert isinstance(image, jax.Array)
        batch_dims = image.shape[:-3]
        tr_rng, dropout_rng, a_rng, m_rng, e_rng = jax.random.split(rngs, 5)

        # pre-process the inputs
        image = image * 2.0 - 1.0
        image, cond = self._augment.apply(
            variables={},
            images=image,
            rngs={"augment": a_rng},
        )
        assert isinstance(image, jax.Array)
        assert isinstance(cond, jax.Array)

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

        def _loss_fn(params: PyTree) -> typing.Tuple[jax.Array, jax.Array]:
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

                u_out = self._network.apply(
                    variables={"params": params},
                    inputs=z_t,
                    timestamps=timestamps,
                    edm_cond=cond,
                    deterministic=False,
                    rngs={"dropout": dropout_rng},
                    **kwargs,
                )
                assert isinstance(u_out, jax.Array)

                return u_out

            # NOTE: evaluate meanflow identity with bootstrapping
            # This exploits the boundary condition as described in section 4.1
            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            v_target = jax.lax.stop_gradient(e - image)
            v_marg = u_fn(z_t=z, r_in=t, t_in=t)

            u, dudt = jax.jvp(u_fn, (z, r, t), (v_marg, drdt, dtdt))
            v_pred = jnp.add(
                u, (t - r)[..., None, None, None] * jax.lax.stop_gradient(dudt)
            )

            # NOTE: sum over all the pixels, following official implementation
            loss = jnp.sum(jnp.square(v_pred - v_target), axis=(-1, -2, -3))
            if self.adaptive_weight_power > 0.0:
                ada_wt = jnp.power(loss + 1e-2, self.adaptive_weight_power)
                loss = loss / jax.lax.stop_gradient(ada_wt)
            loss = jnp.mean(loss)

            # calculate velocity loss for monitoring
            velocity_loss = jnp.where(
                jnp.equal(t, r)[..., None, None, None],
                jnp.square(u - (e - image)),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            return loss, velocity_loss

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, velocity_loss), grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(
            scalars={
                "loss": loss.mean(),
                "velocity_loss": velocity_loss.mean(),
            },
            histograms={"t": t, "r": r, "t - r": t - r},
        )

        return new_state, outputs


# ==============================================================================
# Variance-Aware Mean Flows
# ==============================================================================
class VAMeanFlowUNetModule(nn.Module):
    r"""Generative model with a NCSN++ backbone trained with ``VaMeanFlow``.

    Attributes:
        features (int): Number of channels in the latent feature maps.
        dropout_rate (float): Dropout rate for the attention blocks.
        epsilon (float): Small constant for numerical stability in `GroupNorm`.
        skip_scale (float): Scaling factor for skip connections.
        predict_variance (bool, optional): Whether to predict the variance of
            the average velocity. Default is ``False``.
        resample_filter (Optional[Sequence[int]], optional): One-dimensional FIR
            filter for up/downsampling. Default is :math:`[1, 1]`.
        deterministic (Optional[bool]): Whether to run deterministically.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
        precision (Any): Numerical precision for the computation.
    """

    features: int
    dropout_rate: float
    epsilon: float
    skip_scale: float
    predict_variance: bool = False
    resample_filter: typing.Sequence[int] = (1, 1)
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        timestep: typing.Tuple[jax.Array, ...],
        edm_cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> typing.Tuple[jax.Array, typing.Optional[jax.Array]]:
        r"""Forward pas the ``VAMeanFlowUNetModule``.

        Args:
            inputs (jax.Array): Input data of shape ``(*, D1, D2, ..., C)``.
            timestep (Tuple[jax.Array, ...]): Time steps of shape ``(*, 1)``.
            edm_cond (jax.Array, optional): Conditioning embeddings for
                EDM data augmentation of shape ``(*, 6)``.
            deterministic (bool, optional): Whether to run deterministically.

        Returns:
            The predicted average velocity field and optionally the variance
            of the average velocity field if ``predict_variance=True``.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # encode the time step conditions
        time_embed = SinusoidalEmbed(self.features * 2, endpoint=True)
        embed = [time_embed(t) for t in timestep]
        cond = jnp.concatenate(embed, axis=-1)

        if edm_cond is not None:
            aug_cond = nn.Dense(
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
            )(edm_cond)
            cond = cond + aug_cond

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
        h = backbone(
            inputs=inputs,
            cond=cond,
            deterministic=m_deterministic,
            with_head=False,
        )

        # shared normalization + activation
        norm_out = nn.GroupNorm(
            num_groups=32,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm_out",
        )
        h_act = jax.nn.silu(norm_out(h))

        # mean head: Conv(C_in, 3x3) -> average velocity
        conv_out = nn.Conv(
            features=inputs.shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            name="conv_out",
        )
        u = conv_out(h_act)

        # variance head: spatial pool -> Dense(1) -> scalar
        if self.predict_variance:
            var_head = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="var_head",
            )
            log_var = var_head(h).squeeze(-1)
        else:
            log_var = None

        return u, log_var


class VAMeanFlowUNetModel(MeanFlowUNetModel):
    r"""Variance-Aware MeanFlow with EMA tangent and flow-matching anchor.

    Args:
        in_channels (int): Number of input image channels.
        image_size (int): Height and width of the input images.
        features (int): Dimensionality of the latent feature map.
        dropout_rate (float): Dropout rate.
        epsilon (float): GroupNorm epsilon.
        skip_scale (float): Skip connection scaling factor.
        resample_filter (Sequence[int]): FIR filter for resampling.
        timestamp_cond (str): Timestamp conditioning type.
        timestamp_sampler (str): Distribution for timestamp sampling.
        timestamp_sampler_kwargs (Dict): Kwargs for the sampler.
        timestamp_overlap_rate (float): Overlap rate between t and r.
        adaptive_weight_power (float): Power for adaptive weighting.
            Set to 0 when using the NLL variant.
        fm_anchor_weight (float): Weight for FM anchor loss.
        fm_anchor_delta_min (float): Min interval for FM anchor.
        fm_anchor_delta_max (float): Max interval for FM anchor.
        predict_variance (bool): Enable heteroscedastic variance
            head (NLL variant).
        variance_floor (float): Minimum variance to prevent
            collapse.
        nll_warmup_steps (int): Steps of MSE before NLL activation.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
        precision (Any): Numerical precision.
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
        fm_anchor_weight: float = 0.5,
        fm_anchor_delta_min: float = 1e-4,
        fm_anchor_delta_max: float = 0.01,
        predict_variance: bool = False,
        variance_floor: float = 1e-4,
        nll_warmup_steps: int = 10_000,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        r"""Instantiate a ``VAMeanFlowUNetModel``."""
        super().__init__(
            in_channels=in_channels,
            image_size=image_size,
            features=features,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            skip_scale=skip_scale,
            resample_filter=resample_filter,
            timestamp_cond=timestamp_cond,
            timestamp_sampler=timestamp_sampler,
            timestamp_sampler_kwargs=timestamp_sampler_kwargs,
            timestamp_overlap_rate=timestamp_overlap_rate,
            adaptive_weight_power=adaptive_weight_power,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.fm_anchor_weight = fm_anchor_weight
        self.fm_anchor_delta_min = fm_anchor_delta_min
        self.fm_anchor_delta_max = fm_anchor_delta_max
        self.predict_variance = predict_variance
        self.variance_floor = variance_floor
        self.nll_warmup_steps = nll_warmup_steps

        # override network with variance-aware version
        self._network = VAMeanFlowUNetModule(
            features=features,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            skip_scale=skip_scale,
            resample_filter=resample_filter,
            predict_variance=predict_variance,
            name="unet",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    @typing_extensions.override
    def forward(
        self,
        *,
        rngs: typing.Any,
        params: typing.Any,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        deterministic: bool = True,
        **kwargs,
    ) -> _model.StepOutputs:
        del kwargs  # unused
        z_1 = jax.random.normal(
            key=rngs,
            shape=shape,
            dtype=jnp.float32,
        )
        timestamps = self._make_timestamps(
            t_in=jnp.ones(z_1.shape[:-3], dtype=jnp.float32),
            r_in=jnp.zeros(z_1.shape[:-3], dtype=jnp.float32),
        )
        u, _ = self._network.apply(
            variables={"params": params},
            inputs=z_1,
            timestamps=timestamps,
            edm_cond=None,
            deterministic=deterministic,
        )

        return _model.StepOutputs(output=z_1 - u)

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: typing.Any,
        state: _train_state.TrainState,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        # NOTE: enforce float32 for training stability
        image = batch["image"].astype(jnp.float32)
        assert isinstance(image, jax.Array)
        batch_dims = image.shape[:-3]
        (
            tr_rng,
            dropout_rng,
            a_rng,
            m_rng,
            e_rng,
            delta_rng,
        ) = jax.random.split(rngs, 6)

        # pre-process the inputs
        image = image * 2.0 - 1.0
        image, cond = self._augment.apply(
            variables={},
            images=image,
            rngs={"augment": a_rng},
        )
        assert isinstance(image, jax.Array)
        assert isinstance(cond, jax.Array)

        # sample begin timestep r and end timestep t
        t, r = sample_t_r(
            key=tr_rng,
            shape=batch_dims,
            dtype=image.dtype,
            distribution=self.timestamp_sampler,
            **self.timestamp_sampler_kwargs,
        )

        t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        r_eq_t_mask = jnp.less(
            jax.random.uniform(
                key=m_rng,
                shape=batch_dims,
                dtype=image.dtype,
            ),
            self.timestamp_overlap_rate,
        )
        r = jnp.where(r_eq_t_mask, t, r)

        # sample e ~ N(0, I)
        e = jax.random.normal(key=e_rng, shape=image.shape, dtype=image.dtype)

        # generate z_t = (1-t)*x_0 + t*e
        z = jnp.add(
            (1 - t[..., None, None, None]) * image,
            t[..., None, None, None] * e,
        )

        # NOTE: evaluate EMA velocity anchor (outside loss_fn)
        ema_timestamps = self._make_timestamps(t, t)
        v_tang, _ = self._network.apply(
            variables={"params": state.ema_params},
            inputs=z,
            timestamps=ema_timestamps,
            edm_cond=cond,
            deterministic=True,
        )
        v_tang = jax.lax.stop_gradient(v_tang)

        def _loss_fn(
            params: PyTree,
        ) -> typing.Tuple[
            jax.Array,
            typing.Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            def u_fn(
                z_t: jax.Array,
                r_in: jax.Array,
                t_in: jax.Array,
            ):
                """Network forward pass."""
                timestamps = self._make_timestamps(t_in, r_in)
                return self._network.apply(
                    variables={"params": params},
                    inputs=z_t,
                    timestamps=timestamps,
                    edm_cond=cond,
                    deterministic=False,
                    rngs={"dropout": dropout_rng},
                    **kwargs,
                )

            # compute JVP with EMA tangent anchor
            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            primals, tangents = jax.jvp(u_fn, (z, r, t), (v_tang, drdt, dtdt))

            u, log_var = primals
            dudt, _ = tangents

            # compound prediction with stop-gradient on JVP and MeanFlow loss
            v_pred = jnp.add(
                u,
                (t - r)[..., None, None, None] * jax.lax.stop_gradient(dudt),
            )
            v_target = jax.lax.stop_gradient(e - image)
            residual_sq = jnp.sum(
                jnp.square(v_pred - v_target),
                axis=(-1, -2, -3),
            )

            if self.predict_variance:
                sigma_sq = jax.nn.softplus(log_var) + self.variance_floor
                d = float(image.shape[-1] * image.shape[-2] * image.shape[-3])
                nll_loss = residual_sq / (2.0 * sigma_sq)
                nll_loss = nll_loss + 0.5 * d * jnp.log(sigma_sq)
                mse_loss = residual_sq

                # MSE during warmup, NLL after
                mf_loss = jnp.where(
                    state.step >= self.nll_warmup_steps,
                    jnp.mean(nll_loss),
                    jnp.mean(mse_loss),
                )
                sigma_sq_mean = jnp.mean(sigma_sq)
            else:
                if self.adaptive_weight_power > 0.0:
                    ada_wt = jnp.power(
                        residual_sq + 1e-2,
                        self.adaptive_weight_power,
                    )
                    residual_sq = residual_sq / (jax.lax.stop_gradient(ada_wt))
                mf_loss = jnp.mean(residual_sq)
                sigma_sq_mean = jnp.zeros(())

            # Flow-Matching anchor loss
            delta = jax.random.uniform(
                key=delta_rng,
                shape=batch_dims,
                minval=self.fm_anchor_delta_min,
                maxval=self.fm_anchor_delta_max,
                dtype=image.dtype,
            )
            u_anchor, _ = u_fn(z_t=z, r_in=t - delta, t_in=t)
            fm_anchor_loss = jnp.mean(
                jnp.sum(
                    jnp.square(u_anchor - v_target),
                    axis=(-1, -2, -3),
                )
            )

            total_loss = mf_loss + self.fm_anchor_weight * fm_anchor_loss

            # velocity monitoring at boundary (r == t)
            velocity_loss = jnp.where(
                jnp.equal(t, r)[..., None, None, None],
                jnp.square(u - (e - image)),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            return total_loss, (
                mf_loss,
                fm_anchor_loss,
                velocity_loss,
                sigma_sq_mean,
            )

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, aux), grads = grad_fn(state.params)
        mf_loss, fm_anchor_loss, velocity_loss, sigma_sq_mean = aux
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        scalars = {
            "loss": total_loss,
            "mf_loss": mf_loss,
            "fm_anchor_loss": fm_anchor_loss,
            "velocity_loss": velocity_loss,
        }
        if self.predict_variance:
            scalars["sigma_sq_mean"] = sigma_sq_mean

        outputs = _model.StepOutputs(
            scalars=scalars,
            histograms={"t": t, "r": r, "t - r": t - r},
        )

        return new_state, outputs


__all__ = [
    "MeanFlowUNetModule",
    "MeanFlowUNetModel",
    "ImprovedMeanFlowUNetModel",
    "VAMeanFlowUNetModule",
    "VAMeanFlowUNetModel",
]
