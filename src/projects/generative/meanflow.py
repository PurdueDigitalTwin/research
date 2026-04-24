import typing

from flax import linen as nn
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax._src import typing as jax_typing
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.generative.model import dit
from src.projects.generative.model import unet
from src.projects.generative.model import vae as _vae
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
        r_mean = kwargs.get("r_mean", mean)
        r_stddev = kwargs.get("r_stddev", stddev)
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
            mean=r_mean,
            stddev=r_stddev,
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
        timestamp_sampler_version: str = "v0",
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
        self.timestamp_sampler_version = timestamp_sampler_version
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
            "inputs": jnp.zeros(
                (1, self.image_size, self.image_size, self.in_channels),
                dtype=jnp.float32,
            ),
            "timestamps": timestamps,
            "edm_cond": jnp.zeros((1, 6), dtype=jnp.float32),
        }
        variables = self._network.init(
            rngs=rngs,
            **dummy_inputs,
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
        tr_rng, dp_rng, a_rng, m_rng, e_rng = jax.random.split(local_rng, 5)

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

        if self.timestamp_sampler_version == "v1":
            # v1: mask-then-clip ordering
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng, shape=batch_dims, dtype=image.dtype
                ),
                self.timestamp_overlap_rate,
            )
            r = jnp.where(r_eq_t_mask, t, r)
            r = jnp.minimum(t, r)
        else:
            # v0: sort-then-mask ordering
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng, shape=batch_dims, dtype=image.dtype
                ),
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
                    rngs={"dropout": dp_rng},
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
            raw_loss = jnp.mean(loss)

            # applies adaptive weight power
            if self.adaptive_weight_power > 0.0:
                ada_wt = jnp.power(
                    loss + self.norm_eps, self.adaptive_weight_power
                )
                loss = loss / jax.lax.stop_gradient(ada_wt)
            loss = jnp.mean(loss)

            # calculate velocity loss for monitoring
            velocity_loss = jnp.where(
                jnp.equal(t, r)[..., None, None, None],
                jnp.square(u - (e - image)),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            return loss, (velocity_loss, raw_loss)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, (velocity_loss, raw_loss)), grads = grad_fn(state.params)
        global_grad_norm = optax.global_norm(grads)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(
            scalars={
                "loss": raw_loss.mean(),
                "velocity_loss": velocity_loss.mean(),
                "global_grad_norm": global_grad_norm,
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
            t_in (jax.Array): Terminal timestamp.
            r_in (jax.Array): Start timestamp.

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


class MeanFlowDiTModule(nn.Module):
    r"""MeanFlow module with a Diffusion Transformer (DiT) backbone.

    Following the official MeanFlow DiT, this uses two separate
    ``TimestampEmbed`` modules for ``t`` and ``h = t - r``, summed
    into a single conditioning vector for the adaLN-Zero blocks.

    Attributes:
        features (int): Hidden dimension of the transformer.
        patch_size (int): Spatial patch size for tokenization.
        depth (int): Number of DiT blocks.
        num_heads (int): Number of attention heads.
        ffn_ratio (int): FFN hidden dim = features * ffn_ratio.
        dropout_rate (float): Dropout rate for label embedding
            (controls null-class token allocation).
        num_classes (int): Number of discrete classes for
            class-conditional generation. Default is 1000
            (ImageNet).
        deterministic (Optional[bool]): Global deterministic flag.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
        precision (Any): Numerical precision for the computation.
    """

    features: int
    patch_size: int = 2
    depth: int = 12
    num_heads: int = 6
    ffn_ratio: int = 4
    dropout_rate: float = 0.0
    num_classes: int = 1000
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        timestamps: typing.Tuple[jax.Array, ...],
        labels: typing.Optional[jax.Array] = None,
        edm_cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass the MeanFlow DiT module.

        Args:
            inputs (jax.Array): Input images ``(*, H, W, C)``.
            timestamps (Tuple[jax.Array, ...]): Timestamps,
                typically ``(t, t - r)`` for ``t_and_t_minus_r``.
            labels (jax.Array, optional): Class labels ``(B,)``
                for class-conditional generation. Labels should
                already have class dropout applied externally
                (dropped labels set to ``num_classes``).
            edm_cond (jax.Array, optional): EDM augmentation cond.
            deterministic (bool, optional): Whether deterministic.

        Returns:
            Predicted average velocity ``(*, H, W, C)``.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # --- Patch embedding ---
        patch_embed = dit.PatchEmbed(
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

        # --- Positional encoding ---
        pos_emb = dit.sinusoidal_patch_enc(
            features=self.features,
            grid_size=int(out.shape[-2] ** 0.5),
            num_extra_tokens=0,
        )
        out = out + pos_emb[None, :, :].astype(self.dtype)

        # --- Dual timestamp embedding (official MeanFlow DiT) ---
        # timestamps[0] = t, timestamps[1] = h (= t - r)
        t_embed = dit.TimestampEmbed(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="t_embed",
        )(timestamps[0])
        h_embed = dit.TimestampEmbed(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="h_embed",
        )(timestamps[1] if len(timestamps) > 1 else timestamps[0])

        cond = t_embed + h_embed

        # --- Optional class label conditioning ---
        if labels is not None:
            y_embed = dit.LabelEmbed(
                features=self.features,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="y_embed",
            )(labels=labels, deterministic=True)
            cond = cond + y_embed

        # --- Optional augmentation conditioning ---
        if edm_cond is not None:
            aug_embed = nn.Dense(
                features=self.features,
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
            cond = cond + aug_embed(edm_cond)

        # --- DiT blocks ---
        for i in range(self.depth):
            block = dit.DiTAdaLNBlock(
                features=self.features,
                num_heads=self.num_heads,
                ffn_ratio=self.ffn_ratio,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=f"dit_block_{i}",
            )
            out = block(
                inputs=out,
                cond=cond,
                deterministic=m_deterministic,
            )

        # --- Final decoder + unpatchify ---
        decoder = dit.AdaLNDecoder(
            features=inputs.shape[-1],
            patch_size=self.patch_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="decoder",
        )
        out = decoder(out, cond=cond)
        out = dit.DiffusionTransformer.unpatchify(
            inputs=out,
            channels=inputs.shape[-1],
            patch_size=self.patch_size,
        )

        return out


class MeanFlowDiTModel(MeanFlowUNetModel):
    r"""MeanFlow with DiT backbone for latent-space training.

    Supports optional VAE for latent-space training on ImageNet.
    When ``vae_path`` is set, images are encoded to latent space
    before the MeanFlow loss, and decoded to pixel space during
    sampling. ``in_channels`` and ``image_size`` refer to the
    latent dimensions (e.g. 4 and 32 for SD VAE on 256x256).

    Args:
        in_channels (int): Latent channels (4 for SD VAE).
        image_size (int): Latent spatial size (32 for 256px).
        features (int): Hidden dim of the DiT.
        patch_size (int): Spatial patch size.
        depth (int): Number of DiT blocks.
        num_heads (int): Number of attention heads.
        ffn_ratio (int): FFN hidden dim multiplier.
        dropout_rate (float): Dropout rate for label embedding.
        num_classes (int): Number of classes (1000 for ImageNet).
        class_dropout_prob (float): Label dropout probability for
            CFG training. Default is 0.1.
        cfg_omega (float): CFG baking omega. Default is 1.0.
        cfg_kappa (float): CFG baking kappa. Default is 0.5.
            Effective guidance scale = (omega + kappa).
        timestamp_cond (str): Timestamp conditioning type.
        timestamp_sampler (str): Distribution for sampling t, r.
        timestamp_sampler_kwargs (Dict): Kwargs for sampler.
        timestamp_overlap_rate (float): Fraction with r = t.
        adaptive_weight_power (float): Power for adaptive wt.
        vae_path (str): Path to pretrained VAE weights dir.
        vae_scaling_factor (float): Latent scaling factor.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
        precision (Any): Numerical precision.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: int,
        patch_size: int = 2,
        depth: int = 12,
        num_heads: int = 6,
        ffn_ratio: int = 4,
        dropout_rate: float = 0.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        cfg_omega: float = 1.0,
        cfg_kappa: float = 0.5,
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
        timestamp_sampler_version: str = "v0",
        adaptive_weight_power: float = 1.0,
        norm_eps: float = 0.01,
        vae_path: typing.Optional[str] = None,
        vae_scaling_factor: float = 0.18215,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        """Initializes the MeanFlow DiT model."""
        self.in_channels = in_channels
        self.image_size = image_size
        self.features = features
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.timestamp_cond = timestamp_cond
        self.timestamp_sampler = timestamp_sampler
        self.timestamp_sampler_kwargs = timestamp_sampler_kwargs
        self.timestamp_overlap_rate = timestamp_overlap_rate
        self.timestamp_sampler_version = timestamp_sampler_version
        self.adaptive_weight_power = adaptive_weight_power
        self.norm_eps = norm_eps

        # DiT backbone (no EDM augmentation for ImageNet)
        self._augment = None
        self._network = MeanFlowDiTModule(
            features=features,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            name="dit",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        # Frozen VAE for latent-space training
        self._vae = None
        self._vae_params = None
        self._vae_scaling_factor = vae_scaling_factor
        if vae_path is not None:
            vae, vae_params = _vae.AutoencoderKL.from_pretrained(
                vae_path,
                dtype=dtype,
                param_dtype=param_dtype,
            )
            self._vae = vae
            self._vae_params = vae_params

    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> PyTree:
        """Initializes DiT parameters (VAE is frozen)."""
        del batch

        if self.timestamp_cond in [
            "t_and_r",
            "t_and_t_minus_r",
        ]:
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
                "Unsupported timestamp conditioning: "
                f"{self.timestamp_cond}."
            )

        dummy_inputs = {
            "inputs": jnp.zeros(
                (
                    1,
                    self.image_size,
                    self.image_size,
                    self.in_channels,
                ),
                dtype=jnp.float32,
            ),
            "timestamps": timestamps,
            "labels": jnp.zeros((1,), dtype=jnp.int32),
            "edm_cond": None,
        }
        variables = self._network.init(
            rngs=rngs,
            **dummy_inputs,
            deterministic=True,
        )

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
        """MeanFlow training step in latent space."""
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        (
            tr_rng,
            dp_rng,
            vae_rng,
            m_rng,
            e_rng,
            cfg_rng,
        ) = jax.random.split(local_rng, 6)

        # Encode to latent space
        if "latent_mean" in batch:
            # Pre-encoded latents: skip VAE forward pass
            mean = batch["latent_mean"].astype(jnp.float32)
            logvar = batch["latent_logvar"].astype(jnp.float32)
            batch_dims = mean.shape[:-3]
            std = jnp.exp(0.5 * logvar)
            noise = jax.random.normal(vae_rng, mean.shape, dtype=mean.dtype)
            x = (mean + std * noise) * self._vae_scaling_factor
        elif self._vae is not None:
            image = batch["image"].astype(jnp.float32)
            batch_dims = image.shape[:-3]
            image = image * 2.0 - 1.0
            mean, logvar = self._vae.apply(
                {"params": self._vae_params},
                image,
                method=self._vae.encode,
            )
            std = jnp.exp(0.5 * logvar)
            noise = jax.random.normal(vae_rng, mean.shape, dtype=mean.dtype)
            x = (mean + std * noise) * self._vae_scaling_factor
        else:
            image = batch["image"].astype(jnp.float32)
            batch_dims = image.shape[:-3]
            x = image * 2.0 - 1.0

        # Extract class labels
        if "label" in batch:
            labels = batch["label"].astype(jnp.int32)
        else:
            labels = jnp.full(batch_dims, self.num_classes, dtype=jnp.int32)

        # Sample timestamps
        t, r = sample_t_r(
            key=tr_rng,
            shape=batch_dims,
            dtype=x.dtype,
            distribution=self.timestamp_sampler,
            **self.timestamp_sampler_kwargs,
        )

        if self.timestamp_sampler_version == "v1":
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng,
                    shape=batch_dims,
                    dtype=x.dtype,
                ),
                self.timestamp_overlap_rate,
            )
            r = jnp.where(r_eq_t_mask, t, r)
            r = jnp.minimum(t, r)
        else:
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng,
                    shape=batch_dims,
                    dtype=x.dtype,
                ),
                self.timestamp_overlap_rate,
            )
            r = jnp.where(r_eq_t_mask, t, r)

        e = jax.random.normal(key=e_rng, shape=x.shape, dtype=x.dtype)
        z = jnp.add(
            (1 - t[..., None, None, None]) * x,
            t[..., None, None, None] * e,
        )

        # Ground truth velocity (used in CFG formula + dropout)
        v = e - x

        def _loss_fn(
            params: PyTree,
        ) -> typing.Tuple[jax.Array, jax.Array]:
            # --- CFG baking: compute guided velocity target ---
            # Following the official MeanFlow, the CFG formula
            # mixes ground-truth velocity v = e - x with network
            # predictions v_uncond and v_cond at h=0.
            ts_h0 = self._make_timestamps(t_in=t, r_in=t)

            # Unconditional velocity (null class)
            null_labels = jnp.full_like(labels, self.num_classes)
            v_uncond = self._network.apply(
                variables={"params": jax.lax.stop_gradient(params)},
                inputs=z,
                timestamps=ts_h0,
                labels=null_labels,
                edm_cond=None,
                deterministic=True,
            )
            # Conditional velocity (with class labels)
            v_cond = self._network.apply(
                variables={"params": jax.lax.stop_gradient(params)},
                inputs=z,
                timestamps=ts_h0,
                labels=labels,
                edm_cond=None,
                deterministic=True,
            )

            # Official CFG formula (meanflow/meanflow.py):
            #   v_g = omega*v + (1-omega-kappa)*v_uncond
            #         + kappa*v_cond
            # where v = e - x is the ground-truth velocity.
            # With omega=1.0, kappa=0.5:
            #   v_g = (e-x) - 0.5*v_uncond + 0.5*v_cond
            v_g = (
                self.cfg_omega * v
                + (1.0 - self.cfg_omega - self.cfg_kappa) * v_uncond
                + self.cfg_kappa * v_cond
            )

            # --- Class dropout: dropped samples revert to
            # ground-truth velocity and get null labels ---
            drop_mask = jnp.less(
                jax.random.uniform(cfg_rng, shape=batch_dims),
                self.class_dropout_prob,
            )
            # Dropped: target = v (ground truth), label = null
            # Kept: target = v_g (guided), label = original
            v_g = jnp.where(drop_mask[..., None, None, None], v, v_g)
            y_inp = jnp.where(drop_mask, self.num_classes, labels)

            def u_fn(
                z_t: jax.Array,
                r_in: jax.Array,
                t_in: jax.Array,
            ) -> jax.Array:
                ts = self._make_timestamps(t_in=t_in, r_in=r_in)
                return self._network.apply(
                    variables={"params": params},
                    inputs=z_t,
                    timestamps=ts,
                    labels=y_inp,
                    edm_cond=None,
                    deterministic=False,
                    rngs={"dropout": dp_rng},
                    **kwargs,
                )

            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            # JVP tangent is v_g (guided velocity), matching
            # official MeanFlow CFG baking.
            u, dudt = jax.jvp(u_fn, (z, r, t), (v_g, drdt, dtdt))
            u_target = (
                v_g
                - jnp.clip(t - r, a_min=0.0, a_max=1.0)[..., None, None, None]
                * dudt
            )

            per_sample_loss = jnp.sum(
                jnp.square(u - jax.lax.stop_gradient(u_target)),
                axis=(-1, -2, -3),
            )
            raw_loss = jnp.mean(per_sample_loss)

            if self.adaptive_weight_power > 0.0:
                ada_wt = jnp.power(
                    per_sample_loss + self.norm_eps,
                    self.adaptive_weight_power,
                )
                per_sample_loss = per_sample_loss / jax.lax.stop_gradient(
                    ada_wt
                )
            loss = jnp.mean(per_sample_loss)

            # --- diagnostic metrics (not in gradient graph) ---
            is_boundary = jnp.equal(t, r)
            n_boundary = jnp.sum(is_boundary).astype(jnp.float32)
            n_interior = jnp.sum(~is_boundary).astype(jnp.float32)

            # per-sample raw MSE (before adaptive weight)
            raw_per_sample = jnp.sum(
                jnp.square(u - jax.lax.stop_gradient(u_target)),
                axis=(-1, -2, -3),
            )

            # boundary loss (t==r): flow-matching component
            boundary_loss = jnp.where(is_boundary, raw_per_sample, 0.0)
            boundary_loss = jnp.sum(boundary_loss) / jnp.maximum(
                n_boundary, 1.0
            )

            # interior loss (t!=r): MeanFlow identity component
            interior_loss = jnp.where(~is_boundary, raw_per_sample, 0.0)
            interior_loss = jnp.sum(interior_loss) / jnp.maximum(
                n_interior, 1.0
            )

            # JVP correction magnitude
            correction = (t - r)[..., None, None, None] * dudt
            dudt_magnitude = jnp.mean(
                jnp.sum(jnp.square(correction), axis=(-1, -2, -3))
            )

            velocity_loss = jnp.where(
                is_boundary[..., None, None, None],
                jnp.square(u - (e - x)),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            diagnostics = (
                velocity_loss,
                raw_loss,
                boundary_loss,
                interior_loss,
                dudt_magnitude,
            )
            return loss, diagnostics

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, diagnostics), grads = grad_fn(state.params)
        (
            velocity_loss,
            raw_loss,
            boundary_loss,
            interior_loss,
            dudt_magnitude,
        ) = diagnostics
        global_grad_norm = optax.global_norm(grads)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(
            scalars={
                "loss": raw_loss.mean(),
                "boundary_loss": boundary_loss.mean(),
                "interior_loss": interior_loss.mean(),
                "velocity_loss": velocity_loss.mean(),
                "dudt_magnitude": dudt_magnitude.mean(),
                "global_grad_norm": global_grad_norm,
            },
            histograms={
                "t": t,
                "r": r,
                "t - r": t - r,
            },
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
        batch: typing.Optional[typing.Any] = None,
        **kwargs,
    ) -> _model.StepOutputs:
        """One-step sampling with optional VAE decoding.

        CFG is baked into the weights during training, so
        inference uses a single forward pass with class labels
        (no guidance-scale interpolation needed).

        Args:
            rngs (jax.Array): Random key for noise sampling.
            params (FrozenDict): DiT model parameters.
            shape (Sequence): Pixel-space shape from the batch.
            deterministic (bool): Whether deterministic.
            batch (Dict, optional): Batch dict with ``"label"``
                key for class-conditional sampling.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated images in ``[-1, 1]``.
        """
        del kwargs

        # Generate noise in latent space
        batch_dims = shape[:-3]
        latent_shape = batch_dims + (
            self.image_size,
            self.image_size,
            self.in_channels,
        )
        z_1 = jax.random.normal(
            key=rngs,
            shape=latent_shape,
            dtype=self._network.dtype,
        )
        timestamps = self._make_timestamps(
            t_in=jnp.ones(batch_dims, dtype=jnp.float32),
            r_in=jnp.zeros(batch_dims, dtype=jnp.float32),
        )

        # Extract class labels from batch
        if batch is not None and "label" in batch:
            labels = batch["label"][: batch_dims[0]].astype(jnp.int32)
        else:
            labels = jnp.full(batch_dims, self.num_classes, dtype=jnp.int32)

        # One-step MeanFlow: z_0 = z_1 - u(z_1, 0, 1)
        # No CFG at inference — guidance is baked into weights.
        z_0 = z_1 - self._network.apply(
            variables={"params": params},
            inputs=z_1,
            timestamps=timestamps,
            labels=labels,
            edm_cond=None,
            deterministic=deterministic,
        )

        # Decode to pixel space if VAE is available
        if self._vae is not None:
            z_0 = z_0 / self._vae_scaling_factor
            out = self._vae.apply(
                {"params": self._vae_params},
                z_0,
                method=self._vae.decode,
            )
        else:
            out = z_0

        return _model.StepOutputs(output=out)


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
        tr_rng, dp_rng, a_rng, m_rng, e_rng = jax.random.split(local_rng, 5)

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

        if self.timestamp_sampler_version == "v1":
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng, shape=batch_dims, dtype=image.dtype
                ),
                self.timestamp_overlap_rate,
            )
            r = jnp.where(r_eq_t_mask, t, r)
            r = jnp.minimum(t, r)
        else:
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng, shape=batch_dims, dtype=image.dtype
                ),
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
                    rngs={"dropout": dp_rng},
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
            raw_loss = jnp.mean(loss)
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

            return loss, (velocity_loss, raw_loss)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, (velocity_loss, raw_loss)), grads = grad_fn(state.params)
        global_grad_norm = optax.global_norm(grads)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(
            scalars={
                "loss": raw_loss.mean(),
                "velocity_loss": velocity_loss.mean(),
                "global_grad_norm": global_grad_norm,
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
        timestamps: typing.Tuple[jax.Array, ...],
        edm_cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> typing.Tuple[jax.Array, typing.Optional[jax.Array]]:
        r"""Forward pas the ``VAMeanFlowUNetModule``.

        Args:
            inputs (jax.Array): Input data of shape ``(*, D1, D2, ..., C)``.
            timestamps (Tuple[jax.Array, ...]): Time steps of shape ``(*, 1)``.
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
        embed = [time_embed(t) for t in timestamps]
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

        # variance head: per-location Dense(C) -> per-pixel-per-channel
        # log-variance ``(B, H, W, C)``. Each pixel's variance is a
        # linear projection of the shared post-norm+silu activation,
        # capturing spatial heteroscedasticity. Paired with the
        # per-pixel diagonal Gaussian NLL in
        # ``VAMeanFlowUNetModel._loss_fn``.
        if self.predict_variance:
            var_head = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="var_head",
            )
            log_var = var_head(h_act)
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
        adaptive_weight_power (float): Inherited from the parent class
            but unused by VaMF. Both variants use the SNR schedule
            ``w(t) = (1-t)^2 / (t^2 + snr_epsilon)`` per the paper's
            Algorithm 2 as the per-sample timestep weighting. The
            NLL variant applies it on top of the β-NLL per-pixel
            term; see PR (f.1) / audit Revision 4 (P1d) for why
            relying on the variance head alone is insufficient.
        snr_epsilon (float): Stabilizer in the SNR weight denominator
            ``w(t) = (1 - t)^2 / (t^2 + snr_epsilon)``. Active in
            both the MSE and NLL variants.
        fm_anchor_weight (float): Weight for FM anchor loss.
        fm_anchor_delta_min (float): Min interval for FM anchor.
        fm_anchor_delta_max (float): Max interval for FM anchor.
        predict_variance (bool): Enable heteroscedastic variance
            head (NLL variant).
        variance_floor (float): Minimum variance to prevent
            collapse.
        nll_warmup_steps (int): End of the MSE->NLL linear ramp. For
            steps ``< nll_warmup_steps - nll_ramp_steps`` the loss is
            pure MSE; for steps ``>= nll_warmup_steps`` the loss is
            pure per-pixel diagonal Gaussian NLL; in between the two
            are linearly interpolated.
        nll_ramp_steps (int): Number of steps over which the MSE->NLL
            transition is linearly ramped, ending at
            ``nll_warmup_steps``. Default 2_000.
        nll_beta (float): β-NLL weighting exponent per Seitzer et al.,
            "On the Pitfalls of Heteroscedastic Uncertainty Estimation
            with Probabilistic Neural Networks", ICLR 2022. The
            per-pixel NLL is multiplied by
            ``stop_gradient(sigma_sq ** nll_beta)`` so that the
            mean-head gradient wrt ``v_pred`` becomes
            ``stop_gradient(sigma_sq ** (nll_beta - 1)) * (v_pred -
            v_target)``. With ``nll_beta=1.0`` the mean head sees a
            plain MSE-scale gradient regardless of ``sigma_sq``, which
            prevents the ``1/sigma_sq`` amplification failure mode
            observed on run ``hz3dpmz4`` (audit Revision 3, P1c).
            ``nll_beta=0.0`` recovers the plain NLL (the known
            pitfall). ``stop_gradient`` ensures the variance head's
            fixed point ``sigma_sq* = E[r^2]`` is preserved for all
            values of ``nll_beta``. Default 1.0.
        no_fm_anchor (bool): Ablation flag. When ``True``, skip the
            flow-matching anchor branch entirely: no anchor forward
            pass, ``fm_anchor_loss`` reported as 0, and ``total_loss``
            collapses to ``mf_loss``. Used by the R5a/R5d ablations to
            isolate the FM anchor's contribution. Default ``False``.
        boundary_tangent (bool): Ablation flag. When ``True``, replace
            the EMA tangent ``u_{theta-bar}(z, t, t)`` with the
            current model's own boundary prediction
            ``u_theta(z, t, t)`` (still under ``stop_gradient``). This
            isolates the contribution of EMA averaging vs. simply
            having a deterministic boundary tangent. Used by the R5b
            ablation. Default ``False``.
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
        timestamp_sampler_version: str = "v0",
        adaptive_weight_power: float = 0.0,
        snr_epsilon: float = 1e-2,
        fm_anchor_weight: float = 0.5,
        fm_anchor_delta_min: float = 1e-4,
        fm_anchor_delta_max: float = 0.01,
        predict_variance: bool = False,
        variance_floor: float = 1e-4,
        nll_warmup_steps: int = 10_000,
        nll_ramp_steps: int = 2_000,
        nll_beta: float = 1.0,
        no_fm_anchor: bool = False,
        boundary_tangent: bool = False,
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
            timestamp_sampler_version=timestamp_sampler_version,
            adaptive_weight_power=adaptive_weight_power,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.snr_epsilon = snr_epsilon
        self.fm_anchor_weight = fm_anchor_weight
        self.fm_anchor_delta_min = fm_anchor_delta_min
        self.fm_anchor_delta_max = fm_anchor_delta_max
        self.predict_variance = predict_variance
        self.variance_floor = variance_floor
        self.nll_warmup_steps = nll_warmup_steps
        self.nll_ramp_steps = nll_ramp_steps
        self.nll_beta = nll_beta
        self.no_fm_anchor = no_fm_anchor
        self.boundary_tangent = boundary_tangent

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
        ) = jax.random.split(local_rng, 6)

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

        if self.timestamp_sampler_version == "v1":
            r_eq_t_mask = jnp.less(
                jax.random.uniform(
                    key=m_rng,
                    shape=batch_dims,
                    dtype=image.dtype,
                ),
                self.timestamp_overlap_rate,
            )
            r = jnp.where(r_eq_t_mask, t, r)
            r = jnp.minimum(t, r)
        else:
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

        # NOTE: evaluate the boundary velocity anchor (outside loss_fn).
        # By default, use the EMA model parameters per VaMF (D1).
        # When the ``boundary_tangent`` ablation flag is set, use the
        # current model's own boundary prediction instead — this drops
        # the EMA averaging and isolates whether the deterministic
        # tangent alone is sufficient. ``stop_gradient`` is required
        # in both cases since the JVP tangent must not propagate
        # gradients back into ``params``.
        ema_timestamps = self._make_timestamps(t, t)
        tangent_params = (
            state.params if self.boundary_tangent else state.ema_params
        )
        v_tang, _ = self._network.apply(
            variables={"params": tangent_params},
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
            typing.Tuple[
                jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
            ],
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
            residual_per_pixel = jnp.square(v_pred - v_target)  # (B,H,W,C)
            residual_sq = jnp.sum(
                residual_per_pixel,
                axis=(-1, -2, -3),
            )  # (B,)

            # SNR weighting per paper Algorithm 2:
            #   w(t) = (1 - t)^2 / (t^2 + snr_epsilon)
            # Downweights high-variance timesteps near t=1. Shared
            # by both the MSE and NLL variants so the mean head
            # focuses on low-t samples (which FID is sensitive to)
            # in both paths. Replaces the inherited MeanFlow Karras
            # adaptive weighting.
            #
            # PR (f.1) (audit Revision 4, P1d): the NLL variant
            # initially omitted this weighting on the assumption
            # that the variance head would provide automatic
            # per-sample weighting. In practice the per-pixel
            # diagonal Gaussian NLL has no explicit t-dependence,
            # and once σ² converged to the residual scale the mean
            # head spent uniform gradient across all t. On run
            # ``rpj8lacb`` this decoupled the training-loss
            # landscape from FID: mf_loss / velocity_loss /
            # grad_norm all decreased monotonically through the
            # kill point while FID regressed +12.50 at step 15k.
            # Applying snr_wt restores t-dependent weighting on top
            # of the β-NLL σ²-decoupling.
            snr_wt = jnp.square(1.0 - t) / (
                jnp.square(t) + self.snr_epsilon
            )  # (B,)

            if self.predict_variance:
                # Per-pixel diagonal Gaussian NLL with β-NLL weighting
                # (Seitzer et al., ICLR 2022, "On the Pitfalls of
                # Heteroscedastic Uncertainty Estimation with
                # Probabilistic Neural Networks"):
                #   l_βNLL = sg(σ²^β) · 0.5 · sum_i
                #            [r_i² / σ_i² + log σ_i²]
                # with σ_i² = softplus(log_var_i) + variance_floor per
                # pixel per channel.
                #
                # See docs/generative/vamf/reviews/
                # nll-audit-2026-04-06.md Revision 1 (P0 per-pixel
                # form), Revision 3 (P1c β-NLL decoupling), and
                # Revision 4 (P1d SNR timestep weighting).
                #
                # Why β-NLL: with plain NLL the mean-head gradient
                # wrt v_pred is (v_pred - v_target) / σ², so once
                # σ² converges near the residual scale (~0.13 on
                # CIFAR-10 mid-training) the effective mean-head
                # learning rate is amplified ~8× vs plain MSE. The
                # mean head drifts into a training-loss-optimal but
                # FID-suboptimal basin — observed on run hz3dpmz4.
                # Multiplying by stop_gradient(σ²^β) makes the
                # mean-head gradient stop_gradient(σ²^(β-1)) ·
                # (v_pred - v_target); with self.nll_beta = 1 that
                # is exactly MSE. The stop_gradient blocks the
                # weight from back-propagating, so the variance
                # head's fixed point σ²* = E[r²] is unchanged for
                # all β — only the mean head's effective LR
                # decouples from σ².
                sigma_sq = (
                    jax.nn.softplus(log_var) + self.variance_floor
                )  # (B,H,W,C)
                nll_per_pixel_raw = 0.5 * (
                    residual_per_pixel / sigma_sq + jnp.log(sigma_sq)
                )
                nll_weight = jax.lax.stop_gradient(
                    sigma_sq**self.nll_beta
                )  # (B,H,W,C)
                nll_per_pixel = nll_weight * nll_per_pixel_raw
                nll_loss = jnp.sum(nll_per_pixel, axis=(-1, -2, -3))  # (B,)
                mse_loss = residual_sq  # (B,)

                # Linear MSE -> NLL ramp ending at nll_warmup_steps
                # (P2 fix). For steps < warmup_end - ramp the loss is
                # pure MSE; for steps >= warmup_end it is pure NLL; in
                # between the two are linearly interpolated so the
                # variance head receives non-zero gradient during the
                # transition.
                warmup_end = jnp.float32(self.nll_warmup_steps)
                warmup_start = warmup_end - jnp.float32(self.nll_ramp_steps)
                alpha = jnp.clip(
                    (jnp.float32(state.step) - warmup_start)
                    / jnp.maximum(warmup_end - warmup_start, 1.0),
                    0.0,
                    1.0,
                )
                # Apply SNR weighting to the per-sample loss (both
                # the MSE term during warmup and the β-NLL term
                # post-warmup). snr_wt has shape (B,); mse_loss and
                # nll_loss are per-sample (B,), so the product is a
                # simple per-sample reweighting.
                mf_loss = jnp.mean(
                    snr_wt * ((1.0 - alpha) * mse_loss + alpha * nll_loss)
                )
                sigma_sq_mean = jnp.mean(sigma_sq)
                log_var_std = jnp.std(jnp.log(sigma_sq))
            else:
                mf_loss = jnp.mean(snr_wt * residual_sq)
                sigma_sq_mean = jnp.zeros(())
                log_var_std = jnp.zeros(())

            # Flow-Matching anchor loss. Skipped entirely under the
            # ``no_fm_anchor`` ablation: no second forward pass, the
            # scalar is reported as 0, and ``total_loss`` collapses to
            # ``mf_loss``. Used by the R5a/R5d ablations.
            if self.no_fm_anchor:
                fm_anchor_loss = jnp.zeros(())
                total_loss = mf_loss
            else:
                delta = jax.random.uniform(
                    key=delta_rng,
                    shape=batch_dims,
                    minval=self.fm_anchor_delta_min,
                    maxval=self.fm_anchor_delta_max,
                    dtype=image.dtype,
                )
                u_anchor, log_var_anchor = u_fn(z_t=z, r_in=t - delta, t_in=t)
                fm_residual_per_pixel = jnp.square(
                    u_anchor - v_target
                )  # (B,H,W,C)

                if self.predict_variance:
                    # Per-pixel NLL-consistent FM anchor using the
                    # variance head at the anchor point (P1 fix,
                    # Option A). Keeps ``mf_loss`` and
                    # ``fm_anchor_loss`` on the same per-pixel NLL
                    # scale so ``fm_anchor_weight`` has a consistent
                    # meaning across training and across the MSE/NLL
                    # variants.
                    #
                    # PR (d) correction (audit Revision 2): the FM
                    # anchor must respect the same MSE->NLL alpha
                    # ramp applied to ``mf_loss`` above. Without
                    # the ramp here, during pre-warmup (``alpha=0``
                    # → pure MSE for ``mf_loss``) the variance head
                    # was being trained *exclusively* by the FM
                    # anchor's NLL term. The optimizer drove
                    # ``sigma^2`` downward to shrink
                    # ``0.5 * log sigma^2``, and once ``alpha``
                    # ramped up ``mf_loss`` inherited a
                    # already-shrunk variance head and flipped
                    # sign. Observed on run ``apjbrvz2`` (killed
                    # 2026-04-07 at step ~17k).
                    sigma_sq_anchor = (
                        jax.nn.softplus(log_var_anchor) + self.variance_floor
                    )  # (B,H,W,C)
                    fm_mse_loss = jnp.mean(
                        jnp.sum(fm_residual_per_pixel, axis=(-1, -2, -3))
                    )
                    # Same β-NLL weighting as mf_loss above. The
                    # stop_gradient(σ²_anchor^β) weight keeps the
                    # mean-head gradient at MSE scale while leaving
                    # the variance head's fixed point intact. See
                    # audit Revision 3 (P1c).
                    fm_nll_per_pixel_raw = 0.5 * (
                        fm_residual_per_pixel / sigma_sq_anchor
                        + jnp.log(sigma_sq_anchor)
                    )
                    fm_nll_weight = jax.lax.stop_gradient(
                        sigma_sq_anchor**self.nll_beta
                    )  # (B,H,W,C)
                    fm_nll_per_pixel = fm_nll_weight * fm_nll_per_pixel_raw
                    fm_nll_loss = jnp.mean(
                        jnp.sum(fm_nll_per_pixel, axis=(-1, -2, -3))
                    )
                    fm_anchor_loss = (
                        1.0 - alpha
                    ) * fm_mse_loss + alpha * fm_nll_loss
                else:
                    fm_anchor_loss = jnp.mean(
                        jnp.sum(fm_residual_per_pixel, axis=(-1, -2, -3))
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
                log_var_std,
            )

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, aux), grads = grad_fn(state.params)
        (
            mf_loss,
            fm_anchor_loss,
            velocity_loss,
            sigma_sq_mean,
            log_var_std,
        ) = aux
        global_grad_norm = optax.global_norm(grads)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        # Relative magnitude of the weighted FM anchor term to the MF
        # loss (absolute value, since the per-pixel NLL can go negative
        # when the mean head over-fits). Values >> 1 signal that the
        # FM anchor is drowning out the MF signal and fm_anchor_weight
        # needs tuning.
        fm_mf_ratio = (
            self.fm_anchor_weight
            * jnp.abs(fm_anchor_loss)
            / (jnp.abs(mf_loss) + 1e-8)
        )

        scalars = {
            "loss": total_loss.mean(),
            "mf_loss": mf_loss.mean(),
            "fm_anchor_loss": fm_anchor_loss.mean(),
            "velocity_loss": velocity_loss.mean(),
            "global_grad_norm": global_grad_norm,
            "fm_mf_ratio": fm_mf_ratio,
        }
        if self.predict_variance:
            scalars["sigma_sq_mean"] = sigma_sq_mean
            scalars["log_var_std"] = log_var_std

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
