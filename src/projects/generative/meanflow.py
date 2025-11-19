import typing

import chex
from flax import linen as nn
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax._src import typing as jax_typing
import jaxtyping
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.generative.model import refinenet

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
            One of `["uniform", "lognormal"]`.
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
    elif distribution == "lognormal":

        def _lognormal(
            key: jax.Array,
            shape: jax_typing.Shape,
            dtype: typing.Any,
            mean: float,
            stddev: float,
        ) -> jax.Array:
            z = jax.random.normal(key=key, shape=shape, dtype=dtype)
            return jnp.exp(mean + stddev * z)

        mean = kwargs.get("mean", -0.4)
        stddev = kwargs.get("stddev", 1.0)
        t = _lognormal(
            key=t_key,
            shape=shape,
            dtype=dtype,
            mean=mean,
            stddev=stddev,
        )
        r = _lognormal(
            key=r_key,
            shape=shape,
            dtype=dtype,
            mean=mean,
            stddev=stddev,
        )
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            'Must be one of ["uniform", "lognormal"].'
        )

    return jnp.clip(t, a_min=0.0, a_max=1.0), jnp.clip(r, a_min=0.0, a_max=1.0)


# ==============================================================================
# Helper modules
# ==============================================================================
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


class ConditionEmbed(nn.Module):
    """Encode integer class labels to vectors.

    Attributes:
        features (int): Dimensionality of the output embeddings.
        num_classes (int): Number of classes.
        use_cfg_embedding (bool): Whether to use classifier-free guidance (CFG).
        deterministic (Optional[bool]): Whether to run deterministically.
        dropout_rate (float): Dropout rate for the classifier-free guidance.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
    """

    features: int
    """int: Dimensionality of the output embeddings."""
    num_classes: int
    """int: Number of unique classes."""
    use_cfg_embedding: bool = False
    """bool: Whether to use classifier-free guidance (CFG) embedding."""
    deterministic: typing.Optional[bool] = False
    """Optional[bool]: Whether to run deterministically."""
    dropout_rate: float = 0.0
    """float: Dropout rate for the classifier-free guidance."""
    dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the computation."""
    param_dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the parameters."""

    def setup(self) -> None:
        """Instantiate a `ConditionEmbed` module."""
        self.class_emb = nn.Embed(
            num_embeddings=self.num_classes + int(self.use_cfg_embedding),
            features=self.features,
            name="embedding_table",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
        )

    def __call__(
        self,
        cond: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        """Forward pass the condition encoder.

        Args:
            cond (jax.Array): Integer class labels of shape `(*, )`.
            deterministic (bool, optional): Whether to run deterministically.

        Returns:
            jax.Array: Condition embeddings of shape `(..., features)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        if self.use_cfg_embedding and not m_deterministic:
            raise NotImplementedError(
                "Classifier-free guidance is not implemented yet."
            )

        embedding = self.class_emb(cond)
        return embedding

    @staticmethod
    def _drop_token(
        cond: jax.Array,
        dropout_rate: float,
        rng: jax.Array,
    ) -> jax.Array:
        """Drops class tokens for classifier-free guidance."""
        raise NotImplementedError("This method is not yet implemented.")


class ConditionalInstanceNorm(nn.Module):
    """Instance normalization with conditional inputs."""

    features: int
    """int: Dimensionality of the feature map."""
    use_bias: bool = True
    """bool: Whether to use bias in the normalization."""
    dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the computation."""
    param_dtype: typing.Any = jnp.float32
    """typing.Any: The dtype of the parameters."""

    def setup(self) -> None:
        """Instantiate a `ConditionalInstanceNorm` module."""
        self.instance_norm = nn.LayerNorm(
            reduction_axes=(-3, -2),
            feature_axes=-1,
            use_bias=False,
            use_scale=False,
            epsilon=1e-5,
            name="instance_norm",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        if self.use_bias:
            self.embed = nn.Dense(
                features=3 * self.features,
                use_bias=False,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=0.02,
                    mode="fan_in",
                    distribution="uniform",
                ),
                name="embed",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.embed = nn.Dense(
                features=2 * self.features,
                use_bias=False,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=0.02,
                    mode="fan_in",
                    distribution="uniform",
                ),
                name="embed",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass the conditional instance normalization.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.
            cond (jax.Array): Conditioning embeddings of shape `(*, C)`.

        Returns:
            jax.Array: The normalized feature map of shape `(*, H, W, C)`.
        """
        means = jnp.mean(inputs, axis=(-3, -2), keepdims=False)
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        means = jnp.true_divide(means - m, jnp.sqrt(v + 1e-5))
        means = means[..., None, None, :]
        h = self.instance_norm(inputs)

        if self.use_bias:
            gamma, alpha, beta = jnp.split(self.embed(cond), 3, axis=-1)
            gamma = gamma[..., None, None, :]
            alpha = alpha[..., None, None, :]
            beta = beta[..., None, None, :]
            chex.assert_equal_rank((h, gamma, alpha, beta))
            h = h + means * alpha
            output = h * gamma + beta
        else:
            gamma, alpha = jnp.split(self.embed(cond), 2, axis=-1)
            gamma = gamma[..., None, None, :]
            alpha = alpha[..., None, None, :]
            chex.assert_equal_rank((h, gamma, alpha))
            output = gamma * h + means * alpha

        return output


# ==============================================================================
# Main modules
# ==============================================================================
class MeanFlowUNetModule(nn.Module):
    """Generative model with a RefineNet backbone trained with `MeanFlow`.

    Attributes:
        in_channels (int): Number of channels in the input images.
        image_size (int): Height and width of the input images.
        latent_channels (int): Number of channels in the latent feature maps.
        num_classes (int): Number of conditioning classes.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
    """

    in_channels: int
    """int: Number of channels in the input images."""
    image_size: int
    """int: Height and width of the input images."""
    latent_channels: int
    """int: Number of channels in the latent feature maps."""
    num_classes: int
    """int: Number of conditioning classes."""
    use_cfg_embedding: bool = False
    """bool: Whether to use classifier-free guidance (CFG) embedding."""
    deterministic: typing.Optional[bool] = None
    """Optional[bool]: Whether to run deterministically."""
    dropout_rate: float = 0.0
    """float: Dropout rate for the classifier-free guidance."""
    dtype: typing.Any = None
    """typing.Any: The dtype of the computation."""
    param_dtype: typing.Any = None
    """typing.Any: The dtype of the parameters."""
    precision: typing.Any = None
    """typing.Any: The precision of the computation."""

    def setup(self) -> None:
        r"""Instantiate a `MeanFlowUNetModel` module."""
        self.backbone = refinenet.ConditionalRefineNet(
            in_channels=self.in_channels,
            image_size=self.image_size,
            latent_channels=self.latent_channels,
            norm_module=ConditionalInstanceNorm,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.r_embed = TimestampEmbed(
            features=self.latent_channels,
            frequency=256,
            max_stamp=10_000,
            name="r_embedder",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.t_embed = TimestampEmbed(
            features=self.latent_channels,
            frequency=256,
            max_stamp=10_000,
            name="t_embedder",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.label_embed = ConditionEmbed(
            features=self.latent_channels,
            num_classes=self.num_classes,
            use_cfg_embedding=self.use_cfg_embedding,
            deterministic=self.deterministic,
            dropout_rate=self.dropout_rate,
            name="y_embedder",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        image: jax.Array,
        label: jax.Array,
        begin: typing.Optional[jax.Array] = None,
        end: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        """Forward pass the `MeanFlowUNetModel`.

        Args:
            inputs (jax.Array): Input images of shape `(*, H, W, C)`.
            cond (jax.Array): Conditioning labels of shape `(*,)`.
            begin (jax.Array): Begin timestamp `r` of shape `(*, )`.
            end (jax.Array): End timestamp `t` of shape `(*, )`.

        Returns:
            jax.Array: The predicted average velocity of shape `(*, H, W, C)`.
        """
        # sanity check for the input arrays
        batch_dims = image.shape[:-3]
        dims = chex.Dimensions(
            H=self.image_size,
            W=self.image_size,
            C=self.in_channels,
        )
        chex.assert_shape(image, (*batch_dims, *dims["HWC"]))
        chex.assert_shape(label, batch_dims)
        chex.assert_shape(begin, batch_dims)
        chex.assert_shape(end, batch_dims)

        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        y_emb = self.label_embed(label, deterministic=m_deterministic)
        if begin is not None:
            r_emb = self.r_embed(begin)
        else:
            r_emb = jnp.zeros_like(y_emb)
        if end is not None:
            t_emb = self.t_embed(end)
        else:
            t_emb = jnp.zeros_like(y_emb)
        cond = t_emb + r_emb + y_emb
        output = self.backbone(inputs=image, cond=cond)

        return output


class MeanFlowUNetModel(_model.Model):
    r"""`MeanFlow` generative model with a U-Net backbone.

    Args:
        in_channels (int): Number of channels in the input images.
        image_size (int): Height and width of the (square) input images.
        latent_channels (int): Number of channels in the latent feature maps.
        num_classes (int): Number of conditioning classes.
        use_cfg_embedding (bool): Whether to use classifier-free guidance (CFG).
        dropout_rate (float): Dropout rate for the classifier-free guidance.
        dtype (dtype): The dtype of the computation (default: float32).
        param_dtype (dtype): The dtype of the parameters (default: float32).
        timestamp_cond (Literal): The type of timestamp conditioning.
            One of `["t_and_r", "t_and_t_minus_r",
            "t_and_r_and_t_minus_r", "t_minus_r"]`.
        timestamp_sampler (str): The distribution to sample timestamps from.
            One of `["uniform", "lognormal"]`.
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
        latent_channels: int,
        num_classes: int,
        use_cfg_embedding: bool,
        dropout_rate: float,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
        timestamp_cond: typing.Literal[
            "t_and_r",
            "t_and_t_minus_r",
            "t_and_r_and_t_minus_r",
            "t_minus_r",
        ] = "t_and_t_minus_r",
        timestamp_sampler: str = "lognormal",
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
        self.timestamp_cond = timestamp_cond
        self.timestamp_sampler = timestamp_sampler
        self.timestamp_sampler_kwargs = timestamp_sampler_kwargs
        self.timestamp_overlap_rate = timestamp_overlap_rate
        self.adaptive_weight_power = adaptive_weight_power
        self._network = MeanFlowUNetModule(
            in_channels=in_channels,
            image_size=image_size,
            latent_channels=latent_channels,
            num_classes=num_classes,
            use_cfg_embedding=use_cfg_embedding,
            dropout_rate=dropout_rate,
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
        dummy_inputs = {
            "image": jnp.zeros(
                (1, self.image_size, self.image_size, self.in_channels),
                dtype=jnp.float32,
            ),
            "label": jnp.zeros((1,), dtype=jnp.int32),
            "begin": jnp.zeros((1,), dtype=jnp.float32),
            "end": jnp.zeros((1,), dtype=jnp.float32),
        }
        variables = self.network.init(
            rngs=rngs,
            image=dummy_inputs["image"],
            label=dummy_inputs["label"],
            begin=dummy_inputs["begin"],
            end=dummy_inputs["end"],
            deterministic=True,
        )
        _tabulate_fn = nn.summary.tabulate(self.network, rngs=rngs)
        print(_tabulate_fn(**dummy_inputs, deterministic=True))

        return variables["params"]

    @typing_extensions.override
    def evaluation_step(
        self,
        *,
        params: PyTree,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> _model.StepOutputs:
        del kwargs  # unused

        local_rng = jax.random.fold_in(rngs, jax.process_index())
        e_rng = jax.random.fold_in(local_rng, 0)

        image, label = batch["image"], batch["label"]

        batch_dims = image.shape[:-3]
        dims = chex.Dimensions(
            H=self.image_size,
            W=self.image_size,
            C=self.in_channels,
        )
        chex.assert_shape(image, (*batch_dims, *dims["HWC"]))
        chex.assert_shape(label, batch_dims)

        r = jnp.zeros(batch_dims, dtype=image.dtype)
        t = jnp.ones(batch_dims, dtype=image.dtype)
        sample = jnp.subtract(
            image,
            jnp.einsum(
                "...,...n->...n",
                (t - r),
                self._u_fn(
                    label=label,
                    params=params,
                    deterministic=True,
                )(image, r, t),
            ),
        )

        drdt = jnp.zeros_like(r)
        dtdt = jnp.ones_like(t)
        e = jax.random.normal(key=e_rng, shape=image.shape, dtype=image.dtype)
        v = e - image
        u, dudt = jax.jvp(
            self._u_fn(
                label=label,
                params=params,
                deterministic=True,
            ),
            (e, r, t),
            (v, drdt, dtdt),
        )
        u_target = jax.lax.stop_gradient(
            v
            - jnp.clip(t - r, a_min=0.0, a_max=1.0)[..., None, None, None]
            * dudt,
        )
        loss = jnp.sum(jnp.square(u - u_target), axis=(-1, -2, -3))

        if self.adaptive_weight_power > 0.0:
            ada_wt = jnp.power(loss + 1e-2, self.adaptive_weight_power)
            loss = loss / jax.lax.stop_gradient(ada_wt)
        loss = jnp.mean(loss)

        velocity_loss = jnp.where(
            jnp.equal(t, r)[..., None, None, None],
            jnp.square(u - v),
            jnp.zeros_like(u),
        )
        velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

        return _model.StepOutputs(
            scalars={"loss": loss, "velocity_loss": velocity_loss},
            images={"samples": sample},
        )

    @typing_extensions.override
    def predict_step(
        self,
        *,
        params: jaxtyping.PyTree,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Any:
        # TODO (juanwulu): implement predict step
        raise NotImplementedError("Predict step is not implemented yet.")

    @typing_extensions.override
    def training_step(
        self,
        *,
        state: _train_state.TrainState,
        batch: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        del kwargs  # unused

        local_rng = jax.random.fold_in(rngs, jax.process_index())
        tr_rng = jax.random.fold_in(local_rng, 0)
        mask_rng = jax.random.fold_in(local_rng, 1)
        e_rng = jax.random.fold_in(local_rng, 2)

        image, label = batch["image"], batch["label"]
        batch_dims = image.shape[:-3]

        # step 1: randomly sample begin and end timestamps
        t, r = sample_t_r(
            key=tr_rng,
            shape=batch_dims,
            dtype=image.dtype,
            distribution=self.timestamp_sampler,
            **self.timestamp_sampler_kwargs,
        )
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        # ensure a portion of overlap between t and r
        r_neq_t_mask = jnp.greater_equal(
            jax.random.uniform(
                key=mask_rng,
                shape=batch_dims,
                dtype=image.dtype,
                minval=0.0,
                maxval=1.0,
            ),
            self.timestamp_overlap_rate,
        )
        r = jnp.where(r_neq_t_mask, t, r)

        # sample noise e ~ N(0, I)
        e = jax.random.normal(key=e_rng, shape=image.shape, dtype=image.dtype)

        # generate intermediate z(t)
        z = jnp.add(
            (1 - t[..., None, None, None]) * image,
            t[..., None, None, None] * e,
        )

        # calculate velocity v
        v = e - image

        # step 2: compute the loss
        def _loss_fn(params: PyTree) -> typing.Tuple[jax.Array, jax.Array]:
            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            u, dudt = jax.jvp(
                self._u_fn(
                    label=label,
                    params=params,
                    deterministic=False,
                ),
                (z, r, t),
                (v, drdt, dtdt),
            )
            u_target = jax.lax.stop_gradient(
                v
                - jnp.clip(t - r, a_min=0.0, a_max=1.0)[..., None, None, None]
                * dudt
            )

            # step 3: compute the loss
            loss = jnp.sum(jnp.square(u - u_target), axis=(-1, -2, -3))
            if self.adaptive_weight_power > 0.0:
                ada_wt = jnp.power(loss + 1e-2, self.adaptive_weight_power)
                loss = loss / jax.lax.stop_gradient(ada_wt)
            loss = jnp.mean(loss)

            # calculate velocity loss for monitoring
            velocity_loss = jnp.where(
                jnp.equal(t, r)[..., None, None, None],
                jnp.square(u - v),
                jnp.zeros_like(u),
            )
            velocity_loss = jnp.sum(velocity_loss, axis=(-1, -2, -3)).mean()

            return loss, velocity_loss

        grad_fn = jax.value_and_grad(_loss_fn, argnums=0, has_aux=True)
        (loss, velocity_loss), grads = grad_fn(state.params)
        loss = jax.lax.pmean(loss, axis_name="batch")
        velocity_loss = jax.lax.pmean(velocity_loss, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        return (
            new_state,
            _model.StepOutputs(
                scalars={"loss": loss, "velocity_loss": velocity_loss},
            ),
        )

    def _u_fn(
        self,
        *,
        label: jax.Array,
        params: frozen_dict.FrozenDict,
        deterministic: bool = True,
    ) -> typing.Callable[[jax.Array, jax.Array, jax.Array], typing.Any]:
        r"""Returns the average velocity function `u(z_t, r, t)`."""
        if self.timestamp_cond == "t_and_r":
            return lambda z_t, r, t: self.network.apply(
                variables={"params": params},
                image=z_t,
                label=label,
                begin=r,
                end=t,
                deterministic=deterministic,
            )
        elif self.timestamp_cond == "t_and_t_minus_r":
            return lambda z_t, r, t: self.network.apply(
                variables={"params": params},
                image=z_t,
                label=label,
                begin=t - r,
                end=t,
                deterministic=deterministic,
            )
        elif self.timestamp_cond == "t_and_r_and_t_minus_r":
            # TODO: implement this
            raise NotImplementedError(
                "Conditioning on (t, r, t - r) is not implemented yet."
            )
        elif self.timestamp_cond == "t_minus_r":
            return lambda z_t, r, t: self.network.apply(
                variables={"params": params},
                image=z_t,
                label=label,
                begin=t - r,
                end=None,
                deterministic=deterministic,
            )
        else:
            raise ValueError(
                f"Unsupported timestamp condition: {self.timestamp_cond}. "
                'Must be one of ["t_and_r", "t_and_t_minus_r", '
                '"t_and_r_and_t_minus_r", "t_minus_r"].'
            )
