import functools
import platform
import typing

from flax import linen as nn
from flax import traverse_util
import jax
from jax import numpy as jnp
import jaxtyping
import optax

from src.core import model as _model
from src.core import train_state as _train_state
from src.data import huggingface as _hf_dataset
from src.data import preprocess as _preprocess
from src.projects.generative.model import unet as _unet
from src.utilities import logging as _logging


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


def _make_weight_decay_mask(params: jaxtyping.PyTree) -> jaxtyping.PyTree:
    r"""Exclude biases and normalization layer parameters from weight decay.

    Args:
        params (PyTree): A tree of model parameters.

    Returns:
        A tree with the same structure as `params`, where each leaf is a boolean
        indicating whether to apply weight decay to the corresponding parameter.
    """
    flat_params = traverse_util.flatten_dict(params)

    def should_decay(path):
        # exclude biases
        if path[-1] == "bias":
            return False
        # exclude LayerNorm scale and bias
        if path[-1] == "scale":
            return False
        # exclude embedding parameters
        if "embedding" in path[-1]:
            return False
        # by default, apply weight decay
        return True

    mask = {path: should_decay(path) for path in flat_params}
    return traverse_util.unflatten_dict(mask)


class UAFlowUNetModule(nn.Module):
    r"""ADM U-Net architecture for UA-Flow model.

    Args:
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
    ) -> typing.Tuple[jax.Array, jax.Array]:
        r"""Forward pass of the `DDPMUNetModule`.

        Args:
            inputs (jax.Array): Input data of shape `(*, H, W, C)`.
            timestep (jax.Array): Time step array of shape `(*)`.
            deterministic (typing.Optional[bool]): Whether to apply dropout.
                Merges with the module level attribute `deterministic`.

        Returns:
            A tuple of mean and log-variance predictions, each is an array with
                the shape of `(*, H, W, out_features)`.
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
        backbone = _unet.HoNetwork(
            features=self.features,
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
        output = backbone(
            inputs,
            cond=t_emb,
            deterministic=m_deterministic,
            with_head=False,
        )

        # mean and log-variance heads
        out_features = (
            self.out_features
            if isinstance(self.out_features, int)
            else inputs.shape[-1]
        )
        mean = nn.Conv(
            features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="mean_head",
        )(output)
        log_var = nn.Conv(
            features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="log_var_head",
        )(output)

        return mean, log_var


class UAFlowUNetModel:
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: int,
        ch_mults: typing.Sequence[int],
        attn_resolutions: typing.Sequence[int],
        num_res_blocks: int,
        dropout_rate: float,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        self._in_channels = in_channels
        self._image_size = image_size
        self._features = features
        self._dtype = dtype
        self._param_dtype = param_dtype

        # construct the ADM U-Net architecture
        self._network = UAFlowUNetModule(
            features=features,
            ch_mults=ch_mults,
            attn_resolutions=attn_resolutions,
            num_res_blocks=num_res_blocks,
            resample_with_conv=True,
            dropout_rate=dropout_rate,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            name="unet",
        )

    @property
    def network(self) -> UAFlowUNetModule:
        r"""UAFlowUNetModule: The UNet architecture for the UAFlow model."""
        return self._network

    def init(
        self,
        *,
        batch: jaxtyping.PyTree,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        del batch, kwargs  # unused

        variables = self._network.init(
            rngs,
            inputs=jnp.zeros(
                (1, self._image_size, self._image_size, self._in_channels),
                dtype=self._dtype,
            ),
            timestep=jnp.zeros((1,), dtype=self._dtype),
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
                            self._image_size,
                            self._image_size,
                            self._in_channels,
                        ),
                        dtype=self._dtype,
                    ),
                    timestep=jnp.zeros((1,), dtype=self._dtype),
                    deterministic=True,
                )
            )

        return variables["params"]

    def training_step(
        self,
        *,
        state: _train_state.TrainState,
        batch: jaxtyping.PyTree,
        deterministic: bool,
        rngs: jax.Array,
        with_variance: bool = False,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        r"""Computes and returns the training loss in a single gradient step.

        Args:
            state (TrainState): The current training state containing model
                parameters and optimizer state.
            batch (PyTree): A batch of training data.
            deterministic (bool): Whether to run the model in deterministic mode
                (e.g., disable dropout).
            rngs (jax.Array): Random generators for stochastic operations.
            with_variance (bool, optional): Whether to predict the variance and
                apply the `CUFM` loss. Default is `False`.

        Returns:
            A tuple of the updated training state and a dictionary
                containing the computed loss and other outputs.
        """

        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)
        flip_key, t_key, e_key, dropout_key = jax.random.split(local_rng, 4)

        # pre-processing
        image = batch.get("image")
        if not isinstance(image, jax.Array):
            raise ValueError("Invalid `image` in batch for training step.")
        *batch_dims, height, width, channels = image.shape
        image = image.astype(self._dtype).reshape(-1, height, width, channels)
        image = image * 2.0 - 1.0  # NOTE: scale to [-1, 1]
        mask = jax.random.uniform(key=flip_key, shape=image.shape[0:1]) < 0.5
        image = jnp.where(
            mask[:, None, None, None],
            jnp.flip(image, axis=-2),
            image,
        )

        # sample random time steps
        # TODO (juanwu): consider sampling from logit-normal
        t = jax.random.randint(
            key=t_key,
            shape=batch_dims,
            minval=0.0,
            maxval=1.0,
            dtype=self._dtype,
        ).reshape(-1, 1)
        t_expanded = t[..., None, None, :]

        # construct the cfm target
        e = jax.random.normal(key=e_key, shape=image.shape, dtype=self._dtype)
        x_t = (1 - t_expanded) * image + t_expanded * e
        v_target = jax.lax.stop_gradient(e - image)

        def cfm_loss(params: jaxtyping.PyTree) -> jax.Array:
            r"""Computes the conditional flow-matching loss.

            Args:
                params (jaxtyping.PyTree): A tree of model parameters.

            Returns:
                A scalar array the computed loss value.
            """
            v_pred, _ = self._network.apply(
                variables={"params": params},
                inputs=x_t,
                timestep=t,
                deterministic=deterministic,
                rngs={"dropout": dropout_key},
            )
            if not isinstance(v_pred, jax.Array):
                raise TypeError(
                    f"Output has a type of {type(v_pred)}, "
                    "but expect a ``jax.Array`` instead."
                )

            loss = optax.squared_error(v_pred, v_target).mean()
            return loss

        def cufm_loss(
            params: jaxtyping.PyTree,
        ) -> typing.Tuple[jax.Array, _model.StepOutputs]:
            raise NotImplementedError

        if with_variance:
            grad_fn = jax.value_and_grad(cufm_loss, has_aux=False)
            loss, grads = grad_fn(state.params)
        else:
            grad_fn = jax.value_and_grad(cfm_loss, has_aux=False)
            loss, grads = grad_fn(state.params)
        loss = jax.lax.pmean(loss, axis_name="batch")
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        output = _model.StepOutputs(
            scalars={"loss": loss}, histograms={"timestep": t}
        )

        return new_state, output

    def evaluation_step(
        self,
        *,
        params: jaxtyping.PyTree,
        batch: jaxtyping.PyTree,
        deterministic: bool,
        rngs: jax.Array,
        **kwargs,
    ) -> _model.StepOutputs:
        # TODO (juanwu): Implement the evaluation step for the UAFlow model.
        raise NotImplementedError

    def predict_step(
        self,
        *,
        params: jaxtyping.PyTree,
        batch: jaxtyping.PyTree,
        deterministic: bool,
        rngs: jax.Array,
        **kwargs,
    ) -> _model.StepOutputs:
        # TODO (juanwu): Implement the prediction step for the UAFlow model.
        raise NotImplementedError


################################################################################
# Entry points
def train(batch_size: int, seed: int) -> int:
    rng = jax.random.PRNGKey(seed=seed)

    # Log the current platform
    _logging.rank_zero_info("Running on platform: %s", platform.node())
    _logging.rank_zero_info(
        "Running on JAX backend: %s", jax.default_backend()
    )
    _logging.rank_zero_info(
        "Running on JAX process: %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    _logging.rank_zero_info("Running on JAX devices: %r", jax.devices())

    # build dataset
    _logging.rank_zero_info("Building dataset...")
    rng, data_key = jax.random.split(rng, num=2)
    _local_batch_size = batch_size * jax.local_device_count()
    dm = _hf_dataset.CIFAR10DataModule(
        batch_size=_local_batch_size,
        drop_remainder=True,
        shuffle_buffer_size=50_000,
        transform=_preprocess.chain(
            functools.partial(
                _preprocess.filter_keys,
                keys=["image", "label"],
            ),
            functools.partial(
                _preprocess.normalize,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        ),
        rng=data_key,
        use_cache=True,
    )
    _logging.rank_zero_info("Successfully built %s", dm.__class__.__name__)

    # build model
    _logging.rank_zero_info("Building model...")
    rng, model_key = jax.random.split(rng, num=2)
    model = UAFlowUNetModel(
        in_channels=3,
        image_size=32,
        features=128,
        ch_mults=(1, 2, 2, 2),
        attn_resolutions=(16,),
        num_res_blocks=4,
        dropout_rate=0.3,
    )
    params = model.init(batch=None, rngs=model_key)
    _logging.rank_zero_info("Successfully built %s", model.__class__.__name__)

    _logging.rank_zero_info("Building train state...")
    lr_scheduler = optax.constant_schedule(1e-4)
    optimizer = optax.adamw(
        learning_rate=lr_scheduler,
        b1=0.9,
        b2=0.95,
        weight_decay=0.01,
        mask=_make_weight_decay_mask,
    )
    state = _train_state.TrainState.create(
        params=params,
        tx=optimizer,
        ema_rate=0.999,
    )
    jax.block_until_ready(state)
    _logging.rank_zero_info("Successfully built train state.")

    return 1


__all__ = [
    "UAFlowUNetModule",
    "UAFlowUNetModel",
]
