import copy
import typing

from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.rl import common
from src.utilities import logging


# Define the DQNModel class by extending the base Model class
class DQNModel(common.BaseAgent):
    r"""Deep Q-learning policy."""

    def __init__(
        self,
        action_space_dim: int,
        network: typing.Callable[..., nn.Module],
        gamma: float,
        q_target_update_freq: int,
        use_double: bool = False,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        r"""Instantiates a DQN model.

        Args:
            action_space_dim (int): Dimension of the action space.
            gamma (float): Discount factor for future rewards.
            q_target_update_freq (int): Number of training steps before syncing
                the target Q-network parameters with the online one.
            use_double (bool, optional): Whether to train the DQN model with
                double Q-learning. See ``https://arxiv.org/abs/1509.06461``.
            dtype (Any, optional): Data type for computation.
                Default is ``None``.
            param_dtype (Any, optional): Data type for the network parameters.
                Default is ``None``.
            precision (Any, optional): Precision for the computation.
                Default is ``None``.
        """
        self._action_space_dim = action_space_dim
        self._gamma = gamma
        self._network = network(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self._q_target_update_freq = q_target_update_freq
        self._use_double = use_double

    @typing_extensions.override
    def init(
        self,
        *,
        batch: common.StepTuple,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
        r"""Initializes Q-network parameters.

        Args:
            batch (StepSample): A sample of state transition for initialization.
            rngs (jax.random.PRNGKey): Random number generator key.

        Returns:
            A tuple of ``(params, mutables)``.
        """
        del kwargs
        q_params = self._network.init(rngs, batch.state, deterministic=True)

        # We may need to print the model summary for analysis.
        # Note that each layer has its own kernel matrix and bias vector (if
        # use_bias=True)
        _tabulate_fn = nn.summary.tabulate(
            self._network,
            rngs,
            console_kwargs=dict(width=120),
        )
        print(_tabulate_fn(batch.state, deterministic=True))

        return q_params, {"target_params": q_params}

    @typing_extensions.override
    def forward(
        self,
        *,
        batch: common.StepTuple,
        params: jaxtyping.PyTree,
        rngs: typing.Any = None,
        deterministic: bool = True,
        **kwargs,
    ) -> _model.StepOutputs:
        r"""Compute Q-values of ALL possible actions for the given state.

        .. note::

            For ``cartpole``, it will be
            ``[q_value(action=left), q_value(action=right)]``

        Args:
            batch (StepTuple): State transition observation.
            rngs (Any, optional): Random key generator. Default is ``None``.
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `True`.
            params (FrozenDict): The model parameters.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            Q-values for the given state.
        """
        del kwargs, rngs
        out = self._network.apply(
            params,
            batch.state,
            deterministic=deterministic,
        )
        assert isinstance(out, jax.Array)

        return _model.StepOutputs(output=out)

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: common.StepTuple,
        state: _train_state.TrainState,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        del kwargs  # unused parameters

        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        # extract current target parameters
        assert isinstance(state.mutables, dict)
        target_params = state.mutables["target_params"]

        def _loss_fn(params: jaxtyping.PyTree) -> jax.Array:
            # Compute Q-values for current states
            q_values = self._network.apply(
                params,
                batch.state,
                deterministic=False,
                rngs=local_rng,
            )
            assert isinstance(q_values, jax.Array)

            # Select Q-values for the action actually taken
            if batch.action is None:
                raise ValueError("Action is required for Q-learning.")
            one_hot_action = jax.nn.one_hot(
                # NOTE: enusure DQN takes in discrete action indexes
                batch.action[..., 0].astype(jnp.int32),
                num_classes=q_values.shape[-1],
            )
            q_values = q_values * lax.stop_gradient(one_hot_action)
            q_values = jnp.sum(q_values, axis=-1)

            # Compute Q-values for next states using target network
            next_q_values = self._network.apply(
                target_params,
                batch.next_state,
                deterministic=True,
                rngs=local_rng,
            )
            assert isinstance(next_q_values, jax.Array)

            if self._use_double:
                # use Double DQN
                next_q_values_online = self._network.apply(
                    params,
                    batch.next_state,
                    deterministic=True,
                    rngs=local_rng,
                )
                assert isinstance(next_q_values_online, jax.Array)
                max_next_action = jnp.argmax(next_q_values_online, axis=-1)
                max_next_q = (
                    jnp.take_along_axis(
                        next_q_values,
                        max_next_action[:, None],
                        axis=1,
                    )
                    .astype(jnp.float32)
                    .squeeze(axis=-1)
                )
            else:
                # traditional DQN: simply max over action dimension
                max_next_q = jnp.max(next_q_values, axis=-1)

            # Compute TD-target using the Bellman equation
            if batch.done is None:
                d = jnp.zeros_like(max_next_q)
            else:
                d = batch.done.astype(max_next_q.dtype)

            if batch.reward is None:
                r = jnp.zeros_like(max_next_q)
            else:
                r = batch.reward.astype(max_next_q.dtype)
            q_tgt = lax.stop_gradient(r + self._gamma * max_next_q * (1.0 - d))

            # Compute loss as mean squared error
            loss = jnp.mean(optax.squared_error(q_values, q_tgt))

            return loss

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(scalars={"loss": loss.mean()})

        return new_state, outputs

    def configure_train_state(
        self,
        params: typing.Any,
        tx: optax.GradientTransformation,
        **kwargs,
    ) -> _train_state.TrainState:
        del kwargs  # unused argument

        return _train_state.TrainState.create(
            params=params,
            tx=tx,
            mutables={"target_params": copy.deepcopy(params)},
        )

    @property
    def is_on_policy(self) -> bool:
        return False

    @typing_extensions.override
    def on_train_batch_end(
        self,
        *,
        state: _train_state.TrainState,
        step: int,
        **kwargs,
    ) -> typing.Any:
        del kwargs

        if step % self._q_target_update_freq == 0:
            state = state.replace(mutables={"target_params": state.params})
            logging.rank_zero_info("Target network synced!")

        return state
