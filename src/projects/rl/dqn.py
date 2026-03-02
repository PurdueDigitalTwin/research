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
from src.projects.rl import policy
from src.projects.rl import structure


# Define the DQNModel class by extending the base Model class
class DQNModel(_model.Model):
    r"""Deep Q-learning model."""

    def __init__(
        self,
        action_space_dim: int,
        gamma: float,
        use_double: bool = False,
    ) -> None:
        r"""Instantiates a DQN model.

        Args:
            action_space_dim (int): Dimension of the action space.
            gamma (float): Discount factor for future rewards.
            use_double (bool, optional): Whether to train the DQN model with
                double Q-learning. See ``https://arxiv.org/abs/1509.06461``.
        """
        self._action_space_dim = action_space_dim
        self._gamma = gamma
        self._network = policy.MlpPolicy(
            features=256,
            out_features=action_space_dim,
            num_layers=3,
        )
        self._use_double = use_double

    @typing_extensions.override
    def init(
        self,
        *,
        batch: structure.StepTuple,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        r"""Initializes Q-network parameters.

        Args:
            batch (StepSample): A sample of state transition for initialization.
            rngs (jax.random.PRNGKey): Random number generator key.

        Returns:
            A tuple of (q_params, target_params).
        """
        del kwargs
        q_params = self._network.init(rngs, batch.state)

        # We may need to print the model summary for analysis.
        # Note that each layer has its own kernel matrix and bias vector (if
        # use_bias=True)
        _tabulate_fn = nn.summary.tabulate(
            self._network,
            rngs,
            console_kwargs=dict(width=120),
        )
        print(_tabulate_fn(batch.state))

        return q_params

    @typing_extensions.override
    def forward(
        self,
        *,
        batch: structure.StepTuple,
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
        del deterministic, kwargs, rngs
        out = self._network.apply(params, batch.state)
        assert isinstance(out, jax.Array)

        return _model.StepOutputs(output=out)

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: structure.StepTuple,
        state: _train_state.TrainState,
        target_params: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        def _loss_fn(params: jaxtyping.PyTree) -> jax.Array:
            # Compute Q-values for current states
            q_values = self._network.apply(params, batch.state, rngs=local_rng)
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
                rngs=local_rng,
            )
            assert isinstance(next_q_values, jax.Array)

            if self._use_double:
                # use Double DQN
                next_q_values_online = self._network.apply(
                    params,
                    batch.next_state,
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
        new_state = state.apply_gradients(grads=grads)

        outputs = _model.StepOutputs(scalars={"loss": loss.mean()})

        return new_state, outputs
