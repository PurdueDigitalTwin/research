import typing

from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
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
            activation_fn=jax.nn.relu,
        )
        self._use_double = use_double

    @property
    @typing_extensions.override
    def network(self) -> policy.MlpPolicy:
        return self._network

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
        q_params = self.network.init(rngs, batch.state)

        # We may need to print the model summary for analysis.
        # Note that each layer has its own kernel matrix and bias vector (if
        # use_bias=True)
        _tabulate_fn = nn.summary.tabulate(
            self.network,
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
        out = self.network.apply(params, batch.state)
        assert isinstance(out, jax.Array)

        return _model.StepOutputs(output=out)

    @typing_extensions.override
    def compute_loss(
        self,
        *,
        batch: structure.StepTuple,
        params: typing.Any,
        target_params: typing.Any,
        rngs: typing.Any,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        r"""Computes the Q-learning loss using the Bellman equation.

        Args:
            batch (StepTuple): A batch of transition tuples ``(s, a, s', r)``.
            params (Any): State-action value neetwork parameters.
            target_params (Any): Target state-action network parameters.
            rngs (Any): Random key for reproducibility.
            deterministic (bool, optional): Whether to forward pass the network
                in deterministic mode. Not used in this function.

        Returns:
            A tuple of ``(loss, outputs)`` with scalar loss and other outputs.
        """
        del deterministic, kwargs

        # Compute Q-values for current states
        q_values = self.network.apply(params, batch.state, rngs=rngs)
        assert isinstance(q_values, jax.Array)

        # Select Q-values for the action actually taken
        if batch.action is None:
            raise ValueError("Action is required for Q-learning.")
        one_hot_action = jax.nn.one_hot(
            # NOTE: enusure DQN takes in discrete action indexes
            batch.action[..., 0].astype(jnp.int32),
            num_classes=q_values.shape[-1],
        )
        q_values = jnp.sum(q_values * lax.stop_gradient(one_hot_action), -1)

        # Compute Q-values for next states using target network
        next_q_values = self.network.apply(
            target_params,
            batch.next_state,
            rngs=rngs,
        )
        assert isinstance(next_q_values, jax.Array)

        if self._use_double:
            # use Double DQN
            next_q_values_online = self.network.apply(
                params,
                batch.next_state,
                rngs=rngs,
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
        q_target = lax.stop_gradient(r + self._gamma * max_next_q * (1.0 - d))

        # Compute loss as mean squared error
        loss = jnp.mean(optax.squared_error(q_values, q_target))
        step_outputs = _model.StepOutputs(scalars={"loss": loss})

        return loss, step_outputs
