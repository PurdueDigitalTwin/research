import typing

from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
from src.projects.rl.StepTuple import StepTuple
from src.projects.rl.MlpNetwork import MlpPolicy


# Define the DQNModel class by extending the base Model class
class DQNModel(_model.Model):
    r"""Deep Q-learning model."""

    def __init__(self, action_space_dim: int, gamma: float) -> None:
        r"""Instantiates a DQN model.

        Args:
            action_space_dim (int): Dimension of the action space.
            gamma (float): Discount factor for future rewards.
        """
        self._action_space_dim = action_space_dim
        self._gamma = gamma
        self._network = MlpPolicy(
            features=128,
            out_features=action_space_dim,
            num_layers=3,
        )

    @property
    @typing_extensions.override
    def network(self) -> MlpPolicy:
        return self._network

    @typing_extensions.override
    def init(
        self,
        *,
        batch: StepTuple,
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
        # Note that each layer has its own kernel matrix and bias vector (if use_bias=True)
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
        batch: StepTuple,
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
        batch: StepTuple,
        params: typing.Any,
        target_params: typing.Any,
        rngs: typing.Any,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[
        jax.Array, _model.StepOutputs
    ]:  # NOTE: why we need another stepoutputs here?
        r"""Computes the DQN loss using the Bellman equation."""
        del deterministic, kwargs

        # Compute Q-values for current states
        q_values = self.network.apply(params, batch.state, rngs=rngs)
        assert isinstance(q_values, jax.Array)

        # Select Q-values for the action actually taken
        one_hot_action = jax.nn.one_hot(
            batch.action,
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

        # Simply max over action dimension, which is the drawback of DQN
        max_next_q = jnp.max(next_q_values, axis=1)

        # Compute TD-target using the Bellman equation
        if batch.done is None:
            done = jnp.zeros_like(max_next_q)
        else:
            done = batch.done

        TD_target = lax.stop_gradient(
            batch.reward + self._gamma * max_next_q * (1.0 - done)
        )

        # Compute loss as mean squared error
        loss = jnp.mean(optax.squared_error(q_values, TD_target))
        step_outputs = _model.StepOutputs(scalars={"loss": loss})

        return loss, step_outputs
