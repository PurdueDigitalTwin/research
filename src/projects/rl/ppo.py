# The PPO class

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
from src.projects.rl import structure as _structure


# create a PPO model calss by extending the base Model class
class PPO(_model.Model):
    r"""PPO model class."""

    @typing_extensions.override
    def __init__(
        self,
        action_space_dim: int,
        gamma: float,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
    ) -> None:
        r""" Instantiates a PPO model.
        
        Args:
            action_space_dim (int): Dimension of the action space.
            gamma (float): Discount factor for future rewards.
            clip_epsilon (float, optional): Clipping parameter for PPO's surrogate
                objective. It can be tuned but generally it's 0.2.
            lambda_gae (float, optional): GAE parameter for advantage estimation.
            value_coeff (float, optional): Coefficient for the value function loss 
                in the total loss.
            entropy_coeff (float, optional): Coefficient for the entropy bonus in 
                the total loss.
        Returns:
            None.
        """

        self._action_space_dim = action_space_dim
        self._gamma = gamma
        self._clip_epsilon = clip_epsilon
        self._lambda_gae = lambda_gae

        # according to the original paper, the policy has two hidden layers with 
        # 64 units each and tanh activation function.
        self._network = policy.MlpPolicy(
            features=64,
            out_features=action_space_dim,
            num_layers=2,
            activation_fn=jax.nn.tanh,
        )
        self._value_coeff = value_coeff
        self._entropy_coeff = entropy_coeff

    @property
    @typing_extensions.override
    def network(self) -> policy.MlpPolicy:
        r"""Returns the policy network instance.

        Args:
            None.
        Returns:
            policy.MlpPolicy: The policy network instance.
        """

        return self._network
    
    @typing_extensions.override
    def init(
        self,
        *,
        batch: _structure.StepTuple,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        r"""Initializes policy network parameters.

        Args:
            batch (StepTuple): A sample of state transition for initialization.
            rngs (jax.random.PRNGKey): Random number generator key.

        Returns:
            PyTree: Initialized parameters of the policy network.
        """
        del kwargs

        params = self._network.init(rngs, batch.state)

        # We may need to print the model summary for analysis.
        # Note that each layer has its own kernel matrix and bias vector (if
        # use_bias=True)
        _tabulate_fn = nn.summary.tabulate(
            self.network,
            rngs,
            console_kwargs=dict(width=120),
        )
        print(_tabulate_fn(batch.state))

        return params
    
    @typing_extensions.override
    def loss_fn(
        self,
        params: jaxtyping.PyTree,
        batch: _structure.StepTuple,
        old_action_probs: jax.Array,
        advantages: jax.Array,
        deterministic: bool = False,
        **kwargs,
    ) -> jax.Array:
        r"""Computes the PPO surrogate loss.

        Args:
            params (PyTree): Current parameters of the policy network.
            batch (StepTuple): A batch of state transitions for loss computation.
            old_action_probs (jax.Array): Action probabilities of the old policy.
            advantages (jax.Array): Computed advantages for each transition in 
                the batch.
            deterministic (bool, optional): Whether to compute loss in 
                deterministic mode (i.e., no sampling from the policy).

        Returns:
            jax.Array: The computed PPO surrogate loss.
        """
        del deterministic, kwargs

        # Compute the current policy's action probabilities
        action_probs = self._network.apply(params, batch.state)

        # Compute the probability of the taken actions under the current policy
        action_indices = jnp.arange(batch.action.shape[0]), \
            batch.action.astype(jnp.int32)
        current_action_probs = action_probs[action_indices]

        # Compute the ratio of current to old action probabilities
        assert old_action_probs.shape == current_action_probs.shape, \
            "Shape of old_action_probs and current_action_probs must match."
        old_action_probs = jnp.clip(old_action_probs, 1e-8, 1.0)  
        ratios = current_action_probs / (old_action_probs + 1e-8)

        # NOTE: clip the advantages for stability. Should I do this?
        advantages = jnp.clip(advantages, -10.0, 10.0)

        # Compute the surrogate loss L^CLIP
        clip_epsilon = self._clip_epsilon
        surrogate_loss = jnp.minimum(
            ratios * advantages,
            jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
        )

        # compute values of the current and next state
        values = self._network.apply(params, batch.state)
        next_values = self._network.apply(params, batch.next_state)

        # Compute the Value Function loss L^VF
        value_targets = lax.stop_gradient(batch.reward + self._gamma * \
                                          next_values * (1.0 - batch.done))
        value_loss = jnp.mean(optax.squared_error(values, value_targets))

        # According to the original paper, they don't use an entropy bonus
        entropy_bonus = 0.0

        # we need to maximize the surrogate total loss
        # NOTE: should I mean the surrogate loss?
        total_loss = -jnp.mean(surrogate_loss) + self._value_coeff * \
            value_loss - self._entropy_coeff * entropy_bonus
        return total_loss
