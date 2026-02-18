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
from src.utilities import logging
from src.projects.rl import policy
from src.projects.rl import structure as _structure


# create a PPO model calss by extending the base Model class
class PPOModel(_model.Model):
    r"""PPO model class."""

    @typing_extensions.override
    def __init__(
        self,
        action_space_dim: int,
        gamma: float,
        lambda_gae: float,
        clip_epsilon: float,
        value_coeff: float,
        entropy_coeff: float,
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
        self._network = policy.ActorCriticPolicy(
            features=64,
            out_features=action_space_dim,
            num_layers=2,
            activation_fn=jax.nn.tanh,
        )
        self._value_coeff = value_coeff
        self._entropy_coeff = entropy_coeff

    @property
    @typing_extensions.override
    def network(self) -> policy.ActorCriticPolicy:
        r"""Returns the policy network instance.

        Args:
            None.
        Returns:
            policy.ActorCriticPolicy: The policy network instance.
        """

        return self._network
    
    @typing_extensions.override
    def init(
        self,
        *,
        state: jax.Array,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        r"""Initializes policy network parameters.

        Args:
            state (jax.Array): A sample state input for initialization.
            rngs (jax.random.PRNGKey): Random number generator key.

        Returns:
            PyTree: Initialized parameters of the policy network.
        """
        del kwargs

        params = self.network.init(rngs, state)

        # We may need to print the model summary for analysis.
        # Note that each layer has its own kernel matrix and bias vector (if
        # use_bias=True)
        _tabulate_fn = nn.summary.tabulate(
            self.network,
            rngs,
            console_kwargs=dict(width=120),
        )
        print(_tabulate_fn(state))

        return params
    
    @typing_extensions.override
    def forward(
        self,
        *,
        state: jax.Array,
        params: jaxtyping.PyTree,
        rngs: typing.Any = None,
        **kwargs,
    ) -> typing.Tuple[jax.Array, jax.Array]:
        r"""Forward pass the policy network to compute action logits and state value.
        
        Args:
            state (jax.Array): Input state array of shape `(*, D)`.
            params (PyTree): Current parameters of the policy network.
            rngs (jax.random.PRNGKey, optional): Random number generator key for
                stochastic operations. Default is ``None``.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            Action logits and state values for the given state.
        """
        del kwargs
        
        logits, values = self._network.apply(params, rngs=rngs, inputs=state)
        assert isinstance(logits, jax.Array)
        assert isinstance(values, jax.Array)

        return logits, values
    
    @typing_extensions.override
    def compute_loss(
        self,
        *,
        state: jax.Array,
        action: jax.Array,
        params: jaxtyping.PyTree,
        log_old_act_probs: jax.Array,
        advantages: jax.Array,
        value_targets: jax.Array,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        r"""Computes the PPO surrogate loss.

        Args:
            state (jax.Array): Input state array of shape `(*, D)`.
            action (jax.Array): Actions taken in the rollout, with shape `(*,)` 
                where each entry is an integer representing the action index.
            params (PyTree): Current parameters of the policy network.
            log_old_act_probs (jax.Array): Log action probabilities of the old 
                policy.
            advantages (jax.Array): Computed advantages for each transition in 
                the rollout.
            rngs (jax.random.PRNGKey): Random number generator key for stochastic
                operations.
            deterministic (bool, optional): Whether to compute loss in 
                deterministic mode (i.e., no sampling from the policy).

        Returns:
            jax.Array: The computed PPO surrogate loss.
        """
        del kwargs

        # Compute the current policy's action probabilities
        logits, values = self._network.apply(params, rngs=rngs, inputs=state)

        assert isinstance(logits, jax.Array)
        assert isinstance(values, jax.Array)

        # Squeeze values to match the shape of value_targets (rollout_steps,)
        # This prevents optax.squared_error from broadcasting into an N x N matrix
        values = values.squeeze(axis=-1)

        # Apply log softmax to get action probabilities
        # dimension: (rollout_steps, action_space_dim)
        curr_log_probs = jax.nn.log_softmax(logits)

        # Gather the log probabilities of the actions taken in the rollout
        # dimension: (rollout_steps,)
        log_act_probs = jnp.take_along_axis(
            curr_log_probs, 
            action.astype(jnp.int32)[..., None],
            axis=-1
        ).squeeze(axis=-1)

        # Compute the ratio of current to old action probabilities
        ratios = jnp.exp(log_act_probs - log_old_act_probs)

        # Normalize the advantages (not mention by the reference)
        advantages = (advantages - jnp.mean(advantages)) / \
            (jnp.std(advantages) + 1e-8)

        # Compute the surrogate loss L^CLIP
        clip_epsilon = self._clip_epsilon
        surrogate_loss = jnp.minimum(
            ratios * advantages,
            jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
        )
        
        # Average the surrogate loss over the rollout steps
        surrogate_loss = jnp.mean(surrogate_loss)

        # Compute the Value Function loss L^VF
        # NOTE: when should I add stop_gradient
        value_targets = lax.stop_gradient(value_targets)
        
        # Average the value loss over the rollout steps
        value_loss = jnp.mean(optax.squared_error(values, value_targets))

        # According to the original paper, they don't use an entropy bonus
        entropy_bonus = 0.0

        # We want to minimize the surrogate total loss
        total_loss = -surrogate_loss + self._value_coeff * value_loss - \
            self._entropy_coeff * entropy_bonus
        
        assert isinstance(total_loss, jax.Array)

        step_outputs = _model.StepOutputs(scalars={"loss": total_loss})

        return total_loss, step_outputs