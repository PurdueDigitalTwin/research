# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment
# Reference: https://arxiv.org/abs/2110.06169
################################################
# NOTE: We might need to use clipped double Q-learning similar to TD3
# to mitigate overestimation bias in Q-learning. 
# Reference: https://arxiv.org/pdf/1802.09477


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


# Define the IQLModel class by extending the base Model class
class IQLModel(_model.Model):
    r"""Implicit Q-learning model."""

    def __init__(
        self,
        action_space_dim: int,
        tau: float,
        gamma: float,
        beta: float
    ) -> None:
        r"""Instantiates an IQL model.
        Three networks in IQL: value network, Q-network, and policy network.

        Args:
            action_space_dim (int): Dimension of the action space.
            tau (float): Expectile parameter for value learning.
            gamma (float): Discount factor for future rewards.
            beta (float): inverse temperature for policy learning.
            
        """
        self._action_space_dim = action_space_dim
        self._tau = tau
        self._gamma = gamma
        self._beta = beta
        
        self._value_network = policy.MlpPolicy(
            features=256,
            out_features=1,
            num_layers=2,
        )
        self._q_network = policy.MlpPolicy(
            features=256,
            out_features=action_space_dim,
            num_layers=2,
        )
        self._policy_network = policy.MlpPolicy(
            features=256,
            out_features=action_space_dim,
            num_layers=2,
        )
    

    @typing_extensions.override
    def init(
        self,
        *,
        batch: structure.StepTuple,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        r"""Initializes value network, Q-network, and policy network parameters.

        Args:
            batch (StepSample): A sample of state transition for initialization.
            rngs (jax.random.PRNGKey): Random number generator key.

        Returns:
            A tuple of (value_params, q_params, policy_params).
        """
        del kwargs
        value_params = self._value_network.init(rngs, batch.state)
        q_params = self._q_network.init(rngs, batch.state)
        policy_params = self._policy_network.init(rngs, batch.state)

        # Print the model summary for analysis
        print("Value Network Summary:")
        print(self._value_network.tabulate(rngs, batch.state))
        print("Q Network Summary:")
        print(self._q_network.tabulate(rngs, batch.state))
        print("Policy Network Summary:")
        print(self._policy_network.tabulate(rngs, batch.state))

        return value_params, q_params, policy_params
    
    
    @typing_extensions.override
    def forward(
        self,
        *,
        params: typing.Any,
        batch: structure.StepTuple,
        **kwargs,
    ) -> typing.Any:
        r"""Forward pass the IQL model to compute value, Q-values, and policy 
        outputs.

        Args:
            params (Any): A tuple of (value_params, q_params, policy_params).
            batch (StepSample): A sample of state transition for forward pass.
        
        Returns:
            A tuple of (value, q_values, policy_output).
        """
        del kwargs

        value_params, q_params, policy_params = params

        value_output = self._value_network.apply(value_params, batch.state)
        q_output = self._q_network.apply(q_params, batch.state)
        policy_output = self._policy_network.apply(policy_params, batch.state)

        # May add some assertions here to check the outputs

        return value_output, q_output, policy_output
    

    def _expectile_loss(
        self,
        value: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        r"""Computes the expectile loss for value learning.

        Args:
            value (jax.Array): The predicted value from the value network.
            target (jax.Array): The target value computed from the Q-network.
        
        Returns:
            The expectile loss for value learning.
        """
        diff = target - value
        weight = jnp.where(diff > 0, self._tau, 1 - self._tau)
        return weight * (diff ** 2)
    

    @typing_extensions.override
    def training_step(
        self,
        *,
        params: typing.Any,
        batch: structure.StepTuple,
        train_state: _train_state.TrainState,
        target_params: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Any:
        r"""Performs a training step for the IQL model.

        Args:
            params (Any): A tuple of (value_params, q_params, policy_params).
            batch (StepSample): A sample of state transition for training step.
            **kwargs: Keyword arguments consumed by the training step.
        
        Returns:
            A tuple of (value_loss, q_loss, policy_loss).
        """
        del kwargs

        value_params, q_params, policy_params = params

        def _value_loss_fn(value_params: jaxtyping.PyTree) -> jax.Array:
            value_output = self._value_network.apply(value_params, batch.state)
            value_output = typing.cast(jax.Array, value_output)

            q1_target = self._q_network.apply(target_params[0], batch.state, \
                                              batch.action)
            q2_target = self._q_network.apply(target_params[1], batch.state, \
                                              batch.action)
            q1_target = typing.cast(jax.Array, q1_target)
            q2_target = typing.cast(jax.Array, q2_target)

            q_target_min = jnp.minimum(q1_target, q2_target)

            # Compute value loss based on expectile regression
            # NOTE: Optax has no exceptile loss, so we manually define one.
            # NOTE: how we compute the expectation over data samples?
            value_loss = self._expectile_loss(value_output, q_target_min)

            return jnp.mean(value_loss)
        
        def _q_loss_fn(q_params: jaxtyping.PyTree) -> jax.Array:
            q_output = self._q_network.apply(q_params, batch.state)
            q_output = typing.cast(jax.Array, q_output)

            next_value_output = self._value_network.apply(
                value_params,
                batch.next_state,
            )
            next_value_output = typing.cast(jax.Array, next_value_output)

            # Compute Q-loss based on TD error
            target_q = batch.reward + self._gamma * \
                (1 - jnp.asarray(batch.done)) * next_value_output
            q_loss = jnp.mean((q_output - target_q) ** 2)

            return q_loss
        
        def _policy_loss_fn(policy_params: jaxtyping.PyTree) -> jax.Array:
            policy_output = self._policy_network.apply(policy_params, batch.state)
            policy_output = typing.cast(jax.Array, policy_output)

            q1_target = self._q_network.apply(target_params[0], batch.state, \
                                              batch.action)
            q2_target = self._q_network.apply(target_params[1], batch.state, \
                                              batch.action)
            q1_target = typing.cast(jax.Array, q1_target)
            q2_target = typing.cast(jax.Array, q2_target)

            q_policy_min = jnp.minimum(q1_target, q2_target)

            value_output = self._value_network.apply(value_params, batch.state)
            value_output = typing.cast(jax.Array, value_output)

            # Compute policy loss based on advantage-weighted regression
            advantage = q_policy_min - value_output
            policy_loss = -jnp.mean(jnp.exp(self._beta * advantage) * \
                                    jnp.log(policy_output))

            return policy_loss
        
        value_loss, value_grads = jax.value_and_grad(_value_loss_fn)(value_params)
        q_loss, q_grads = jax.value_and_grad(_q_loss_fn)(q_params)
        policy_loss, policy_grads = jax.value_and_grad(_policy_loss_fn)(policy_params)

        value_grads = jax.lax.pmean(value_grads, axis_name="batch")
        q_grads = jax.lax.pmean(q_grads, axis_name="batch")
        policy_grads = jax.lax.pmean(policy_grads, axis_name="batch")

        new_value_train_state = train_state.apply_gradients(grads=value_grads)
        new_q_train_state = train_state.apply_gradients(grads=q_grads)
        new_policy_train_state = train_state.apply_gradients(grads=policy_grads)

        outputs = _model.StepOutputs(
            scalars={
                "value_loss": value_loss.mean(),
                "q_loss": q_loss.mean(),
                "policy_loss": policy_loss.mean(),
            }
        )

        new_train_state = (new_value_train_state, new_q_train_state, \
                           new_policy_train_state)

        return new_train_state, outputs
