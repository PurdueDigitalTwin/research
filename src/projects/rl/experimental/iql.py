# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment: mujoco/halfcheetah/medium-v0 from minari
# Reference: https://arxiv.org/abs/2110.06169
################################################
# NOTE: We might need to use clipped double Q-learning similar to TD3
# to mitigate overestimation bias in Q-learning.
# Reference: https://arxiv.org/pdf/1802.09477


import functools
import typing

import chex
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.rl import structure
from src.projects.rl.experimental import policy
from src.utilities import logging


# Define the IQLModel class by extending the base Model class
class IQLModel(_model.Model):
    r"""Implicit Q-learning model."""

    def __init__(
        self, action_space_dim: int, tau: float, gamma: float, beta: float
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
            activation=nn.relu,
        )
        # for Q network, input: s, a; output: Q(s, a)
        self._q_network = policy.MlpPolicy(
            features=256,
            out_features=1,
            num_layers=2,
            activation=nn.relu,
        )
        self._policy_network = policy.GaussianPolicy(
            features=256,
            out_features=action_space_dim,
            num_layers=2,
            activation=nn.tanh,
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
        rng_v, rng_q1, rng_q2, rng_p = jax.random.split(rngs, 4)

        value_params = self._value_network.init(rng_v, batch.state)

        if batch.state is None or batch.action is None:
            raise ValueError(
                "State and Action must not be None for IQL updates."
            )
        q_input = jnp.concatenate([batch.state, batch.action], axis=-1)
        q1_params = self._q_network.init(rng_q1, q_input)
        q2_params = self._q_network.init(rng_q2, q_input)
        q_params = (q1_params, q2_params)
        policy_params = self._policy_network.init(rng_p, batch.state)

        # Print the model summary for analysis
        print("Value Network Summary:")
        print(self._value_network.tabulate(rng_v, batch.state))
        print("Q1 Network Summary:")
        print(self._q_network.tabulate(rng_q1, q_input))
        print("Q2 Network Summary:")
        print(self._q_network.tabulate(rng_q2, q_input))
        print("Policy Network Summary:")
        print(self._policy_network.tabulate(rng_p, batch.state))

        return value_params, q_params, policy_params

    @typing_extensions.override
    def forward(
        self,
        *,
        params: typing.Any,
        batch: structure.StepTuple,
        **kwargs,
    ) -> typing.Any:
        r"""Forward pass the IQL model to compute value, Q-values, and policy outputs.

        Args:
            params (Any): A tuple of (value_params, q_params, policy_params).
            batch (StepSample): A sample of state transition for forward pass.

        Returns:
            A tuple of (value, q_values, policy_output).
        """
        del kwargs

        value_params, q_params, policy_params = params

        value_output = self._value_network.apply(value_params, batch.state)

        assert (
            batch.state is not None
        ), "State data is required for Q-network \
            forward pass."
        assert (
            batch.action is not None
        ), "Action data is required for Q-network \
            forward pass."
        q_input = jnp.concatenate([batch.state, batch.action], axis=-1)
        q_output = self._q_network.apply(q_params, q_input)
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
        return weight * (diff**2)

    # @typing_extensions.override
    @functools.partial(jax.jit, static_argnames=["self"])
    def training_v_step(
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
        del kwargs, rngs

        v_params, q_params, p_params = params

        if batch.state is None or batch.action is None:
            raise ValueError(
                "State and Action must not be None for IQL updates."
            )
        q_input = jnp.concatenate([batch.state, batch.action], axis=-1)

        def _value_loss_fn(value_params: jaxtyping.PyTree) -> jax.Array:
            value_output = self._value_network.apply(value_params, batch.state)
            value_output = typing.cast(jax.Array, value_output)

            # target params have same structure as q_params, which is a tuple of
            # (value_params, q_params, policy_params)
            q1_target = self._q_network.apply(target_params[0], q_input)
            q2_target = self._q_network.apply(target_params[1], q_input)
            q1_target = typing.cast(jax.Array, q1_target)
            q2_target = typing.cast(jax.Array, q2_target)

            q_target_min = jnp.minimum(q1_target, q2_target)

            # Compute value loss based on expectile regression
            # NOTE: Optax has no exceptile loss, so we manually define one.
            # NOTE: how we compute the expectation over data samples?
            value_loss = self._expectile_loss(value_output, q_target_min)

            return jnp.mean(value_loss)

        v_loss, v_grads = jax.value_and_grad(_value_loss_fn)(v_params)
        new_v_train_state = train_state.apply_gradients(grads=v_grads)

        outputs = _model.StepOutputs(
            scalars={
                "value_loss": v_loss.mean(),
            }
        )

        return new_v_train_state, outputs

    @functools.partial(jax.jit, static_argnames=["self"])
    def training_q_step(
        self,
        *,
        params: typing.Any,
        batch: structure.StepTuple,
        train_state: _train_state.TrainState,
        target_params: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Any:
        del kwargs, rngs

        value_params, q_params, policy_params = params

        if batch.state is None or batch.action is None:
            raise ValueError(
                "State and Action must not be None for IQL updates."
            )
        q_input = jnp.concatenate([batch.state, batch.action], axis=-1)
        batch_dims = q_input.shape[:-1]

        def _q_loss_fn(q_params: jaxtyping.PyTree) -> jax.Array:
            q1_output = self._q_network.apply(q_params[0], q_input)
            q2_output = self._q_network.apply(q_params[1], q_input)
            q1_output = typing.cast(jax.Array, q1_output).squeeze(-1)
            q2_output = typing.cast(jax.Array, q2_output).squeeze(-1)

            next_value_output = self._value_network.apply(
                value_params,
                batch.next_state,
            )
            next_value_output = typing.cast(jax.Array, next_value_output)
            next_value_output = next_value_output.squeeze(-1)

            # Compute Q-loss based on TD error
            # NOTE: `batch.reward` and `batch.done` have a shape of `(B,)`,
            # while `next_value_output` has a shape of `(B, 1)`. This can cause
            # broadcasing issue. We need to ensure shape compatibility here.
            chex.assert_equal_shape(
                [batch.reward, batch.done, next_value_output]
            )
            target_q = (
                batch.reward
                + self._gamma
                * (1 - jnp.asarray(batch.done))
                * next_value_output
            )
            chex.assert_shape(target_q, batch_dims)

            chex.assert_equal_shape([q1_output, q2_output, target_q])
            q_loss = jnp.add(
                jnp.mean((q1_output - target_q) ** 2),
                jnp.mean((q2_output - target_q) ** 2),
            )

            return q_loss

        q_loss, q_grads = jax.value_and_grad(_q_loss_fn)(q_params)
        new_q_train_state = train_state.apply_gradients(grads=q_grads)

        outputs = _model.StepOutputs(
            scalars={
                "q_loss": q_loss.mean(),
            }
        )

        return new_q_train_state, outputs

    @functools.partial(jax.jit, static_argnames=["self"])
    def training_p_step(
        self,
        *,
        params: typing.Any,
        batch: structure.StepTuple,
        train_state: _train_state.TrainState,
        target_params: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Any:
        del kwargs, rngs

        value_params, q_params, policy_params = params

        if batch.state is None or batch.action is None:
            raise ValueError(
                "State and Action must not be None for IQL updates."
            )
        q_input = jnp.concatenate([batch.state, batch.action], axis=-1)

        def _policy_loss_fn(policy_params: jaxtyping.PyTree) -> jax.Array:
            # For discrete action space the output is the categorical logits
            # distribution, and for continuous action space the output is the
            # mean of a Gaussian distribution.
            mean, log_std = self._policy_network.apply(
                policy_params,
                batch.state,
            )
            mean = typing.cast(jax.Array, mean)
            log_std = typing.cast(jax.Array, log_std)

            std = jnp.exp(log_std)
            log_prob = -0.5 * (
                ((batch.action - mean) / std) ** 2
                + 2 * log_std
                + jnp.log(2 * jnp.pi)
            )
            log_prob = jnp.sum(log_prob, axis=-1)

            q1_target = self._q_network.apply(target_params[0], q_input)
            q2_target = self._q_network.apply(target_params[1], q_input)
            q1_target = typing.cast(jax.Array, q1_target)
            q2_target = typing.cast(jax.Array, q2_target)

            q_target = jnp.minimum(q1_target, q2_target)

            value_output = self._value_network.apply(value_params, batch.state)
            value_output = typing.cast(jax.Array, value_output)

            advantage = q_target - value_output
            weights = jnp.exp(self._beta * advantage)
            # clip weights to avoid instability
            weights = jnp.clip(weights.squeeze(-1), a_max=100.0)

            # NOTE: `log_prob` has a shape of `(B,)` while `weights` has a
            # shape of `(B, 1)`. This can cause broadcasing issue.
            chex.assert_equal_shape([log_prob, weights])
            policy_loss = -jnp.mean(weights * log_prob)

            return policy_loss

        p_loss, policy_grads = jax.value_and_grad(_policy_loss_fn)(
            policy_params
        )
        new_p_train_state = train_state.apply_gradients(grads=policy_grads)

        outputs = _model.StepOutputs(
            scalars={
                "policy_loss": p_loss.mean(),
            }
        )

        return new_p_train_state, outputs
