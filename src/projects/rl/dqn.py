# file created on Jan. 27, 2026
# Yaguang: used for rl learning (DQN)
################################################
# environment: gym CartPole-v1
# framework: jax + flax linen
# Modifications to original algorithm:
# 1. Use Huber loss instead of MSE loss for more stable training.
################################################
# qustion: why not recommend using nnx?

import collections
import copy
import functools
import os
import random
import typing

from absl import app
from absl import flags
import flax
from flax import linen as nn
from flax import serialization
import gymnasium as gym
import jax
from jax import lax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging

# Running flags
flags.DEFINE_integer(
    name="batch_size",
    default=32,
    required=False,
    help="Batch size for training and evaluation",
)
flags.DEFINE_integer(
    name="num_episodes",
    default=500,
    required=False,
    help="Total number of episodes for training.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Working directory",
)


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
class MlpPolicy(nn.Module):  # NOTE: why no init for this class?
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
    """

    features: int
    out_features: int
    num_layers: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact  # decorator indicates that this is a compact module. NOTE: what this means?
    def __call__(self, state: jax.Array) -> jax.Array:
        r"""Forward pass the policy network `\pi(a|s;\theta)`

        Args:
            state (jax.Array): Input state array of shape `(*, D)`

        Returns:
            Action index of shape `(*, 1)`.
        """
        out = state.astype(self.dtype)
        for i in range(self.num_layers):
            fc = nn.Dense(
                features=(
                    self.features
                    if i != self.num_layers - 1
                    else self.out_features
                ),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",  # fan_avg means average of fan_in and fan_out, fan_in means input dim, fan_out means output dim
                    distribution="uniform",  # uniform means uniform distribution
                ),
                use_bias=True,  # use bias term
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"fc_{i+1:d}",
            )
            out = fc(out)
            if i != self.num_layers - 1:
                out = jax.nn.relu(
                    out
                )  # apply relu activation function for hidden layers

        return out


# Define the DQNModel class by extending the base Model class
class DQNModel(_model.Model):
    r"""Deep Q-learning model."""

    def __init__(self, action_space_dim: int, gamma: float) -> None:
        r"""Instantiates a DQN model.

        Args:
            action_space_dim (int): Dimension of the action space.
            gamma (float): Discount factor for future rewards.

        Returns:
            None.
        """
        self.action_space_dim = action_space_dim
        self.gamma = gamma
        self.module = MlpPolicy(
            features=128,
            out_features=action_space_dim,
            num_layers=3,
        )

    def network(self) -> nn.Module:
        r"""NOTE: useless function here.
        Returns:
            nn.Module: The neural network module.
        """
        pass

    def init(
        self, rngs: jax.random.PRNGKey, state: jax.Array
    ) -> typing.Tuple[
        jax.Array, jax.Array
    ]:  # NOTE: why there are both __init__ and init?
        r"""Initializes Q network and target Q network parameters.

        Args:
            rngs (jax.random.PRNGKey): Random number generator key.
            state (jax.Array): Example state array for initialization.
        Returns:
            A tuple of (q_params, target_params).
        """
        # NOTE: can I change this to self.q_params?
        q_params = self.module.init(
            rngs=rngs,
            state=state,
        )
        target_params = copy.deepcopy(q_params)

        return q_params, target_params

    def forward(
        self, q_params: jax.Array, state: jax.Array
    ) -> jax.Array:  # NOTE: should it be typing.Tuple?
        r"""Compute Q-values of ALL possible actions for the given state. For cartpole, it will be
        [q_value(action=left), q_value(action=right)]

        Args:
            state (jax.Array): Input state array.

        Returns:
            jax.Array: Q-values for the given state.
        """
        return self.module.apply(q_params, state)

    def compute_loss(
        self,
        *,
        state: jax.Array,
        action: jax.Array,
        next_state: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        # original args
        params: typing.Any,
        target_params: typing.Any,
        rngs: typing.Any,  # NOTE: we aren't using these three args.
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[
        jax.Array, _model.StepOutputs
    ]:  # NOTE: why we need another stepoutputs here?
        r"""Computes the DQN loss using the Bellman equation.

        Args:
            state (jax.Array): Current state array.
            action (jax.Array): Action taken array.
            next_state (jax.Array): Next state based on the action taken array.
            reward (jax.Array): Reward received array.
            done (jax.Array): Done flag array. Done == 1 if episode ends after this step, otherwise 0.
            params (Any): Model parameters.
            rngs (Any): Random number generators.
            deterministic (bool): Whether to use deterministic actions.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (loss, StepOutputs).
        """
        # Compute Q-values for current states
        q_values = self.forward(params, state)

        # Select Q-values for the action actually taken
        one_hot_action = jax.nn.one_hot(action, num_classes=q_values.shape[-1])
        q_values = jnp.sum(q_values * lax.stop_gradient(one_hot_action), -1)

        # Compute Q-values for next states using target network
        next_q_values = self.forward(
            target_params, next_state
        )  # NOTE: initially I use params here and the loss isn't decreasing.

        # Simply max over action dimension, which is the drawback of DQN
        max_next_q = jnp.max(next_q_values, axis=1)

        # Compute TD-target using the Bellman equation
        TD_target = reward + lax.stop_gradient(
            self.gamma * max_next_q * (1.0 - done)
        )

        # Compute loss as mean squared error
        # loss = jnp.mean((TD_target - q_action) ** 2)
        # NOTE: use Huber loss for more faster and stable training
        loss = jnp.mean(optax.squared_error(q_values, TD_target))
        step_outputs = _model.StepOutputs(scalars={"loss": loss})

        return loss, step_outputs


# Create a replay buffer class to store experiences
class ReplayBuffer:
    def __init__(self, capacity=10000) -> None:
        r"""Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.

        Returns:
            None.
        """
        self.buffer = collections.deque(
            maxlen=capacity
        )  # deque is a double-ended queue.

    def add(self, s, a, r, s_next, d) -> None:
        r"""Adds a new experience to the replay buffer. The buffer stores tuples of (s, a, r,
        s_next, d). Each element of the queue is a tuple.

        Args:
            s: state
            a: action
            r: reward
            s_next: next state
            d: done flag

        Returns:
            None.
        """
        if d is None:
            d = False
        self.buffer.append((s, a, r, s_next, d))

    def sample(
        self, batch_size
    ) -> typing.Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(
            self.buffer, batch_size
        )  # randomly sample batch_size number of experiences from the buffer
        s, a, r, s_next, d = zip(*batch)  # unzip the batch into separate lists

        return (
            jnp.array(s),
            jnp.array(a),
            jnp.array(r),
            jnp.array(s_next),
            jnp.array(d, dtype=jnp.bool_),
        )


# the training step function
def train_step(
    rngs: jax.Array,
    state: _train_state.TrainState,
    agent: _model.Model,
    target_params: jax.Array,
    batch: typing.Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
    r"""Performs a single training step.

    Args:
        params (jax.Array): Current agent parameters.
        target_params (jax.Array): Target network parameters.
        batch (Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]): Batch of experiences.
        rngs (jax.random.PRNGKey): Random number generator key.

    Returns:
        A tuple of (updated_params, updated_opt_state, loss).
    """
    states, actions, rewards, next_states, dones = batch
    local_rng = jax.random.fold_in(rngs, state.step)

    # Compute loss and gradients
    def loss_fn(params):
        return agent.compute_loss(
            state=states,
            action=actions,
            next_state=next_states,
            reward=rewards,
            done=dones,
            params=params,
            target_params=target_params,
            rngs=local_rng,
            deterministic=False,
        )

    # Get gradients
    grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grads_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


def main(_: typing.List[str]) -> int:
    del _  # NOTE: unused arguments

    # Example usage of the DQNModel class:
    # Define Hyperparameters
    learning_rate = 1e-3
    buffer_capacity = 30000
    epsilon = 0.05  # for epsilon-greedy policy
    target_update_freq = 1000  # target network update frequency (in steps)
    gamma = 0.99  # discount factor

    # Random keys for JAX
    rngs = jax.random.PRNGKey(0)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Create Gym cartpole environment
    env = gym.make("CartPole-v1")

    # Create an instance of the DQNModel
    # Create an instance of the MlpPolicy (inside the DQNModel). For jax nn, we need to initialize the parameters first.
    agent = DQNModel(
        action_space_dim=env.action_space.n,
        gamma=gamma,
    )

    # We may need to print the model summary for analysis. Note that each layer has its own kernel matrix and bias vector (if use_bias=True)

    # Initialize agent parameters
    rngs, init_rng = jax.random.split(rngs, num=2)
    (
        q_params,
        target_params,
    ) = agent.init(  # NOTE: why we need rngs and state here?
        rngs=init_rng,
        state=jnp.zeros((1, env.observation_space.shape[0])),
    )
    print(
        nn.summary.tabulate(agent.module, init_rng)(
            jnp.zeros((1, env.observation_space.shape[0]))
        )
    )

    # Create train state instance
    train_state = _train_state.TrainState.create(
        params=q_params,
        tx=optax.adam(learning_rate=learning_rate),
    )

    # log loss for analysis
    loss_log = []

    # Create two individual rngs for training and sampling
    rngs, train_rng, buffer_rng, sample_rng = jax.random.split(rngs, num=4)
    p_train_step = functools.partial(train_step, rngs=train_rng)
    p_train_step = jax.jit(p_train_step, static_argnames=["agent"])

    # Populates the replay buffer
    logging.rank_zero_info("Populating buffer...")
    state, _ = env.reset()
    for step in range(buffer_capacity):
        sample_step_rng = jax.random.fold_in(buffer_rng, step)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = (
            terminated or truncated
        )  # NOTE: why need both terminated and truncated?

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, _ = env.reset()
    logging.rank_zero_info("Populating buffer... DONE!")

    # The main training loop
    logging.rank_zero_info("Training...")
    for episode in range(flags.FLAGS.num_episodes):
        state, _ = env.reset()
        done = False
        episode_losses = []

        # done marks the end of each episode
        while not done:
            sample_step_rng = jax.random.fold_in(sample_rng, train_state.step)

            # Epsilon-greedy action selection
            if jax.random.uniform(key=sample_step_rng) < epsilon:
                # Exploration: random action
                action = env.action_space.sample()
            else:
                # Exploitation: select best action based on Q-values
                q_values = agent.forward(q_params, jnp.array(state[None, :]))
                action = int(jnp.argmax(q_values))

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = (
                terminated or truncated
            )  # NOTE: why need both terminated and truncated?

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Sample a batch of experiences from the replay buffer and train the agent
            if len(replay_buffer.buffer) >= buffer_capacity:
                batch = replay_buffer.sample(flags.FLAGS.batch_size)

                # Run the JIT-compiled update step
                train_state, loss = p_train_step(
                    state=train_state,
                    agent=agent,
                    target_params=target_params,
                    batch=batch,
                )
                loss_log.append(loss)
                episode_losses.append(loss)

                # Update target network periodically
                if train_state.step % target_update_freq == 0:
                    target_params = copy.deepcopy(train_state.params)

        logging.rank_zero_info(
            "Episode %d | Episode Loss: %.6f",
            episode + 1,
            sum(episode_losses) / len(episode_losses),
        )

    # When the trainning is done, save the serialized model parameters to a file
    with open("dqn_model_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(train_state.params))

    # Close the environment
    env.close()

    # Plot the loss curve
    plt.plot(loss_log)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss Curve")
    plt.savefig(os.path.join(flags.FLAGS.work_dir, "dqn_loss_curve.png"))
    # plt.show()
    print("Training loss curve saved as dqn_loss_curve.png")

    ##################################################################
    # Test the trained agent: initialize a new environment
    env = gym.make("CartPole-v1")
    dummy_state = jnp.zeros((1, env.observation_space.shape[0]))
    dummy_params, _ = agent.init(rngs, dummy_state)

    # Load the model parameters and test the trained agent
    with open("dqn_model_params.msgpack", "rb") as f:
        loaded_params = serialization.from_bytes(dummy_params, f.read())

    # Run the testing loop
    num_test_episodes = 5

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Forward pass (Note: pure exploitation)
            # Add batch dimension [None, :] because the model expects (batch, features)
            q_values = agent.forward(loaded_params, jnp.array(state[None, :]))

            # Select the best action
            action = int(jnp.argmax(q_values))

            # Step
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Test Episode {episode + 1}: Reward = {total_reward}")

    env.close()

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
