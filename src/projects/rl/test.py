# file created on Jan. 27, 2026
# Yaguang: used for rl learning (DQN)
################################################
# environment: gym CartPole-v1
# framework: jax + flax linen
################################################
# qustion: why not recommend using nnx?

import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import copy
import optax
import typing
import random
import collections
import gymnasium as gym

import jax
from flax import linen as nn
from jax import numpy as jnp
from src.core import model


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
class MlpPolicy(nn.Module): # question: why no init for this class?
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
    """

    features: int
    out_features: int
    num_layers: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact # decorator indicates that this is a compact module. question: what this means?
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
                features=(self.features if i != self.num_layers - 1 else self.out_features),
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
                out = jax.nn.relu(out)  # apply relu activation function for hidden layers

        return out
    

# Define the DQNModel class by extending the base Model class
class DQNModel(model.Model):
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

    def network(self) -> nn.Module:
        r"""Defines the neural network module.
        
        Args: 
            module (nn.Module): The neural network module.

        Returns:
            The neural network module.
        """
        self.module = MlpPolicy(
            features=128,
            out_features=self.action_space_dim,
            num_layers=3,
            dtype=jnp.float32, # question: float32 or float64?
            param_dtype=jnp.float32,
        )

        return self.module

    def init(self, rngs: jax.random.PRNGKey, state: jax.Array) -> typing.Tuple[jax.Array, jax.Array]: # question: why there are both __init__ and init?
        r"""Initializes Q network and target Q network parameters.

        Args:
            rngs (jax.random.PRNGKey): Random number generator key.
            state (jax.Array): Example state array for initialization.
        Returns:
            A tuple of (q_params, target_params).
        """
        # question: can I change this to self.q_params?
        q_params = self.module.init(
            rngs=rngs,
            state=state,
        )
        target_params = copy.deepcopy(q_params)

        return q_params, target_params

    def forward(self, q_params: jax.Array, state: jax.Array) -> jax.Array:  # question: should it be typing.Tuple?
        r"""compute Q-values of ALL possible actions for the given state.
            For cartpole, it will be [q_value(action=left), q_value(action=right)]

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
        rngs: typing.Any,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[jax.Array, model.StepOutputs]: # question: what is StepOutputs here?
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
        q_action = jax.vmap(lambda q, a: q[a])(q_values, action)

        # Compute Q-values for next states using target network
        next_q_values = self.forward(params, next_state)

        # Simply max over action dimension, which is the drawback of DQN
        max_next_q = jnp.max(next_q_values, axis=1)

        # Compute TD-target using the Bellman equation
        TD_target = reward + self.gamma * max_next_q * (1.0 - done)

        # Compute loss as mean squared error
        loss = jnp.mean((TD_target - q_action) ** 2)

        step_outputs = model.StepOutputs(
            scalars={"loss": loss},
        )

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
        self.buffer = collections.deque(maxlen=capacity) # deque is a double-ended queue.

    def add(self, s, a, r, s_next, d) -> None:
        r"""Adds a new experience to the replay buffer. The buffer stores tuples of (s, a, r, s_next, d).
            Each element of the queue is a tuple.
        
        Args:
            s: state
            a: action
            r: reward
            s_next: next state
            d: done flag
        
        Returns:
            None.
        """
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size) -> typing.Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size) # randomly sample batch_size number of experiences from the buffer
        s, a, r, s_next, d = zip(*batch) # unzip the batch into separate lists

        return (
            jnp.array(s),
            jnp.array(a),
            jnp.array(r),
            jnp.array(s_next),
            jnp.array(d, dtype=jnp.float32), # question: can I change this to boolen?
        )

# Example usage of the DQNModel class:
# Define Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_episodes = 5000
buffer_capacity = 8000
epsilon = 0.1  # for epsilon-greedy policy

# Create Gym cartpole environment
env = gym.make("CartPole-v1")

# Create an instance of the MlpPolicy. For jax nn, we need to initialize the parameters first.
# We may need to print the model summary for analysis. Note that each layer has its own kernel matrix and bias vector (if use_bias=True)
# network = MlpPolicy(features=128, out_features=2, num_layers=3)
# params = network.init(  # question: why there is rngs and state here?
#     rngs=jax.random.PRNGKey(0),  # rngs means random number generators
#     state=jnp.zeros((batch_size, env.observation_space.shape[0])),
# )
# print(
#     nn.summary.tabulate(network, jax.random.PRNGKey(0))(
#         jnp.zeros((batch_size, env.observation_space.shape[0]))
#     )
# )

# Random keys for JAX
rngs = jax.random.PRNGKey(0)

# Define optimizer
optimizer = optax.adam(learning_rate)

# Create replay buffer
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# Create an instance of the DQNModel
model = DQNModel(
    action_space_dim=env.action_space.n,
    gamma=0.99,
)

# Initialize model parameters
q_params, target_params = model.init(
    rngs=rngs,
    state=jnp.zeros((batch_size, env.observation_space.shape[0])), # question: initially there are no date in the buffer, can use batch_size here?
)

# The main training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    # done marks the end of each episode
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: random action
            action = env.action_space.sample()
        else:
            # Exploitation: select best action based on Q-values
            q_values = model.forward(q_params, jnp.array(state[None, :]))
            action = int(jnp.argmax(q_values))

        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # question: why need both terminated and truncated?
        episode_reward += reward

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state

        # Sample a batch of experiences from the replay buffer
        if len(replay_buffer.buffer) >= batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

            # Compute loss and gradients adn update model parameters
            
    print(f"Episode {episode + 1}, Reward: {episode_reward}")

# Close the environment
env.close()
