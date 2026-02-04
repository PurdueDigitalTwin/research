# file created on Jan. 27, 2026
# Yaguang: used for rl learning (DQN)
################################################
# environment: gym CartPole-v1
# framework: jax + flax linen
# Modifications to original algorithm:
# 1. Use Huber loss instead of MSE loss for more stable training.
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
import flax
from flax import linen as nn
from jax import numpy as jnp
from src.core import model

import matplotlib.pyplot as plt


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
class MlpPolicy(nn.Module): # NOTE: why no init for this class?
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
    """

    features: int
    out_features: int
    num_layers: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact # decorator indicates that this is a compact module. NOTE: what this means?
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
        self.module = MlpPolicy(
            features=128,
            out_features=action_space_dim,
            num_layers=3,
            dtype=jnp.float32, # NOTE: float32 or float64?
            param_dtype=jnp.float32,
        )

    def network(self) -> nn.Module:
        r"""NOTE: useless function here.
        Returns:
            nn.Module: The neural network module.
        """
        pass
        

    def init(self, rngs: jax.random.PRNGKey, state: jax.Array) -> typing.Tuple[jax.Array, jax.Array]: # NOTE: why there are both __init__ and init?
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

    def forward(self, q_params: jax.Array, state: jax.Array) -> jax.Array:  # NOTE: should it be typing.Tuple?
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
        target_params: typing.Any,
        rngs: typing.Any, # NOTE: we aren't using these three args.
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Tuple[jax.Array, model.StepOutputs]: # NOTE: why we need another stepoutputs here?
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
        next_q_values = self.forward(target_params, next_state) # NOTE: initially I use params here and the loss isn't decreasing.

        # Simply max over action dimension, which is the drawback of DQN
        max_next_q = jnp.max(next_q_values, axis=1)

        # Compute TD-target using the Bellman equation
        TD_target = reward + self.gamma * max_next_q * (1.0 - done)

        # Compute loss as mean squared error
        # loss = jnp.mean((TD_target - q_action) ** 2) 
        # NOTE: use Huber loss for more faster and stable training
        loss = jnp.mean(optax.huber_loss(q_action, TD_target, delta=1.0))

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
            jnp.array(d, dtype=jnp.float32), # NOTE: can I change this to boolen?
        )


# the training step function
@jax.jit
def train_step(
    params: jax.Array,
    target_params: jax.Array,
    opt_state: optax.OptState,
    batch: typing.Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    rngs: jax.random.PRNGKey,
) -> typing.Tuple[jax.Array, optax.OptState, model.StepOutputs]:
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
            rngs=rngs,
            deterministic=False,
        )
    
    # Get gradients
    grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grads_fn(params)

    # Update parameters using optimizer
    updates, updated_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    return updated_params, updated_opt_state, loss


# Example usage of the DQNModel class:
# Define Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_episodes = 1000
buffer_capacity = 30000
epsilon = 0.05  # for epsilon-greedy policy
target_update_freq = 1000  # target network update frequency (in steps)
gamma = 0.99  # discount factor

# Create Gym cartpole environment
env = gym.make("CartPole-v1")

# Random keys for JAX
rngs = jax.random.PRNGKey(0)

# Create replay buffer
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# Create an instance of the DQNModel
# Create an instance of the MlpPolicy (inside the DQNModel). For jax nn, we need to initialize the parameters first.
agent = DQNModel(
    action_space_dim=env.action_space.n,
    gamma=gamma,
)

# We may need to print the model summary for analysis. Note that each layer has its own kernel matrix and bias vector (if use_bias=True)
print(
    nn.summary.tabulate(agent.module, jax.random.PRNGKey(0))(
        jnp.zeros((batch_size, env.observation_space.shape[0]))
    )
)

# Initialize agent parameters
q_params, target_params = agent.init( # NOTE: why we need rngs and state here?
    rngs=rngs,
    state=jnp.zeros((batch_size, env.observation_space.shape[0])), # NOTE: initially there are no data in the buffer, can use batch_size here?
)

# Define optimizer
optimizer = optax.adam(learning_rate)

# Initialize optimizer state
opt_state = optimizer.init(q_params)

# log loss for analysis
loss_log = []

# the target network is updated every fixed number of steps
total_steps = 0

# The main training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    # done marks the end of each episode
    while not done:
        total_steps += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: random action
            action = env.action_space.sample()
        else:
            # Exploitation: select best action based on Q-values
            q_values = agent.forward(q_params, jnp.array(state[None, :]))
            action = int(jnp.argmax(q_values))

        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # NOTE: why need both terminated and truncated?
        episode_reward += reward

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        # Sample a batch of experiences from the replay buffer and train the agent
        if len(replay_buffer.buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)

            # Run the JIT-compiled update step
            q_params, opt_state, loss = train_step(q_params, target_params, opt_state, batch, rngs)
            # print(loss)
            loss_log.append(loss)

            # Update target network periodically
            if total_steps % target_update_freq == 0:
                target_params = copy.deepcopy(q_params)

    print(f"Episode {episode + 1}, Episode Reward: {episode_reward}")

# When the trainning is done, save the serialized model parameters to a file
with open("dqn_model_params.msgpack", "wb") as f:
    f.write(flax.serialization.to_bytes(q_params))

# Close the environment
env.close()

# Plot the loss curve
plt.plot(loss_log)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("DQN Training Loss Curve")
plt.savefig("dqn_loss_curve.png")
# plt.show()
print("Training loss curve saved as dqn_loss_curve.png")

##################################################################
# Test the trained agent: initialize a new environment
env = gym.make("CartPole-v1")
dummy_state = jnp.zeros((1, env.observation_space.shape[0]))
dummy_params, _ = agent.init(rngs, dummy_state)

# Load the model parameters and test the trained agent
with open("dqn_model_params.msgpack", "rb") as f:
    loaded_params = flax.serialization.from_bytes(dummy_params, f.read())

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

