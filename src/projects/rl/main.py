# file created on Jan. 27, 2026
# The purpose of this file is for rl learning (DQN)
################################################
# framework: jax + flax linen
# environment: gym CartPole-v1
# reference: https://arxiv.org/pdf/1312.5602
################################################
# NOTE: the performance is not good (and the loss is not decreasing). The 
# testing rewards are approxmately 80-200 (max reward is 500).
# NOTE: will DQN encounters overfitting problem if trainning for too long?
# NOTE: usually 2x batch size we do 2x learning rate, and buffer capacity is 
# 10x of the batch size.


import copy
import functools
import os
import typing

from absl import app
from absl import flags
from flax import serialization
import gymnasium as gym
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging
from src.projects.rl.dqn import DQNModel
from src.projects.rl.StepTuple import StepTuple
from src.projects.rl.ReplayBuffer import ReplayBuffer


# Running flags
flags.DEFINE_integer(
    name="batch_size",
    default=512,
    required=False,
    help="Batch size for training and evaluation",
)
flags.DEFINE_integer(
    name="num_episodes",
    default=10_000,
    required=False,
    help="Total number of episodes for training.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Working directory",
)


# the training step function
def train_step(
    rngs: jax.Array,
    state: _train_state.TrainState,
    agent: _model.Model,
    target_params: jax.Array,
    batch: StepTuple,
) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
    r"""Performs a single training step.

    Args:
        params (jax.Array): Current agent parameters.
        target_params (jax.Array): Target network parameters.
        batch (StepTuple): Batch of experiences.
        rngs (jax.random.PRNGKey): Random number generator key.

    Returns:
        A tuple of updated training state and loss value.
    """
    local_rng = jax.random.fold_in(rngs, state.step)

    # Compute loss and gradients
    def loss_fn(params):
        return agent.compute_loss(
            batch=batch,
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
    # use annealing epsilon for better performance.
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_episodes = 5000
    
    target_update_freq = 1000  # target network update frequency (in steps)
    gamma = 0.99  # discount factor

    # Random keys for JAX
    rngs = jax.random.PRNGKey(0)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Create Gym cartpole environment
    env = gym.make("CartPole-v1")

    # Create an instance of the MlpPolicy (inside the DQNModel).
    # For `jax.nn`, we need to initialize the parameters first.
    agent = DQNModel(action_space_dim=env.action_space.n, gamma=gamma)

    # Initialize agent parameters
    rngs, init_rng = jax.random.split(rngs, num=2)
    q_params = agent.init(
        batch=StepTuple(
            state=jnp.zeros((1, env.observation_space.shape[0])),
        ),
        rngs=init_rng,
    )
    target_params = copy.deepcopy(q_params)

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
    p_eval_step = jax.jit(agent.forward)

    # Populates the replay buffer
    logging.rank_zero_info("Populating buffer...")
    state, _ = env.reset()
    for step in range(buffer_capacity):
        sample_step_rng = jax.random.fold_in(buffer_rng, step)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = (
            terminated or truncated
        )

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
            # Epsilon-greedy action selection
            progress = min(1.0, episode / epsilon_decay_episodes)
            epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
            sample_step_rng = jax.random.fold_in(sample_rng, train_state.step)

            if jax.random.uniform(key=sample_step_rng) < epsilon:
                # Exploration: random action
                action = env.action_space.sample()
            else:
                # Exploitation: select best action based on Q-values
                q_values = p_eval_step(
                    batch=StepTuple(state=jnp.array(state[None, :])),
                    params=train_state.params,
                ).output
                action = jnp.argmax(q_values, axis=-1).item()

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = (
                terminated or truncated
            )

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Sample a batch of experiences from the replay buffer and train 
            # the agent
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
                    logging.rank_zero_info("Target network synced!")

        logging.rank_zero_info(
            "Episode %d | Episode Loss: %.6f",
            episode + 1,
            sum(episode_losses) / len(episode_losses),
        )

    # When the trainning is done, save the serialized model parameters to a file
    with open("dqn_model_params.msgpack", "wb") as f:
        f.write(serialization.msgpack_serialize(train_state.params))

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
    # Test the trained agent
    ##################################################################
    env = gym.make("CartPole-v1")

    # Load the model parameters and test the trained agent
    with open("dqn_model_params.msgpack", "rb") as f:
        loaded_params = serialization.msgpack_restore(f.read())

    # Run the testing loop
    num_test_episodes = 5

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Forward pass (Note: pure exploitation)
            # Add batch dimension [None, :] because the model expects (batch, 
            # features)
            q_values = p_eval_step(
                batch=StepTuple(state=jnp.array(state[None, :])),
                params=loaded_params,
            ).output

            # Select the best action
            action = jnp.argmax(q_values, axis=-1).item()

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
