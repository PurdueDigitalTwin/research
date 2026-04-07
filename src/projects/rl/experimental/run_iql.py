# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment
# Reference: https://arxiv.org/abs/2110.06169
################################################
# NOTE: directly read data from .hdf5 is slow. We can first load the data 
# into memory and then create a replay buffer.
# NOTE: Alghough it's offline RL, we still need an environment to evaluate
# the performance of the agent periodically during training.


import copy
import functools
import os
import typing

from absl import app
from absl import flags
from flax import jax_utils
from flax import serialization
import gymnasium as gym
import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import minari

from src.core import train_state as _train_state
from src.projects.rl.experimental import iql
from src.projects.rl import replay_buffer as _buffer
from src.projects.rl import structure as _struct
from src.utilities import logging
from src.utilities import training

# Running flags
flags.DEFINE_integer(
    name="num_episodes",
    default=10_000,
    required=False,
    help="Total number of episodes for training.",
)
flags.DEFINE_integer(
    name="buffer_capacity",
    default=1_000_000, # total steps in (minari show mujoco/halfcheetah/medium-v0)
    required=False,
    help="Maximum number of experiences to store in the replay buffer.",
)
flags.DEFINE_integer(
    name="eval_every_n_episodes",
    default=10,
    required=False,
    help="Evaluation frequency (in episodes) during training.",
)
flags.DEFINE_float(
    name="gamma",
    default=0.99,
    required=False,
    help="Discount factor for future rewards.",
)
flags.DEFINE_float(
    name="tau",
    default=0.7,
    required=False,
    help="Expectile parameter for value learning in IQL.",
)
flags.DEFINE_float(
    name="beta",
    default=3.0,
    required=False,
    help="Inverse temperature for policy learning in IQL.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=3e-4,
    required=False,
    help="Learning rate for the optimizer.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Working directory",
)


def flatten_data(dataset: minari.MinariDataset) -> _struct.StepTuple:
    r"""Flattens a Minari dataset into a tuple of transitions.

    Args:
        dataset (minari.Dataset): A Minari dataset containing episodes of data.

    Returns:
        A StepTuple containing flattened transitions.
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for episode in dataset.iterate_episodes():
        states.append(episode.observations[:-1]) # [s0, s1, s2, s3]
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        next_states.append(episode.observations[1:]) # [s1, s2, s3, s4]
        dones.append(episode.terminations)

    return _struct.StepTuple(
        state=jnp.array(np.concatenate(states, axis=0)),
        action=jnp.array(np.concatenate(actions, axis=0)),
        reward=jnp.array(np.concatenate(rewards, axis=0)),
        next_state=jnp.array(np.concatenate(next_states, axis=0)),
        done=jnp.array(np.concatenate(dones, axis=0)),
    )


def normalize_states(data: _struct.StepTuple) -> _struct.StepTuple:
    r"""Normalizes the state observations in the dataset for better performance.

    Args:
        data: A tuple of (states, actions, rewards, next_states, dones).

    Returns:
        A StepTuple with normalized states and next_states.
    """
    assert data.state is not None, "State data is required for normalization."
    assert data.next_state is not None, \
        "Next state data is required for normalization."

    state_mean = jnp.mean(data.state, axis=0)
    state_std = jnp.std(data.state, axis=0) + 1e-8

    normalized_state = (data.state - state_mean) / state_std
    normalized_next_state = (data.next_state - state_mean) / state_std

    return _struct.StepTuple(
        state=normalized_state,
        action=data.action,
        reward=data.reward,
        next_state=normalized_next_state,
        done=data.done,
    )


def main(argv: typing.List[str]) -> None:
    del argv  # Unused.

    # NOTE: refer to minari documentation on the difference between simple,
    # medium, and expert datasets.
    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")
    flat_data = flatten_data(dataset)

    # Normalize the state
    flat_data = normalize_states(flat_data)

    # Create a gym environment for evaluation.
    env = gym.make("HalfCheetah-v4")
    state_size = env.observation_space.shape
    action_size = env.action_space.shape

    assert state_size is not None, "State space must be a Box space."
    assert action_size is not None, "Action space must be a Box space."

    logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_size,
        action_size,
    )

    # Create a replay buffer and load the dataset into the buffer.
    replay_buffer = _buffer.ReplayBuffer(
        capacity=flags.FLAGS.buffer_capacity,
        state_size=state_size,
        action_size=action_size,
    )

    # Create an IQL agent using IQL model and policy.
    agent = iql.IQLModel(
        action_space_dim=action_size[0],
        gamma=flags.FLAGS.gamma,
        tau=flags.FLAGS.tau,
        beta=flags.FLAGS.beta,
    )

    # initialize agent's parameters using a batch of data from the replay buffer
    # NOTE: Minari dataset is collected in episodes, so we need to flatten the 
    # dataset into transitions before loading it into the replay buffer.
    rngs = jax.random.PRNGKey(42) # Use seed 42 for the training and evaluation.
    rngs, init_rng = jax.random.split(rngs)
    params = agent.init(
        batch=_struct.StepTuple(state=jnp.zeros((1, *state_size))),
        rngs=init_rng,
    )

    # Create a trainstate instance for the agent.
    optimizer = optax.adam(learning_rate=flags.FLAGS.learning_rate)
    train_state = _train_state.TrainState.create(
        params=params,
        tx=optimizer,
    )
    target_params = copy.deepcopy(train_state.params)

    # log loss and reward for analysis
    loss_log , reward_log = [], []




if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
