# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment
# Reference: https://arxiv.org/abs/2110.06169
################################################
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

from src.core import model
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
    name="batch_size",
    default=256,
    required=False,
    help="Number of transitions to sample in each training batch.",
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
flags.DEFINE_float(
    name="alpha",
    default=0.005,
    required=False,
    help="Soft update coefficient for target network updates.",
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


def sample_batch(
        rng: typing.Any,
        *,
        batch_size: int,
        num_samples: int,
        flat_data: _struct.StepTuple
    ) -> _struct.StepTuple:
    r"""Samples a batch of transitions from the flattened dataset.

    Args:
        rng_key: A JAX random key for reproducibility.
        batch_size: Number of transitions to sample in the batch.
        num_samples: Total number of transitions in the flattened dataset.
        flat_data: A StepTuple containing the flattened dataset.

    Returns:
        A StepTuple containing the sampled batch of transitions.
    """
    assert flat_data.state is not None, "State data is required for sampling."
    assert flat_data.next_state is not None, "Next state data is required for \
        sampling."
    assert flat_data.action is not None, "Action data is required for sampling."
    assert flat_data.reward is not None, "Reward data is required for sampling."
    assert flat_data.done is not None, "Done data is required for sampling."

    indices = jax.random.choice(rng, num_samples, (batch_size,), replace=False)

    return _struct.StepTuple(
        state=flat_data.state[indices],
        action=flat_data.action[indices],
        reward=flat_data.reward[indices],
        next_state=flat_data.next_state[indices],
        done=flat_data.done[indices],
    )


def main(argv: typing.List[str]) -> None:
    del argv  # Unused.

    # NOTE: refer to minari documentation on the difference between simple,
    # medium, and expert datasets.
    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    # Preprocessing: flatten the dataset and normalize the states.
    flat_data = flatten_data(dataset)
    flat_data = normalize_states(flat_data)
    assert flat_data.state is not None, "State data is required for training."

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
    v_params, q_params, p_params = agent.init(
        batch=_struct.StepTuple(
            state=jnp.zeros((1, *state_size)),
            action=jnp.zeros((1, *action_size)),
            reward=jnp.zeros((1, 1)),
            next_state=jnp.zeros((1, *state_size)),
            done=jnp.zeros((1, 1)),
        ),
        rngs=init_rng,
    )

    # Create a trainstate instance for the agent.
    # optimizer = optax.adam(learning_rate=flags.FLAGS.learning_rate)
    
    # Create a train state for each network (value, q, policy) in the IQL model.
    v_state = _train_state.TrainState.create(
        # apply_fn=model.
        params=v_params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )
    q_state = _train_state.TrainState.create(
        # apply_fn=model._q_network.apply,
        params=q_params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )
    p_state = _train_state.TrainState.create(
        # apply_fn=model._policy_network.apply,
        params=p_params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )

    train_state = (v_state, q_state, p_state)

    target_params = copy.deepcopy(train_state[1].params)

    # log loss and reward for analysis
    loss_log , reward_log = [], []
 
    # Main loop: Sample batches from the dataset and train the agent.
    num_samples = flat_data.state.shape[0]

    for episode in range(1, flags.FLAGS.num_episodes + 1):
        rngs, sample_rng = jax.random.split(rngs)
        batch = sample_batch(
            rng=sample_rng,
            batch_size=flags.FLAGS.batch_size,
            num_samples=num_samples,
            flat_data=flat_data,
        )

        # We may set different larning rates for different trainstate
        v_train_state, step_outputs = agent.training_step(
            params=train_state[0].params,
            batch=batch,
            train_state=train_state[0],
            target_params=target_params,
            rngs=rngs,
        )

        q_train_state, step_outputs = agent.training_step(
            params=train_state[1].params,
            batch=batch,
            train_state=train_state[1],
            target_params=target_params,
            rngs=rngs,
        )

        p_train_state, step_outputs = agent.training_step(
            params=train_state[2].params,
            batch=batch,
            train_state=train_state[2],
            target_params=target_params,
            rngs=rngs,
        )

        # Soft update target params
        target_params = jax.tree_util.tree_map(
            lambda tp, p: tp * (1 - flags.FLAGS.alpha) + p * flags.FLAGS.alpha,
            target_params,
            train_state[1].params,
        )

        loss_log.append(step_outputs.output)
        logging.rank_zero_info(
            "Episode %d: Loss = %.4f",
            episode,
            step_outputs.output,
        )

        # Periodically evaluate the agent's performance in the environment.
        if episode % flags.FLAGS.eval_every_n_episodes == 0:
            pass
            # total_reward = training.evaluate_agent(
            #     env=env,
            #     agent=agent,
            #     params=train_state.params[2], # policy params
            #     num_episodes=5, # evaluate for 5 episodes and average the reward
            #     max_steps_per_episode=1000, # max steps per episode
            #     rngs=rngs,
            # )
            # reward_log.append(total_reward)
            # logging.rank_zero_info(
            #     "Episode %d: Average Reward = %.2f",
            #     episode,
            #     total_reward,
            # )


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
