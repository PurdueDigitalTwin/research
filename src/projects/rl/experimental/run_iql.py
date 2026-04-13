# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment: mujoco/halfcheetah/medium-v0 from minari
# Reference: https://arxiv.org/abs/2110.06169
################################################
# NOTE: Although it's offline RL, we still need an environment to evaluate
# the performance of the agent periodically during training.

import copy
import os
import typing

from absl import app
from absl import flags
import gymnasium as gym
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import minari
import numpy as np
import optax

from src.core import train_state as _train_state
from src.projects.rl import structure as _struct
from src.projects.rl.experimental import iql
from src.utilities import logging

# Running flags
flags.DEFINE_string(
    name="dataset_name",
    default="minari/halfcheetah/medium-v0",
    required=False,
    help="Name of the Minari dataset to load.",
)
flags.DEFINE_integer(
    name="num_episodes",
    default=5000,
    required=False,
    help="Total number of episodes for training.",
)
flags.DEFINE_integer(
    name="buffer_capacity",
    default=1_000_000,  # total steps in (minari show mujoco/halfcheetah/medium-v0)
    required=False,
    help="Maximum number of experiences to store in the replay buffer.",
)
flags.DEFINE_integer(
    name="batch_size",
    default=512,
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
        states.append(episode.observations[:-1])  # [s0, s1, s2, s3]
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        next_states.append(episode.observations[1:])  # [s1, s2, s3, s4]
        dones.append(episode.terminations)

    return _struct.StepTuple(
        state=jnp.array(np.concatenate(states, axis=0)),
        action=jnp.array(np.concatenate(actions, axis=0)),
        reward=jnp.array(np.concatenate(rewards, axis=0)),
        next_state=jnp.array(np.concatenate(next_states, axis=0)),
        done=jnp.array(np.concatenate(dones, axis=0)),
    )


def normalize_states(
    data: _struct.StepTuple,
) -> typing.Tuple[jax.Array, jax.Array, _struct.StepTuple]:
    r"""Normalizes the state observations in the dataset for better performance.

    Args:
        data: A tuple of (states, actions, rewards, next_states, dones).

    Returns:
        A tuple of (state_mean, state_std, normalized_data) where:
        - state_mean: The mean of the state observations.
        - state_std: The standard deviation of the state observations.
        - normalized_data: A StepTuple with normalized states and next_states.
    """
    assert data.state is not None, "State data is required for normalization."
    assert (
        data.next_state is not None
    ), "Next state data is required for normalization."

    state_mean = jnp.mean(data.state, axis=0)
    state_std = jnp.std(data.state, axis=0) + 1e-8

    normalized_state = (data.state - state_mean) / state_std
    normalized_next_state = (data.next_state - state_mean) / state_std

    normalized_data = _struct.StepTuple(
        state=normalized_state,
        action=data.action,
        reward=data.reward,
        next_state=normalized_next_state,
        done=data.done,
    )

    return state_mean, state_std, normalized_data


def sample_batch(
    rng: typing.Any,
    *,
    batch_size: int,
    num_samples: int,
    flat_data: _struct.StepTuple,
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
    assert (
        flat_data.next_state is not None
    ), "Next state data is required for \
        sampling."
    assert (
        flat_data.action is not None
    ), "Action data is required for sampling."
    assert (
        flat_data.reward is not None
    ), "Reward data is required for sampling."
    assert flat_data.done is not None, "Done data is required for sampling."

    indices = jax.random.choice(rng, num_samples, (batch_size,), replace=False)

    return _struct.StepTuple(
        state=flat_data.state[indices],
        action=flat_data.action[indices],
        reward=flat_data.reward[indices],
        next_state=flat_data.next_state[indices],
        done=flat_data.done[indices],
    )


def evaluate_agent(
    env: gym.Env,
    agent: iql.IQLModel,
    policy_params: typing.Any,
    state_mean: jax.Array,
    state_std: jax.Array,
    num_episodes: int = 5,
) -> float:
    r"""Evaluates the agent's performance in the environment.

    Args:
        env: The gym environment to evaluate in.
        agent: The IQL agent to evaluate.
        policy_params: The parameters of the agent's policy network.
        state_mean: The mean used for normalizing states during training.
        state_std: The standard deviation used for normalizing states during training.
        num_episodes: Number of episodes to run for evaluation.

    Returns:
        The average reward obtained over the evaluation episodes.
    """
    episode_rewards = []

    @jax.jit
    def get_action(params, s):
        # Forward pass the policy network to get the action for the given state.
        # The range of the action for tanh activation is [-1, 1], which matches
        # the action space of HalfCheetah-v4.
        mean, _ = agent._policy_network.apply(params, s)
        return mean

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            # Normalize the observation using the same mean and std as during
            # training.
            norm_obs = (jnp.array(obs) - state_mean) / state_std

            # Get action from the policy network and step in the environment.
            action = get_action(policy_params, norm_obs[None, :])
            action = np.array(action).squeeze()

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards))


def main(argv: typing.List[str]) -> None:
    del argv  # Unused.

    # NOTE: refer to minari documentation on the difference between simple,
    # medium, and expert datasets.
    dataset = minari.load_dataset(
        flags.FLAGS.dataset_name,
        download=True,
    )

    # Preprocessing: flatten the dataset and normalize the states.
    flat_data = flatten_data(dataset)
    state_mean, state_std, flat_data = normalize_states(flat_data)
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
    rngs = jax.random.PRNGKey(
        42
    )  # Use seed 42 for the training and evaluation.
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
    loss_log, reward_log = [], []

    # Main loop: Sample batches from the dataset and train the agent.
    num_samples = flat_data.state.shape[0]

    # Stage 1: Train the value and q networks using the sampled batches.
    logging.rank_zero_info("Starting Stage 1: Critic Training (V & Q)...")
    for episode in range(1, flags.FLAGS.num_episodes + 1):
        rngs, sample_rng = jax.random.split(rngs)
        batch = sample_batch(
            rng=sample_rng,
            batch_size=flags.FLAGS.batch_size,
            num_samples=num_samples,
            flat_data=flat_data,
        )

        # Update value and q networks using the sampled batch.
        # We may set different larning rates for different trainstate
        params = tuple(s.params for s in train_state)

        v_train_state, v_outputs = agent.training_v_step(
            params=params,
            batch=batch,
            train_state=train_state[0],
            target_params=target_params,
            rngs=rngs,
        )

        q_train_state, q_outputs = agent.training_q_step(
            params=params,
            batch=batch,
            train_state=train_state[1],
            target_params=target_params,
            rngs=rngs,
        )

        train_state = (v_train_state, q_train_state, train_state[2])

        # Soft update target params
        target_params = jax.tree_util.tree_map(
            lambda tp, p: tp * (1 - flags.FLAGS.alpha) + p * flags.FLAGS.alpha,
            target_params,
            train_state[1].params,
        )

        # log the loss for analysis
        loss_log.append(
            (v_outputs.scalars["value_loss"], q_outputs.scalars["q_loss"])
        )

        # logging every 10 episodes for better visibility
        if episode % 10 == 0:
            logging.rank_zero_info(
                "Episode %d: Value Loss = %.4f, Q Loss = %.4f",
                episode,
                v_outputs.scalars["value_loss"],
                q_outputs.scalars["q_loss"],
            )

    # Stage 2: Extract policy network using the sampled batches and the trained
    # value and q networks.
    logging.rank_zero_info("Starting Stage 2: Policy Extraction...")
    for episode in range(1, flags.FLAGS.num_episodes + 1):
        rngs, sample_rng = jax.random.split(rngs)
        batch = sample_batch(
            rng=sample_rng,
            batch_size=flags.FLAGS.batch_size,
            num_samples=num_samples,
            flat_data=flat_data,
        )

        # Update policy network using the sampled batch.
        params = tuple(s.params for s in train_state)

        p_train_state, p_outputs = agent.training_p_step(
            params=params,
            batch=batch,
            train_state=train_state[2],
            target_params=target_params,
            rngs=rngs,
        )

        train_state = (train_state[0], train_state[1], p_train_state)

        # log the loss for analysis
        loss_log.append(p_outputs.scalars["policy_loss"])

        # logging every 10 episodes for better visibility
        if episode % 10 == 0:
            logging.rank_zero_info(
                "Episode %d: Policy Loss = %.4f",
                episode,
                p_outputs.scalars["policy_loss"],
            )

        # Periodically evaluate the agent's performance in the environment.
        if episode % flags.FLAGS.eval_every_n_episodes == 0:
            avg_reward = evaluate_agent(
                env=env,
                agent=agent,
                policy_params=train_state[2].params,
                state_mean=state_mean,
                state_std=state_std,
                num_episodes=5,
            )

            reward_log.append(avg_reward)
            logging.rank_zero_info(
                "Episode %d: Average Reward = %.4f",
                episode,
                avg_reward,
            )

    # Plot the training loss and reward curves for analysis.
    # Plot four figures in 2*2
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("IQL Training Curves", fontsize=16)

    # Plot the value loss curve
    v_loss = [loss[0] for loss in loss_log if isinstance(loss, tuple)]
    axs[0, 0].plot(v_loss, color="blue", label="Value Loss")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Value Loss")

    # Plot the q loss curve
    q_loss = [loss[1] for loss in loss_log if isinstance(loss, tuple)]
    axs[0, 1].plot(q_loss, color="orange", label="Q Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Q Loss")

    # Plot the policy loss curve
    p_loss = [loss for loss in loss_log if not isinstance(loss, tuple)]
    axs[1, 0].plot(p_loss, color="green", label="Policy Loss")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Policy Loss")

    # Plot the reward curve
    axs[1, 1].plot(reward_log, color="red", label="Average Reward")
    axs[1, 1].set_xlabel("Episode (x10)")
    axs[1, 1].set_ylabel("Average Reward")

    # Save the figure to the working directory
    fig_path = os.path.join(flags.FLAGS.work_dir, "iql_training_curves.png")
    plt.savefig(fig_path)
    plt.close()
    logging.rank_zero_info("Saved training curves to %s", fig_path)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
