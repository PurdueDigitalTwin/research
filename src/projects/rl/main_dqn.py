# file created on Jan. 27, 2026
# The purpose of this file is for rl learning (DQN)
################################################
# framework: jax + flax linen
# environment: gym CartPole-v1
# reference: https://arxiv.org/pdf/1312.5602
# Double DQN: https://arxiv.org/pdf/1509.06461s
################################################
# NOTE: the performance is not stable due to the biased nature of the Q-learning. 
# The reward curve can fluctuate a lot during training, and it can be hard to 
# determine when the model is converged.
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
from src.projects.rl import dqn as _dqn
from src.projects.rl import replay_buffer as _buffer
from src.projects.rl import structure as _struct
from src.utilities import logging


# Running hyperparameters
flags.DEFINE_integer(
    name="batch_size",
    default=512,
    required=False,
    help="Batch size for training and evaluation",
)
flags.DEFINE_integer(
    name="eval_every_n_episodes",
    default=10,
    required=False,
    help="Evaluation frequency (in episodes) during training.",
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
flags.DEFINE_bool(
    name="use_double",
    default=False,
    required=False,
    help="Whether to use Double DQN",
)
flags.DEFINE_float(
    name="learning_rate",
    default=1e-4,
    required=False,
    help="learning rate for training the DQN agent",
)
flags.DEFINE_integer(
    name="buffer_capacity",
    default=30000,
    required=False,
    help="Capacity of the replay buffer",
)
flags.DEFINE_float(
    name="epsilon_start",
    default=1.0, 
    required=False,
    help="Starting value of epsilon for epsilon-greedy policy",
)
flags.DEFINE_float(
    name="epsilon_end",
    default=0.01,
    required=False,
    help="Ending value of epsilon for epsilon-greedy policy",
)
flags.DEFINE_integer(
    name="epsilon_decay_episodes",
    default=3000,
    required=False,
    help="Number of episodes over which epsilon is decayed",
)
flags.DEFINE_integer(
    name="target_update_freq",
    default=3000,
    required=False,
    help="Target network update frequency (in steps)",
)
flags.DEFINE_float(
    name="gamma",
    default=0.99,
    required=False,
    help="Discount factor for future rewards",
)

# the training step function
def train_step(
    rngs: jax.Array,
    state: _train_state.TrainState,
    agent: _model.Model,
    target_params: jax.Array,
    batch: _struct.StepTuple,
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

    # compute loss and gradients using JAX's automatic differentiation
    grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grads_fn(state.params)

    # similar to "theta_new = theta_old - learning_rate * grad" 
    # in vanilla gradient descent
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


def main(_: typing.List[str]) -> int:
    del _  # NOTE: unused arguments
    # Random keys for JAX
    rngs = jax.random.PRNGKey(0)

    # Create Gym cartpole environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape or (1,)
    action_size = env.action_space.shape or (1,)
    logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_size,
        action_size,
    )

    # Create replay buffer
    replay_buffer = _buffer.ReplayBuffer(
        capacity=flags.FLAGS.buffer_capacity,
        state_size=state_size,
        action_size=action_size,
    )

    # Create an instance of the MlpPolicy (inside the DQNModel).
    # For `jax.nn`, we need to initialize the parameters first.
    agent = _dqn.DQNModel(
        action_space_dim=env.action_space.n,  # type: ignore
        gamma=flags.FLAGS.gamma,
        use_double=flags.FLAGS.use_double,
    )

    # Initialize agent parameters
    rngs, init_rng = jax.random.split(rngs, num=2)
    q_params = agent.init(
        batch=_struct.StepTuple(state=jnp.zeros((1, *state_size))),
        rngs=init_rng,
    )
    target_params = copy.deepcopy(q_params)

    # Create train state instance
    train_state = _train_state.TrainState.create(
        params=q_params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )

    # log loss for analysis
    loss_log = []

    # log reward for analysis
    reward_log = []

    # Create two individual rngs for training and sampling
    rngs, train_rng, buffer_rng, sample_rng = jax.random.split(rngs, num=4)
    p_train_step = functools.partial(train_step, rngs=train_rng)
    p_train_step = jax.jit(p_train_step, static_argnames=["agent"])
    p_eval_step = jax.jit(agent.forward)

    # Populates the replay buffer
    logging.rank_zero_info("Populating buffer...")
    state, info = env.reset()
    for step in range(flags.FLAGS.buffer_capacity):
        sample_step_rng = jax.random.fold_in(buffer_rng, step)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, info = env.reset()
    logging.rank_zero_info("Populating buffer... DONE!")

    # The main training loop
    logging.rank_zero_info("Training...")
    for episode in range(flags.FLAGS.num_episodes + 1):
        state, info = env.reset()
        done = False
        episode_losses = []
        episode_reward = 0

        # done marks the end of each episode
        while not done:
            # Epsilon-greedy action selection
            progress = min(1.0, episode / flags.FLAGS.epsilon_decay_episodes)
            epsilon = flags.FLAGS.epsilon_start + progress * \
                (flags.FLAGS.epsilon_end - flags.FLAGS.epsilon_start)
            sample_step_rng = jax.random.fold_in(sample_rng, train_state.step)

            if jax.random.uniform(key=sample_step_rng) < epsilon:
                # Exploration: random action
                action = env.action_space.sample()
            else:
                # Exploitation: select best action based on Q-values
                q_values = p_eval_step(
                    batch=_struct.StepTuple(state=jnp.array(state[None, :])),
                    params=train_state.params,
                ).output
                action = jnp.argmax(q_values, axis=-1).item()

            # Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += float(reward)

            # Sample a batch of experiences from the replay buffer and train
            # the agent
            if len(replay_buffer) >= flags.FLAGS.buffer_capacity:
                sample_key = jax.random.fold_in(sample_rng, train_state.step)
                batch = replay_buffer.sample(
                    key=sample_key,
                    batch_size=flags.FLAGS.batch_size,
                )

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
                if train_state.step % flags.FLAGS.target_update_freq == 0:
                    target_params = copy.deepcopy(train_state.params)
                    logging.rank_zero_info("Target network synced!")

        if episode % flags.FLAGS.eval_every_n_episodes == 0:
            eval_env = gym.make("CartPole-v1")
            eval_reward, eval_episode = [], 0
            while eval_episode < 5:
                state = eval_env.reset()[0]
                done = False
                total_reward = 0

                while not done:
                    # Forward pass (Note: pure exploitation)
                    # Add batch dimension [None, :] because the model expects
                    #  (batch, features)
                    q_values = p_eval_step(
                        batch=_struct.StepTuple(
                            state=jnp.array(state[None, :]),
                        ),
                        params=train_state.params,
                    ).output

                    # Select the best action
                    action = jnp.argmax(q_values, axis=-1).item()

                    # Step
                    (
                        state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = eval_env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)

                eval_reward.append(total_reward)
                eval_episode += 1

            logging.rank_zero_info(
                f"Eval at episode {episode:d} | "
                f"Min Reward = {min(eval_reward)} | "
                f"Max Reward = {max(eval_reward)} | "
                f"Average Reward = {sum(eval_reward) / len(eval_reward)}."
            )

            eval_env.close()

            # Check if environment is solved
            if sum(eval_reward) / len(eval_reward) >= 500.0:
                logging.rank_zero_info(
                    "Environment solved in %d episodes!",
                    episode,
                )
                # break

        # Log episode reward
        reward_log.append(episode_reward)
        # logging.rank_zero_info(
        #     "Episode %d | Episode Reward: %.2f | Episode Loss: %.4f | "
        #     "Epsilon: %.3f",
        #     episode + 1,
        #     episode_reward,
        #     sum(episode_losses) / len(episode_losses) if episode_losses else 0.0,
        #     epsilon,
        # )

    # When the trainning is done, save the serialized model parameters to a file
    with open(os.path.join(flags.FLAGS.work_dir, "dqn_model_params.msgpack"), 
              "wb") as f:
        f.write(serialization.msgpack_serialize(train_state.params))

    # Close the environment
    env.close()

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss Curve
    ax1.plot(loss_log, color="tab:blue", alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("DQN Training Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Reward Curve
    ax2.plot(reward_log, color="tab:orange", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Episode Reward")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig(os.path.join(flags.FLAGS.work_dir, "dqn_loss_curve.png"))
    logging.rank_zero_info("Training loss curve saved as dqn_loss_curve.png")

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
