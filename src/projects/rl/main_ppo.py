# file created on Feb. 10, 2026
# The purpose of this file is for rl learning (PPO)
################################################
# framework: jax + flax linen
# environment: gym CartPole-v1
# reference: https://arxiv.org/pdf/1707.06347
# https://en.wikipedia.org/wiki/Policy_gradient_method#cite_note-3
# useful code repo: 
# https://github.com/ericyangyu/PPO-for-Beginners?tab=readme-ov-file
################################################
# Note: value-based methods struggle when the action space is large (continous 
# action space). Usually they are less efficient in terms of trainning time.
# Note: policy-based methods can learn stochastic policies where there isn't a 
# single best action, and they can also learn deterministic policies.
# Note: policy-based methods encourages exploration by nature.


import copy
import functools
import os
import typing

from absl import app
from absl import flags
from flax import linen as nn
from flax import serialization
import gymnasium as gym
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import matplotlib.pyplot as plt
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging
from src.projects.rl import ppo
from src.projects.rl import structure as _structure
from src.projects.rl import policy
from src.projects.rl import replay_buffer as _replay_buffer


# Running hyperparameters
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Directory to save training logs and checkpoints.",
)
flags.DEFINE_integer(
    name="num_episodes",
    default=1000,
    required=False,
    help="Number of episodes to train the PPO model.",
)
flags.DEFINE_integer(
    name="Buffer_capacity",
    default=30000,
    required=False,
    help="Maximum size of the replay buffer to store experience tuples.",
)
flags.DEFINE_float(
    name="gamma",
    default=0.99,
    required=False,
    help="Discount factor for future rewards in PPO training.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=3e-4,
    required=False,
    help="Learning rate for the PPO optimizer.",
)
flags.DEFINE_float(
    name="clip_epsilon",
    default=0.2,
    required=False,
    help="Clipping parameter for PPO's surrogate objective.",
)
flags.DEFINE_float(
    name="lambda_gae",
    default=0.95,
    required=False,
    help="GAE parameter for advantage estimation in PPO.",
)
flags.DEFINE_float(
    name="value_coeff",
    default=0.5,
    required=False,
    help="Coefficient for the value function loss in the total PPO loss.",
)
flags.DEFINE_float(
    name="entropy_coeff",
    default=0.0,
    required=False,
    help="Coefficient for the entropy bonus in the total PPO loss.",
)
flags.DEFINE_integer(
    name="batch_size",
    default=256,
    required=False,
    help="Batch size for sampling experience from the replay buffer during PPO training.",
)


def compute_gae(
    batch: _structure.StepTuple,
    gamma: float,
    lambda_gae: float,
) -> jax.Array:
    r"""Computes Generalized Advantage Estimation (GAE) for a batch of transitions.

    Args:
        batch (StepTuple): A batch of experience tuples containing states, actions,
            rewards, next states, and done flags.
        gamma (float): Discount factor for future rewards.
        lambda_gae (float): GAE parameter controlling bias-variance trade-off.
        gamma (float): Discount factor for future rewards.
        lambda_gae (float): GAE parameter controlling bias-variance trade-off.
    Returns:
        jax.Array: Computed advantages for each transition, shape (batch_size,).
    """

    # # Compute the GAE for each transition in the batch
    # advantages = jnp.zeros_like(batch.reward)
    # gae = 0.0

    # # Iterate backwards through the batch to compute GAE
    # for i in reversed(range(len(batch.reward))):
    #     if i == len(batch.reward) - 1:
    #         next_value = batch.next_value[i]
    #     else:
    #         next_value = batch.next_value[i + 1]

    #     # Compute TD error
    #     td_error = batch.reward[i] + gamma * next_value * (1.0 - \
    #         batch.done[i]) - batch.value[i]

    #     # Update GAE using the recurrence relation
    #     gae = td_error + gamma * lambda_gae * (1.0 - batch.done[i]) * gae

    #     # Store the computed advantage for this transition
    #     advantages = advantages.at[i].set(gae)

    # return advantages
    return jnp.zeros_like(batch.reward), jnp.zeros_like(batch.reward)


def train_step(
    rngs: jax.Array,
    state: _train_state.TrainState,
    agent: _model.Model,
    batch: _structure.StepTuple,
    log_old_act_probs: jax.Array,
    value_targets: jax.Array,
    advantages: jax.Array,
) -> typing.Tuple[_train_state.TrainState, jax.Array]:
    r"""Performs a single training step for the PPO model.

    Args:
        rngs (jax.Array): Random number generator keys for stochastic operations.
        state (_train_state.TrainState): Current training state containing model
            parameters and optimizer state.
        agent (_model.Model): The PPO model instance to be trained.
        batch (StepTuple): A batch of experience tuples containing states, actions,
            rewards, next states, and done flags.
        log_old_act_probs (jax.Array): Log action probabilities of the old 
            policy for the given batch of transitions.
        value_targets (jax.Array): Computed value targets for each transition 
            in the batch.
        advantages (jax.Array): Computed advantages for each transition in the batch.
    Returns:
        Tuple[TrainState, jax.Array]: Updated training state and computed loss for
            the training step.
    """

    local_rngs = jax.random.fold_in(rngs, state.step)

    # Compute the loss and gradients
    def loss_fn(params: jaxtyping.PyTree) -> jax.Array:
        return agent.compute_loss(
            params=params,
            batch=batch,
            log_old_act_probs=log_old_act_probs,
            advantages=advantages,
            value_targets=value_targets,
            rngs=local_rngs,
        )

    # Compute gradients of the loss with respect to the model parameters
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)

    # similar to "theta_new = theta_old - learning_rate * grad" 
    # in vanilla gradient descent
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


def main(argv: typing.List[str]) -> None:
    r"""Main function to set up the environment, model, and training loop for PPO.

    Args:
        argv (List[str]): Command-line arguments (not used in this implementation).
    Returns:
        None.
    """
    del argv  # Unused.

    # Random key for reproducibility
    rngs = jax.random.PRNGKey(0)

    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_shape,
        action_shape,
    )

    # Create a replay buffer to store experience tuples
    replay_buffer = _replay_buffer.ReplayBuffer(
        capacity=flags.FLAGS.Buffer_capacity,
        state_size=state_shape,
        action_size=(action_shape,),
    )

    # Initialize the PPO policy network
    agent = ppo.PPOModel(
        action_space_dim=action_shape,
        gamma=flags.FLAGS.gamma,
        clip_epsilon=flags.FLAGS.clip_epsilon,
        lambda_gae=flags.FLAGS.lambda_gae,
        value_coeff=flags.FLAGS.value_coeff,
        entropy_coeff=flags.FLAGS.entropy_coeff,
    )

    # Initialize agent parameters
    rngs, init_rng = jax.random.split(rngs)

    params = agent.init(
        batch=_structure.StepTuple(
            state=jnp.zeros(state_shape, dtype=jnp.float32)
        ),
        rngs=init_rng,
    )

    # Create a training state instance with adam optimizer
    train_state = _train_state.TrainState.create(
        params=params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )

    # Log loss and rewards during training
    loss_log = []
    reward_log = []

    # Create two independent random keys for and training and sampling
    rngs, train_rng, buffer_rng, sample_rng, eval_rng = jax.random.split(rngs, num=5)
    p_train_step = functools.partial(train_step, rngs=train_rng)
    p_train_step = jax.jit(p_train_step, static_argnames=["agent"])
    p_eval_step = jax.jit(agent.forward)

    # NOTE: PPO uses on-policy, do we still need a replay buffer?
    # Populate the replay buffer with initial experience
    logging.rank_zero_info("Populating buffer...")
    state, _ = env.reset()
    for step in range(flags.FLAGS.Buffer_capacity):
        sample_step_rng = jax.random.fold_in(buffer_rng, step)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, _ = env.reset()
    logging.rank_zero_info("Populating buffer... DONE!")

    # The main training loop
    for episode in range(flags.FLAGS.num_episodes + 1):

        while not done:

            if len(replay_buffer) >= flags.FLAGS.Buffer_capacity:

                # Sample a batch of experience from the replay buffer
                sample_key = jax.random.fold_in(sample_rng, train_state.step)
                batch = replay_buffer.sample(
                    key=sample_key,
                    batch_size=flags.FLAGS.batch_size,
                )

                # Compute old action probabilities and advantages for the sampled 
                # batch
                old_act_logits, old_values = p_eval_step(
                    batch=batch,
                    params=train_state.params,
                    rngs=eval_rng,
                )

                # Apply log softmax to get log action probabilities
                log_old_act_probs = jax.nn.log_softmax(old_act_logits)

                # TODO: compute advantages using GAE.
                advantages, value_targets = compute_gae(
                    batch=batch,
                    gamma=flags.FLAGS.gamma,
                    lambda_gae=flags.FLAGS.lambda_gae,
                )

                # Perform a JIT-compiled training step
                train_state, loss = p_train_step(
                    state=train_state,
                    agent=agent,
                    batch=batch,
                    log_old_act_probs=log_old_act_probs,
                    value_targets=value_targets,
                    advantages=advantages,
                )

                # Log the loss for analysis
                loss_log.append(loss)

        # Log the episode reward for analysis
        reward_log.append(jnp.sum(batch.reward))

        if episode % 10 == 0:
            logging.rank_zero_info(
                "Episode %d: Loss = %.4f, Reward = %.2f",
                episode,
                loss,
                jnp.sum(batch.reward),
            )
    
    # When the trainning is done, save the serialized model parameters to a file
    with open(os.path.join(flags.FLAGS.work_dir, "ppo_model_params.msgpack"), 
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
    ax1.set_title("PPO Training Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Reward Curve
    ax2.plot(reward_log, color="tab:orange", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Episode Reward")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig(os.path.join(flags.FLAGS.work_dir, "ppo_loss_curve.png"))
    logging.rank_zero_info("Training loss curve saved as ppo_loss_curve.png")

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
