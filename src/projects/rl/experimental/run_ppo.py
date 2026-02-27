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
# NOTE: why using advantage instead of value target?
# NOTE: why KL equals second order direvative


import copy
import functools
import os
import typing
import numpy as np

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
from src.projects.rl.experimental import ppo
from src.projects.rl import structure as _structure
from src.projects.rl import policy


# Running hyperparameters
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Directory to save training logs and checkpoints.",
)
flags.DEFINE_integer(
    name="num_episodes",
    default=50,
    required=False,
    help="Number of episodes to train the PPO model.",
)
flags.DEFINE_integer(
    name="rollout_steps",
    default=2048,
    required=False,
    help="Number of steps to collect in each rollout phase before updating the " \
    "PPO model.",
)
flags.DEFINE_float(
    name="gamma",
    default=0.99,
    required=False,
    help="Discount factor for future rewards in PPO training.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=1e-3,
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
    default=1.0,
    required=False,
    help="Coefficient for the value function loss in the total PPO loss.",
)
flags.DEFINE_float(
    name="entropy_coeff",
    default=0.01,
    required=False,
    help="Coefficient for the entropy bonus in the total PPO loss.",
)
flags.DEFINE_integer(
    name="minibatch",
    default=64,
    required=False,
    help="Size of minibatches for PPO updates.",
)
flags.DEFINE_integer(
    name="training_epochs",
    default=1,
    required=False,
    help="Number of epochs to train on each rollout batch.",
)
flags.DEFINE_integer(
    name="num_envs",
    default=32,
    required=False,
    help="Number of trajectories to collect for evaluation.",
)


# NOTE: we need to use the updated values to compute GAE


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    gamma: float,
    lambda_gae: float,
) -> jax.Array:
    r"""Computes Generalized Advantage Estimation (GAE) for the given rollout data.

    Args:
        rewards (jax.Array): Rewards collected during the rollout, with shape 
            `(rollout_steps, num_envs)`.
        values (jax.Array): Value estimates for each state in the rollout, with
            shape `(rollout_steps, num_envs)`.
        dones (jax.Array): Binary indicators of episode termination for each step
            in the rollout, with shape `(rollout_steps, num_envs)`.
        next_values (jax.Array): Value estimates for the state following each step
            in the rollout, with shape `(rollout_steps, num_envs)`.
        gamma (float): Discount factor for future rewards.
        lambda_gae (float): GAE parameter controlling bias-variance trade-off.
    Returns:
        jax.Array: Computed advantages for each step in the rollout, with
            shape `(rollout_steps, num_envs)`.
    """

    transitions = (rewards, values, next_values, dones)

    # Experiment: decouple advantage and value target
    # next_values = jnp.concatenate([values[1:], next_values[-1:]], axis=0)
    
    # We scan backwards through the trajectory
    def scan_surrogate_fn(gae_carry: jax.Array, transition: typing.Tuple) -> \
        typing.Tuple[jax.Array, jax.Array]:
        reward, value, next_val, done = transition
        
        # Temporal Difference Error: delta = r + gamma * V(s') * (1 - done) - V(s)
        delta = reward + gamma * next_val * (1.0 - done) - value
        
        # GAE: A = delta + gamma * lambda * (1 - done) * A_{t+1}
        gae = delta + gamma * lambda_gae * (1.0 - done) * gae_carry
        return gae, gae
    
    # Initialize the carry to be all zeros
    surrogate_init_carry = jnp.zeros_like(next_values[0], dtype=jnp.float32)
    
    # Run lax.scan in reverse
    _, advantages = lax.scan(scan_surrogate_fn, surrogate_init_carry, 
                             transitions, reverse=True)
    
    # advantages = jax.lax.stop_gradient(advantages)
    
    return advantages


def train_step(
    rngs: jax.Array,
    train_state: _train_state.TrainState,
    agent: _model.Model,
    state: jax.Array,
    action: jax.Array,
    log_old_act_probs: jax.Array,
    value_targets: jax.Array,
    advantages: jax.Array,
) -> typing.Tuple[_train_state.TrainState, jax.Array]:
    r"""Performs a single training step for the PPO model.

    Args:
        rngs (jax.Array): Random number generator keys for stochastic operations.
        train_state (_train_state.TrainState): Current training state containing 
            model parameters and optimizer state.
        agent (_model.Model): The PPO model instance to be trained.
        state (jax.Array): Input state array of shape `(*, D)`.
        action (jax.Array): Actions taken in the rollout, with shape `(*,)`
            where each entry is an integer representing the action index.
        log_old_act_probs (jax.Array): Log action probabilities of the old 
            policy for the given rollout.
        value_targets (jax.Array): Computed value targets for each transition 
            in the rollout.
        advantages (jax.Array): Computed advantages for each transition in the 
            rollout.
    Returns:
        Tuple[TrainState, jax.Array]: Updated training state and computed loss for
            the training step.
    """

    local_rngs = jax.random.fold_in(rngs, train_state.step)

    # Compute the loss and gradients
    def loss_fn(params: jaxtyping.PyTree) -> tuple[jax.Array, _model.StepOutputs]:
        return agent.compute_loss(
            state=state,
            action=action,
            params=params,
            log_old_act_probs=log_old_act_probs,
            advantages=advantages,
            value_targets=value_targets,
            rngs=local_rngs,
        )

    # Compute gradients of the loss with respect to the model parameters
    grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, step_outputs), grads = grads_fn(train_state.params)

    # similar to "theta_new = theta_old - learning_rate * grad" 
    # in vanilla gradient descent
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, step_outputs    


def main(argv: typing.List[str]) -> int:
    r"""Main function to set up the environment, model, and training loop for PPO.
    
    Note: 
        This function initializes the CartPole environment, sets up the PPO model,
        and runs the main training loop where it collects rollouts, computes GAE, 
        and updates the PPO model using the collected data. It also logs training
        progress and saves the trained model parameters and loss curves at the end.

    Args:
        argv (List[str]): Command-line arguments (not used in this implementation).
    Returns:
        int: Exit code (0 for success).
    """
    del argv  # Unused.

    # Random key for reproducibility
    rngs = jax.random.PRNGKey(0)

    # Create the CartPole environment
    # Run N envs in parallel to collect N trajectories for each rollout
    num_envs = flags.FLAGS.num_envs
    env = gym.make_vec("CartPole-v1", num_envs=num_envs)

    single_obs_space = env.single_observation_space
    single_act_space = env.single_action_space

    # NOTE: forr now we are using discrete action space
    assert isinstance(single_act_space, gym.spaces.Discrete)

    state_shape = single_obs_space.shape
    action_shape = single_act_space.n

    logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_shape,
        action_shape,
    )

    # Initialize the PPO policy network
    agent = ppo.PPOModel(
        action_space_dim=action_shape.astype(int),
        gamma=flags.FLAGS.gamma,
        clip_epsilon=flags.FLAGS.clip_epsilon,
        lambda_gae=flags.FLAGS.lambda_gae,
        value_coeff=flags.FLAGS.value_coeff,
        entropy_coeff=flags.FLAGS.entropy_coeff,
    )

    # Initialize agent parameters
    rngs, init_rng = jax.random.split(rngs)
    assert state_shape is not None
    params = agent.init(
        state=jnp.zeros((1, *state_shape)),
        rngs=init_rng,
    )

    # Optional: use annealing learning rate to prevent collapsing
    # Each episode has (Rollout / Minibatch) updates * Training Epochs
    updates_per_episode = (flags.FLAGS.rollout_steps // flags.FLAGS.minibatch) * \
        flags.FLAGS.training_epochs
    total_updates = flags.FLAGS.num_episodes * updates_per_episode

    lr_schedule = optax.linear_schedule(
        init_value=flags.FLAGS.learning_rate,
        end_value=3e-7,  # Keep a small "learning floor"
        transition_steps=total_updates 
    )

    # Create a training state instance with adam optimizer
    train_state = _train_state.TrainState.create(
        params=params,
        tx=optax.adam(learning_rate=3e-4),
    )

    # Log loss and rewards during training
    surrogate_loss_log = []
    value_loss_log = []
    loss_log = []
    reward_log = []
    prob_ratio_mean_log = []

    # Create two independent random keys for and training and sampling
    rngs, train_rng, sample_rng = jax.random.split(rngs, num=3)

    # JIT-compile the training step and evaluation step for efficiency.
    p_train_step = functools.partial(train_step, rngs=train_rng)
    p_train_step = jax.jit(p_train_step, static_argnames=["agent"])
    p_eval_step = jax.jit(agent.forward)

    # The main training loop
    for episode in range(flags.FLAGS.num_episodes):

        # state shape: (num_envs, state_dim)
        state, _ = env.reset()
        done = False

        # Rollout: collect experience by interacting with the environment 
        # using the current policy
        # logging.rank_zero_info("Collecting rollout data for episode %d", episode)

        rollout_states, rollout_actions, rollout_rewards = [], [], []
        rollout_dones, rollout_values, rollout_log_probs = [], [], []
        rollout_next_states = []

        for rollout_step in range(flags.FLAGS.rollout_steps):

            # Evaluate current policy
            logits, value = p_eval_step(
                params=train_state.params,
                state=state,
            )

            # squeeze value from shape (num_envs, 1) to (num_envs,)
            value = value.squeeze(axis=-1)

            # Sample action from logits
            sample_key = jax.random.fold_in(sample_rng, rollout_step)
            # action shape: (num_envs,) where each entry is an integer representing
            # the action index for each environment in the vectorized environment.
            action = jax.random.categorical(sample_key, logits)
            
            # Get log prob of taken action
            curr_log_prob = jax.nn.log_softmax(logits)
            log_prob = jnp.take_along_axis(
                curr_log_prob,
                action[..., None],
                axis=-1,
            ).squeeze(axis=-1)

            # Step the environment
            # each has the shape (num_envs, *)
            # NOTE: can I do this via jnp?
            next_state, reward, terminated, truncated, _ = \
                env.step(np.asarray(action))
            done = terminated | truncated

            # Store transition
            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            rollout_values.append(value)
            rollout_log_probs.append(log_prob)
            rollout_next_states.append(next_state)

            state = next_state

            # vectorized env has automatic reset so we don't need to manually reset
            # the environment when done is True.

        # Convert lists to JAX arrays
        # each has shape (rollout_steps, num_envs, *)
        states_arr = jnp.array(rollout_states)
        actions_arr = jnp.array(rollout_actions)
        rewards_arr = jnp.array(rollout_rewards)
        dones_arr = jnp.array(rollout_dones)
        values_arr = jnp.array(rollout_values)
        log_probs_arr = jnp.array(rollout_log_probs)
        next_states_arr = jnp.array(rollout_next_states)

        # GAE: compute advantages and value targets for the collected rollout data
        # logging.rank_zero_info("Computing GAE for episode %d", episode)

        # Get value of the state that follows the rollout
        # NOTE: to obey bellman equation, the next value should be learned rather 
        # than using the current rollout traj.
        # next_values shape: (rollout_steps, num_envs)
        _, next_values = p_eval_step(
            params=train_state.params,
            state=next_states_arr,
        )
        next_values = next_values.squeeze(axis=-1)

        # Compute advantages and value targets
        advantages = compute_gae(
            rewards=rewards_arr,
            dones=dones_arr,
            values=values_arr,
            next_values=next_values,
            gamma=flags.FLAGS.gamma,
            lambda_gae=flags.FLAGS.lambda_gae,
        )

        # Normalize the advantages (not mention by the reference)
        # NOTE: sinc the variance of the advantage is 1, L_VF should ideally 
        # converge to 1.
        advantages = (advantages - jnp.mean(advantages)) / \
            (jnp.std(advantages) + 1e-8)
        
        # Compute value targets.
        value_targets = rewards_arr + flags.FLAGS.gamma * next_values * (1.0 - \
                                                                         dones_arr)

        # Update the PPO model using the collected rollout data and 
        # computed advantages
        # logging.rank_zero_info("Updating PPO model for episode %d", episode)
        # Update the PPO model using minibatches
        # Flatten the arrays before minibatching
        states_arr = states_arr.reshape(-1, *state_shape)
        actions_arr = actions_arr.reshape(-1)
        log_probs_arr = log_probs_arr.reshape(-1)
        advantages = advantages.reshape(-1)
        value_targets = value_targets.reshape(-1)

        batch_size = flags.FLAGS.minibatch
        # Update num_steps to the total transition count
        num_total_steps = flags.FLAGS.rollout_steps * num_envs
        
        episode_surrogate_loss = []
        episode_value_loss = []
        episode_loss = []
        episode_prob_ratio_mean = []
        
        # sample mini-batches M <= NT
        for _ in range(flags.FLAGS.training_epochs): # train for k epochs
            # Shuffle the indices at the start of each epoch
            rngs, shuffle_rng = jax.random.split(rngs)
            permutation = jax.random.permutation(shuffle_rng, num_total_steps)
            
            # Iterate through the buffer in minibatches
            for i in range(0, num_total_steps, batch_size):
                indices = permutation[i : i + batch_size]
                
                # Slice the data for this minibatch
                mb_states = states_arr[indices]
                mb_actions = actions_arr[indices]
                mb_log_probs = log_probs_arr[indices]
                mb_advantages = advantages[indices]
                mb_value_targets = value_targets[indices]

                # Update the model using the minibatch
                train_state, step_outputs = p_train_step(
                    rngs=train_rng,
                    train_state=train_state,
                    agent=agent,
                    state=mb_states,
                    action=mb_actions,
                    log_old_act_probs=mb_log_probs,
                    value_targets=mb_value_targets,
                    advantages=mb_advantages,
                )
                
                # Track metrics
                episode_surrogate_loss.append(float(step_outputs.
                                                    scalars["surrogate_loss"]))
                episode_value_loss.append(float(step_outputs.scalars["value_loss"]))
                episode_loss.append(step_outputs.scalars["loss"])
                episode_prob_ratio_mean.append(
                    step_outputs.scalars["prob_ratio_mean"]
                )

        # Logging
        mean_episode_surrogate_loss = jnp.mean(jnp.array(episode_surrogate_loss))
        mean_episode_value_loss = jnp.mean(jnp.array(episode_value_loss))
        mean_episode_loss = jnp.mean(jnp.array(episode_loss))
        mean_episode_prob_ratio_mean = jnp.mean(jnp.array(episode_prob_ratio_mean))

        surrogate_loss_log.append(float(mean_episode_surrogate_loss))
        value_loss_log.append(float(mean_episode_value_loss))
        loss_log.append(float(mean_episode_loss))
        prob_ratio_mean_log.append(float(mean_episode_prob_ratio_mean))

        # Evaluate the current policy every 10 episodes by running it in the 
        # environment.
        if episode % 1 == 0:
            eval_env = gym.make("CartPole-v1")
            eval_reward = []

            # Run 5 evaluation episodes and average the total reward to get a 
            # more stable estimate of the policy's performance.
            for _ in range(5):
                state, _ = eval_env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    logits, _ = p_eval_step(
                        state=state,
                        params=train_state.params,
                    )
                    action = jnp.argmax(logits)
                    state, reward, terminated, truncated, _ = \
                        eval_env.step(int(action))
                    done = terminated or truncated
                    episode_reward += float(reward)

                eval_reward.append(episode_reward)
                
            mean_eval_reward = jnp.mean(jnp.array(eval_reward))
            reward_log.append(float(mean_eval_reward))

            logging.rank_zero_info(
                "Episode %d: surrogate loss: %.4f, value loss: %.4f, Loss = %.4f, \
                    Eval Reward = %.2f, Mean Prob Ratio = %.4f",
                episode,
                float(mean_episode_surrogate_loss),
                float(mean_episode_value_loss),
                float(mean_episode_loss),
                float(mean_eval_reward),
                float(mean_episode_prob_ratio_mean),
            )
    
    # When the trainning is done, save the serialized model parameters to a file
    with open(os.path.join(flags.FLAGS.work_dir, "ppo_model_params.msgpack"), 
              "wb") as f:
        f.write(serialization.msgpack_serialize(train_state.params))

    # Close the environment
    env.close()

    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    (ax1, ax2, ax3, ax4, ax5, _) = axes.flatten()

    # Plot 1: Total Loss (Surrogate Loss + Value Loss)
    ax1.plot(loss_log, color="tab:blue", alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("PPO Total Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Evaluation Reward
    ax2.plot(reward_log, color="tab:orange", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Evaluation Reward")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Surrogate Loss
    ax3.plot(surrogate_loss_log, color="tab:green", alpha=0.7)
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Surrogate Loss")
    ax3.set_title("Surrogate Loss (L_CLIP)")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Plot 4: Value Loss
    ax4.plot(value_loss_log, color="tab:red", alpha=0.7)
    ax4.set_xlabel("Training Steps")
    ax4.set_ylabel("Value Loss")
    ax4.set_title("Value Loss (L_VF)")
    ax4.grid(True, linestyle="--", alpha=0.6)

    ax5.plot(prob_ratio_mean_log, color="tab:orange", linewidth=2)
    ax5.set_xlabel("Training Steps")
    ax5.set_ylabel("Mean Probability Ratio")
    ax5.set_title("Mean Probability Ratio Across Episodes")
    ax5.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig(os.path.join(flags.FLAGS.work_dir, "ppo_loss_curve.png"))
    logging.rank_zero_info("Training loss curve saved as ppo_loss_curve.png")

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
