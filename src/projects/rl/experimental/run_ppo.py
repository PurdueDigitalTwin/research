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
    default=1000,
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


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_value: jax.Array,
    gamma: float,
    lambda_gae: float,
) -> typing.Tuple[jax.Array, jax.Array]:
    r"""Computes Generalized Advantage Estimation (GAE) for

    Args:
        rewards (jax.Array): Rewards collected during the rollout, with shape 
            `(rollout_steps,)`.
        values (jax.Array): Value estimates for each state in the rollout, with
            shape `(rollout_steps,)`.
        dones (jax.Array): Binary indicators of episode termination for each step
            in the rollout, with shape `(rollout_steps,)`.
        next_value (jax.Array): Value estimate for the state following the last
            step in the rollout, with shape `()`.
        gamma (float): Discount factor for future rewards.
        lambda_gae (float): GAE parameter controlling bias-variance trade-off.
    Returns:
        Tuple[jax.Array, jax.Array]: A tuple containing:
            - advantages: Computed advantages for each step in the rollout, with
                shape `(rollout_steps,)`.
            - value_targets: Computed value targets for each step in the rollout,
                with shape `(rollout_steps,)`.
    """

    # Squeeze all inputs to ensure they are strictly 1D arrays (T,)
    # This prevents shape broadcasting errors inside lax.scan
    rewards = jnp.squeeze(rewards)
    values = jnp.squeeze(values)
    dones = jnp.squeeze(dones)
    next_value = jnp.squeeze(next_value)

    # -------------------------------------------------------------------------
    # Equation 12: \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
    # -------------------------------------------------------------------------
    # Shift values by 1 to represent V(s_{t+1}) and append the final next_value
    next_values = jnp.append(values[1:], next_value)
    
    # Calculate delta for the whole batch. 
    # We multiply by (1.0 - dones) so V(s_{t+1}) is 0 if the episode ended.
    deltas = rewards + gamma * next_values * (1.0 - dones) - values

    # -------------------------------------------------------------------------
    # Equation 11: \hat{A}_t = \delta_t + (\gamma \lambda)\delta_{t+1} + ...
    # -------------------------------------------------------------------------
    # This sum can be computed efficiently as a backward recurrence:
    # \hat{A}_t = \delta_t + (\gamma \lambda) * \hat{A}_{t+1}
    def scan_fn(gae_carry: jax.Array, transition: typing.Tuple) -> typing.Tuple[jax.Array, jax.Array]:
        delta, done = transition
        
        # If the episode ended (done=1.0), the advantage chain breaks (carry becomes 0)
        gae = delta + gamma * lambda_gae * (1.0 - done) * gae_carry
        return gae, gae

    # Initialize the carry strictly as a float32 scalar
    init_carry = jnp.array(0.0, dtype=jnp.float32)
    
    # Run jax.lax.scan backward through the deltas
    _, advantages = lax.scan(scan_fn, init_carry, (deltas, dones), reverse=True)
    
    # -------------------------------------------------------------------------
    # Compute Value Targets (Returns)
    # -------------------------------------------------------------------------
    # V_{target} = Advantage + V(s_t)
    value_targets = advantages + values
    
    return advantages, value_targets


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
    grads_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grads_fn(train_state.params)

    # similar to "theta_new = theta_old - learning_rate * grad" 
    # in vanilla gradient descent
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, loss


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
    env = gym.make("CartPole-v1")
    state_shape = env.observation_space.shape

    # NOTE: For now we use a discrete action space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_shape = env.action_space.n

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

    # Create a training state instance with adam optimizer
    train_state = _train_state.TrainState.create(
        params=params,
        tx=optax.adam(learning_rate=flags.FLAGS.learning_rate),
    )

    # Log loss and rewards during training
    loss_log, reward_log = [], []

    # Create two independent random keys for and training and sampling
    rngs, train_rng, sample_rng, eval_rng = jax.random.split(rngs, num=4)

    # JIT-compile the training step and evaluation step for efficiency.
    p_train_step = functools.partial(train_step, rngs=train_rng)
    p_train_step = jax.jit(p_train_step, static_argnames=["agent"])
    p_eval_step = jax.jit(agent.forward)

    # The main training loop
    for episode in range(flags.FLAGS.num_episodes):

        state, _ = env.reset()
        done = False

        episode_rewards = []
        current_ep_reward = 0.0

        # Rollout: collect experience by interacting with the environment 
        # using the current policy
        # logging.rank_zero_info("Collecting rollout data for episode %d", episode)

        rollout_states, rollout_actions, rollout_rewards = [], [], []
        rollout_dones, rollout_values, rollout_log_probs = [], [], []

        for step in range(flags.FLAGS.rollout_steps):

            # Evaluate current policy
            logits, value = p_eval_step(
                params=train_state.params,
                rngs=eval_rng,
                state=state,
            )

            # Sample action from logits
            action = jax.random.categorical(sample_rng, logits)
            
            # Get log prob of taken action
            log_prob = jax.nn.log_softmax(logits)[action]

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            # Store transition
            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_rewards.append(reward)
            rollout_dones.append(1.0 if done else 0.0)
            rollout_values.append(value)
            rollout_log_probs.append(log_prob)

            current_ep_reward += float(reward)
            state = next_state

            if done:
                state, _ = env.reset()
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0

        # Convert lists to JAX arrays
        states_arr = jnp.array(rollout_states)
        actions_arr = jnp.array(rollout_actions)
        rewards_arr = jnp.array(rollout_rewards)
        dones_arr = jnp.array(rollout_dones)
        values_arr = jnp.array(rollout_values)
        log_probs_arr = jnp.array(rollout_log_probs)

        # GAE: compute advantages and value targets for the collected rollout data
        # logging.rank_zero_info("Computing GAE for episode %d", episode)

        # Get value of the state that follows the rollout
        # TODO: check this logic
        _, next_value = p_eval_step(
            state=state,
            params=train_state.params,
            rngs=eval_rng,
        )

        # Compute advantages and value targets
        advantages, value_targets = compute_gae(
            rewards=rewards_arr,
            dones=dones_arr,
            values=values_arr,
            next_value=next_value,
            gamma=flags.FLAGS.gamma,
            lambda_gae=flags.FLAGS.lambda_gae,
        )

        # Update the PPO model using the collected rollout data and 
        # computed advantages
        # logging.rank_zero_info("Updating PPO model for episode %d", episode)
        # Train on the collected rollout data
        train_state, loss = p_train_step(
            rngs=train_rng,
            train_state=train_state,
            agent=agent,
            state=states_arr,
            action=actions_arr,
            log_old_act_probs=log_probs_arr,
            value_targets=value_targets,
            advantages=advantages,
        )

        # Logging
        loss_log.append(float(loss))
        mean_reward = jnp.mean(jnp.array(episode_rewards))
        reward_log.append(mean_reward)

        if episode % 10 == 0:
            logging.rank_zero_info(
                "Episode %d: Loss = %.4f, Mean Episode Reward = %.2f",
                episode,
                loss,
                mean_reward,
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
