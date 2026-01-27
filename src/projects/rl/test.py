import sys
import os

# Add the 'src' directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import time
import abc
import typing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer

# --- JAX/FLAX IMPORTS ---
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jaxtyping
from flax.core import frozen_dict
from flax.training.train_state import TrainState

from core.model import Model, StepOutputs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True # Kept for arg compatibility, though we use JAX
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# --- Flax Network Definition ---
class QNetworkFlax(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# --- Model Implementation adhering to Rules ---
class DQNModel(Model):
    def __init__(self, action_dim: int):
        self._network = QNetworkFlax(action_dim=action_dim)

    @property
    def network(self) -> nn.Module:
        return self._network

    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> jaxtyping.PyTree:
        """Initializes the model parameters."""
        return self.network.init(rngs, batch)

    def forward(
        self,
        *,
        rngs: typing.Any,
        deterministic: bool = True,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> StepOutputs:
        """Runs inference (returns Q-values)."""
        obs = kwargs.get("obs")
        q_values = self.network.apply(params, obs)
        return StepOutputs(output=q_values)

    def compute_loss(
        self,
        *,
        rngs: typing.Any,
        deterministic: bool = False,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> typing.Tuple[jax.Array, StepOutputs]:
        """Computes the TD-error loss."""
        # Unpack kwargs
        target_params = kwargs["target_params"]
        observations = kwargs["observations"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_observations = kwargs["next_observations"]
        dones = kwargs["dones"]
        gamma = kwargs["gamma"]

        # Forward pass on current observations
        q_values = self.network.apply(params, observations)
        
        # FIX: 'actions' is already (Batch, 1), so we use it directly.
        # We also squeeze the result to get shape (Batch,) to match the targets.
        q_action = jnp.take_along_axis(q_values, actions, axis=1).squeeze()

        # Compute Target Q-values using the target network
        next_q_values = self.network.apply(target_params, next_observations)
        next_max_q = jnp.max(next_q_values, axis=1)
        td_target = rewards + gamma * next_max_q * (1 - dones)

        # MSE Loss
        loss = jnp.mean((q_action - td_target) ** 2)

        return loss, StepOutputs(
            scalars={"loss": loss, "q_values": jnp.mean(q_values)}
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # --- Initialize Model & Optimizer ---
    dqn_model = DQNModel(action_dim=envs.single_action_space.n)
    
    # Initialize parameters using a dummy input
    dummy_obs = jnp.zeros((1,) + envs.single_observation_space.shape)
    params = dqn_model.init(batch=dummy_obs, rngs=model_key)
    target_params = params

    # Optimizer setup (Optax)
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    # Define the update step (JIT compiled)
    @jax.jit
    def update_step(
        params: frozen_dict.FrozenDict,
        target_params: frozen_dict.FrozenDict,
        opt_state: optax.OptState,
        batch: dict,
    ):
        def loss_fn(p):
            loss, outputs = dqn_model.compute_loss(
                rngs=None,
                params=p,
                target_params=target_params,
                observations=batch['observations'],
                actions=batch['actions'],
                rewards=batch['rewards'],
                next_observations=batch['next_observations'],
                dones=batch['dones'],
                gamma=args.gamma
            )
            return loss, outputs

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, outputs

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu", # CleanRL buffer handles numpy, no need for GPU specific strings here usually
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # --- Training Loop ---
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # ALGO LOGIC: Action selection
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Use the model's forward method
            # JAX requires numpy array inputs
            step_out = dqn_model.forward(rngs=None, params=params, obs=jnp.array(obs))
            q_values = step_out.output
            actions = np.array(jnp.argmax(q_values, axis=1))

        # Execute game
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Logging
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Buffer storage
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # ALGO LOGIC: Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                # Convert CleanRL buffer data (numpy) to JAX arrays
                batch_data = {
                    'observations': jnp.array(data.observations),
                    'actions': jnp.array(data.actions),
                    'rewards': jnp.array(data.rewards).flatten(),
                    'next_observations': jnp.array(data.next_observations),
                    'dones': jnp.array(data.dones).flatten(),
                }

                # Run optimization step
                params, opt_state, loss, outputs = update_step(
                    params, target_params, opt_state, batch_data
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", outputs.scalars['q_values'].item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Update target network
            if global_step % args.target_network_frequency == 0:
                target_params = optax.incremental_update(params, target_params, step_size=args.tau)

    if args.save_model:
        # Note: Standard torch.save won't work for JAX params.
        # This part requires Flax serialization logic if you want to keep it.
        # For this snippet, we will notify that JAX saving is required.
        print("Note: Model saving enabled, but requires `flax.serialization` for JAX params.")
        
    envs.close()
    writer.close()
