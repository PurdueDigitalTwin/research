# Generic Reinforcement Learning Infrastructure

**Author**: [Juanwu Lu](https://github.com/juanwulu)

**Status**: Draft

**Type**: Feature

**Created**: 25-Mar-2026

## Overview

Reinforcement Learning (RL) is a powerful paradigm for training agents to make sequential decisions that maximize the cumulative reward. In the context of autonomous driving and digital twin, RL can be useful in end-to-end autonomous driving, fine-tuning the foundtional driving model, and optimizing the multi-agent simulation. Therefore, we need a universal hub for implementing and experimenting with RL algorithms.

In this design note, we aims to build a fundamental infrastructure that supports online and offline, on-policy and off-policy methods, runs on JAX/Flax, and is compatible with distributed training on TPU VMs.

## Architecture

```text
src/projects/rl/
├── main.py                    # Entry point
├── train.py                   # Training loop
├── common.py                  # Base classes + data structures
├── config.py                  # Fiddle experiment configs
├── replay_buffer.py           # Off-policy circular buffer
├── rollout_buffer.py          # On-policy GAE buffer
├── environment/
│   ├── gym_env.py             # GymEnvironment wrapper
│   ├── vector_env.py          # VectorizedGymEnvironment
│   └── wrappers.py            # ObsNormalize, RecordEpisodeStats, etc. (extensible in the future)
├── network/ (common neural network architectures)
└── agents/
    ├── dqn.py                 # Deep Q-Learning
    ├── vpg.py                 # REINFORCE / VPG
    ├── ppo.py                 # Proximal Policy Optimization
    ├── sac.py                 # Soft Actor-Critic
    └── td3.py                 # Twin Delayed DDPG
```

The training loop in `train.py` is the central orchestrator. It dispatches to either an off-policy or on-policy loop based on `agent.is_on_policy`.

## Data Structures (`common.py`)

### `StepTuple`

A dataclass representing a single environment transition `(s, a, r, s', done)`. Used by off-policy algorithms as the unit of experience.

### `RolloutTuple`

Extends the transition tuple with on-policy fields required for policy gradient methods:

| Field          | Type        | Description                                     |
| -------------- | ----------- | ----------------------------------------------- |
| `state`        | `jax.Array` | Observation                                     |
| `action`       | `jax.Array` | Action taken                                    |
| `reward`       | `jax.Array` | Reward received                                 |
| `next_state`   | `jax.Array` | Next observation                                |
| `done`         | `jax.Array` | Terminal flag                                   |
| `log_prob`     | `jax.Array` | $\\log\\pi(a\|s)$ at collection time            |
| `value`        | `jax.Array` | $V(s\_{t})$ estimate from critic                |
| `advantage`    | `jax.Array` | GAE-Lambda advantage (computed post-collection) |
| `return_to_go` | `jax.Array` | Discounted return (computed post-collection)    |

Implemented as a `@chex.dataclass` following the existing `StepTuple` pattern.

### Experiment Config (`common.py`)

`RLExperimentConfig` gains RL-specific training parameters (separate from `TrainerConfig` which is shared with generative models):

```python
@dataclasses.dataclass(frozen=False, kw_only=True)
class RLExperimentConfig:
    # ... existing fields ...

    # Shared
    batch_size: int = 256
    eval_episodes: int = 10

    # Off-policy
    replay_buffer_capacity: int = 1_000_000
    warmup_steps: int = 1_000         # random actions before training starts

    # On-policy
    n_envs: int = 4                   # parallel environments
    n_rollout_steps: int = 2_048      # steps per env before each update
    gae_lambda: float = 0.95          # GAE-Lambda for advantage estimation
```

### Agent Interface (`common.py`)

`BaseAgent` gains one new abstract method used by the training loop for environment interaction:

```python
class BaseAgent(_model.Model, abc.ABC):

    @abc.abstractmethod
    def select_action(
        self,
        obs: jax.Array,
        params: typing.Any,
        rngs: jax.Array,
        *,
        deterministic: bool = False,
    ) -> typing.Any:
        """Select an action given an observation.

        Returns:
            For off-policy: action array.
            For on-policy: (action, log_prob, value) tuple.
        """
        ...
```

All other existing abstract methods remain unchanged (`configure_train_state`, `is_on_policy`, `init`, `forward`, `training_step`).

## Environment Layer (`environment/`)

### `GymEnvironment` (`gym_env.py`)

Thin wrapper around `gymnasium.Env` that implements `BaseEnvironment`. Exposes `observation_space` and `action_space` properties so the training loop and agents can query dimensionality without magic numbers.

```python
class GymEnvironment(BaseEnvironment):
    def __init__(self, env_id: str, render_mode: str | None = None, **kwargs): ...
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def reset(self, *, seed=None) -> tuple[np.ndarray, dict]: ...
    def close(self) -> None: ...

    @property
    def observation_space(self) -> gymnasium.Space: ...

    @property
    def action_space(self) -> gymnasium.Space: ...
```

Fiddle configs use `fdl.Config(GymEnvironment, env_id="CartPole-v1")` (instead of `fdl.Config(gym.make, id=...)`).

### `VectorizedGymEnvironment` (`vector_env.py`)

Used exclusively by the on-policy training loop in non-distributed mode. Wraps `gymnasium.vector.SyncVectorEnv` or `AsyncVectorEnv`.

```python
class VectorizedGymEnvironment:
    def __init__(self, env_id: str, n_envs: int, async_envs: bool = False): ...
    def step(self, actions: np.ndarray) -> tuple: ...
    def reset(self, *, seed=None) -> tuple: ...
    def close(self) -> None: ...

    @property
    def observation_space(self) -> gymnasium.Space: ...
    @property
    def action_space(self) -> gymnasium.Space: ...
    @property
    def n_envs(self) -> int: ...
```

### Helper functions for environment instantiation (`wrappers.py`)

Gymnasium-compatible wrappers that can be composed via `gym.Wrapper`:

- **`RecordEpisodeStats`**: Accumulates episode return and length; writes `info["episode"] = {"r": ..., "l": ...}` on episode end. Used by the training loop to log episode metrics without special-casing.
- **`NormalizeObservation`**: Online running mean/std normalization using Welford's algorithm. Optional — configured per experiment.
- **`ClipReward`**: Clips rewards to `[-max_abs_reward, max_abs_reward]`. Optional.

## Rollout Buffer (`rollout_buffer.py`)

On-policy algorithms require a different buffer structure than off-policy — trajectories must stay ordered and GAE-Lambda must be computed before training. `ReplayBuffer` (circular, uniform sample) is not reused here.

```python
class RolloutBuffer:
    """Fixed-capacity on-policy trajectory buffer with GAE-Lambda."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        state_size: tuple[int, ...],
        action_size: tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None: ...

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None: ...

    def finish_path(self, last_values: np.ndarray) -> None:
        """Compute GAE-Lambda advantages and discounted returns.
        Must be called after the last step of a rollout before get()."""
        ...

    def get(self) -> RolloutTuple:
        """Return full buffer as a batched RolloutTuple (flat across envs)."""
        ...

    def reset(self) -> None:
        """Clear buffer. Call after each training update."""
        ...

    @property
    def is_full(self) -> bool: ...
```

Implementation note: `finish_path` walks backward through the stored rewards and values to compute `δ_t = r_t + γ·V(s_{t+1}) - V(s_t)` and accumulates `A_t = δ_t + γλ·A_{t+1}` (standard GAE-Lambda). The return is `R_t = A_t + V(s_t)`.

## Shared Network Heads (`agents/networks.py`)

All agents share common output head modules defined here. Each is a `flax.linen.Module` that receives a feature vector from a shared backbone (typically `MultiLayerPerceptron` from `src/nn/mlp.py`) and produces algorithm-specific outputs.

| Class                 | Input    | Output                                | Used By                            |
| --------------------- | -------- | ------------------------------------- | ---------------------------------- |
| `DiscreteActorHead`   | features | logits `[action_dim]`                 | DQN (Q-values), PPO/VPG (discrete) |
| `ContinuousActorHead` | features | `(mean, log_std)` each `[action_dim]` | PPO/VPG (continuous), SAC, TD3     |
| `ValueHead`           | features | scalar `V(s)`                         | PPO, VPG (with baseline)           |
| `QHead`               | features | scalar `Q(s,a)`                       | SAC, TD3                           |

`ContinuousActorHead` clips `log_std` to `[log_std_min, log_std_max]` (default `[-20, 2]`) for numerical stability.

## Algorithm Scaffolds (`agents/`)

Each new agent class follows the same pattern as `DQNModel`: extends `BaseAgent`, stores hyperparameters in `__init__`, implements all abstract methods, and uses `@chex.dataclass` or `state.mutables` for per-algorithm state.

### `VPGAgent` (`agents/vpg.py`)

REINFORCE with optional baseline. Simplest on-policy agent; used to validate the on-policy training loop.

- `is_on_policy = True`
- `select_action(obs, params, rngs, deterministic)` → `(action, log_prob, value=0)`
- `training_step(batch: RolloutTuple, ...)` → policy gradient loss using `return_to_go` (no GAE; `advantage = return_to_go` for REINFORCE)
- Supports discrete (categorical) and continuous (Gaussian) action spaces via `DiscreteActorHead` / `ContinuousActorHead`

### `PPOAgent` (`agents/ppo.py`)

Clipped surrogate PPO with combined actor-critic network.

- `is_on_policy = True`
- `select_action(obs, params, rngs, deterministic)` → `(action, log_prob, value)`
- `training_step(batch: RolloutTuple, ...)` computes three losses:
  - **Policy loss:** `L_CLIP = -E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]`
  - **Value loss:** `L_VF = MSE(V(s_t), return_to_go)`
  - **Entropy bonus:** `L_ENT = -β·H[π(·|s_t)]`
  - Total: `L = L_CLIP + c_1·L_VF - c_2·L_ENT`
- Hyperparameters: `clip_eps=0.2`, `vf_coef=0.5`, `ent_coef=0.01`, `n_update_epochs=10`, `minibatch_size=64`

### `SACAgent` (`agents/sac.py`)

Maximum-entropy actor-critic for continuous action spaces.

- `is_on_policy = False`
- Uses reparameterization trick: `a = μ(s) + σ(s)·ε`, `ε ~ N(0,I)`, with tanh squashing
- Stores in `state.mutables`: twin critic params `(q1, q2)`, target critic params `(q1_target, q2_target)`, and `log_alpha` (learnable temperature)
- `training_step(batch: StepTuple, ...)`:
  - **Critic loss:** `L_Q = MSE(Q_i(s,a), y)` where `y = r + γ(1-d)[min_j Q_j_target(s',a') - α·log_π(a'|s')]`
  - **Actor loss:** `L_π = E[α·log_π(a|s) - min_j Q_j(s,a)]`
  - **Temperature loss:** `L_α = E[-log_α·(log_π(a|s) + H_target)]` where `H_target = -|action_dim|`
- `on_train_batch_end()`: Polyak update `θ_target ← τ·θ + (1-τ)·θ_target` for both critics

### `TD3Agent` (`agents/td3.py`)

Twin Delayed DDPG for continuous actions.

- `is_on_policy = False`
- Deterministic policy; exploration via `N(0, σ_explore)` noise during training
- Stores in `state.mutables`: twin critics, target critics, target actor, and step counter for delayed actor updates
- `training_step(batch: StepTuple, ...)`:
  - **Critic loss:** Bellman backup with target policy smoothing: `a_smooth = clip(μ_target(s') + clip(ε, -c, c), a_low, a_high)`
  - **Actor loss** (every `policy_delay` steps only): `-E[Q1(s, μ(s))]`
- `on_train_batch_end()`: Polyak update of target actor + target critics

## Training Loop (`train.py`)

The `run()` function is restructured into:

```text
run(config: RLExperimentConfig) -> int
├── setup_common()       # RNG, W&B init, Orbax checkpoint manager
├── build_env()          # fdl.build + RecordEpisodeStats wrapper
├── build_agent()        # fdl.build + init + configure_train_state
├── build_optimizer()    # lr schedule + grad clip chain (existing logic)
└── if agent.is_on_policy:
│       _run_on_policy(config, agent, state, ...)
    else:
        _run_off_policy(config, agent, state, ...)
```

### Off-policy loop (`_run_off_policy`)

```python
Phase 1 — Warmup:
  for step in range(config.warmup_steps):
      action = env.action_space.sample()   # random
      buffer.add(obs, action, reward, ...)

Phase 2 — Main loop:
  for step in range(config.trainer.num_train_steps):
      action = agent.select_action(obs, state.params, rng, deterministic=False)
      obs_next, reward, terminated, truncated, info = env.step(action)
      buffer.add(obs, action, reward, obs_next, terminated or truncated)
      obs = obs_next if not (terminated or truncated) else env.reset()[0]

      if len(buffer) >= config.batch_size:
          batch = buffer.sample(config.batch_size, rng)
          state, outputs = jax.jit(agent.training_step)(batch, state, rng)
          state = agent.on_train_batch_end(state=state, step=step)

      if step % config.trainer.eval_every_n_steps == 0:
          _evaluate(agent, eval_env, state, config.eval_episodes)

      if step % config.trainer.log_every_n_steps == 0:
          wandb.log(outputs.scalars, step=step)

      if step % config.trainer.checkpoint_every_n_steps == 0:
          checkpoint_manager.save(step, items={"state": state})
```

### On-policy loop (`_run_on_policy`)

```python
# Build vectorized or pmap environment
if distributed:
    envs = single env per device, pmap collect_step
else:
    envs = VectorizedGymEnvironment(env_id, n_envs=config.n_envs)

buffer = RolloutBuffer(config.n_rollout_steps, n_envs, ...)

for epoch in range(n_epochs):
    obs, _ = envs.reset()
    for _ in range(config.n_rollout_steps):
        action, log_prob, value = agent.select_action(obs, state.params, rng)
        obs_next, reward, done, _, info = envs.step(action)
        buffer.add(obs, action, reward, done, log_prob, value)
        obs = obs_next

    _, _, last_value = agent.select_action(obs, state.params, rng, deterministic=True)
    buffer.finish_path(last_value)
    rollout = buffer.get()

    state, outputs = jax.jit(agent.training_step)(rollout, state, rng)
    buffer.reset()

    wandb.log(episode_stats, step=epoch)
    checkpoint_manager.save(epoch, items={"state": state})
```

### Distributed on-policy (when `--distributed=True`)

Each device runs one environment. `jax.pmap` wraps both `select_action` and `training_step`. Advantages are normalized globally via `jax.lax.pmean`. Uses existing `training.shard()` helper from `src/utilities/training.py`.
