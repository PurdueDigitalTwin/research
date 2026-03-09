import fiddle as fdl
import gymnasium as gym
import optax

from src.core import config as _config
from src.projects.rl import common as _common
from src.projects.rl.agents import dqn


# ==============================================================================
# Configuration constructors
def dqn_mlp_cartpole() -> _common.RLExperimentConfig:
    cfg = _common.RLExperimentConfig(
        project_name="dqn",
        exp_name="cartpole_v0_mlp",
        agent=fdl.Partial(
            dqn.DQNModel,
            action_space_dim=2,
            gamma=0.99,
            q_target_update_freq=3_000,
            use_double=False,
        ),
        environment=fdl.Config(gym.make, id="CartPole-v1"),  # type: ignore
        trainer=_config.TrainerConfig(
            num_train_steps=50_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,
            eval_every_n_steps=10_000,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.constant_schedule,
                value=1e-4,
            ),
            optimizer=fdl.Partial(optax.adam),
        ),
        seed=42,
    )

    return cfg
