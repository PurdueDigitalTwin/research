import platform

import fiddle as fdl
import jax
from jax import numpy as jnp
import optax

from src.projects.rl import common as _common
from src.utilities import logging


def run(config: _common.RLExperimentConfig) -> int:
    _status = 0
    if not isinstance(config, _common.RLExperimentConfig):
        raise TypeError(
            "Expect `config` to be an `RLExperimentConfig` instance, "
            f"but got {type(config)} instead."
        )

    rng = jax.random.PRNGKey(config.seed)

    # Log the current platform
    logging.rank_zero_info("Running on platform: %s", platform.node())
    logging.rank_zero_info("Running on JAX backend: %s", jax.default_backend())
    logging.rank_zero_info(
        "Running on JAX process: %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    logging.rank_zero_info("Running on JAX devices: %r", jax.devices())

    # build environment
    logging.rank_zero_info("Building environment...")
    rng, env_key = jax.random.split(rng, num=2)
    env = fdl.build(config.environment)
    obs, _ = env.reset(seed=env_key[0].item())
    state_size = obs.shape
    logging.rank_zero_info("Successfully built %r", env)

    # build agent
    logging.rank_zero_info("Building RL agent...")
    rng, init_key = jax.random.split(rng, num=2)
    p_agent = fdl.build(config.agent)
    agent = p_agent(
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        precision=config.precision,
    )
    dummy_batch = _common.StepTuple(
        state=jnp.zeros((1, *state_size), dtype=config.dtype)
    )
    params, mutables = agent.init(batch=dummy_batch, rngs=init_key)
    logging.rank_zero_info("Successfully built %r", agent)

    # build train state
    logging.rank_zero_info("Building train state...")
    lr_scheduler = fdl.build(config.optimizer.lr_schedule)
    p_optimizer = fdl.build(config.optimizer.optimizer)
    tx = p_optimizer(learning_rate=lr_scheduler)
    if config.optimizer.grad_clip_method == "norm":
        tx = optax.chain(
            optax.clip_by_global_norm(config.optimizer.grad_clip_value),
            tx,
        )
    elif config.optimizer.grad_clip_method == "value":
        tx = optax.chain(
            optax.clip(config.optimizer.grad_clip_value),
            tx,
        )
    elif config.optimizer.grad_clip_method is not None:
        logging.rank_zero_error(
            "Unknown grad clip method: %s",
            config.optimizer.grad_clip_method,
        )
        return 1
    state = agent.configure_train_state(
        params=params,
        tx=tx,
        mutables=mutables,
        ema_rate=config.optimizer.ema_rate,
    )
    jax.block_until_ready(state)
    logging.rank_zero_info("Successfully built train state")

    env.close()

    return _status
