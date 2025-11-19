from datetime import datetime
import os
import platform
import typing

from absl import app
from absl import flags
from clu import checkpoint
from clu import metric_writers
from clu import platform as clu_platform
from fiddle import absl_flags
import fiddle as fdl
import jax
import optax
import tensorflow as tf

from src.core import config as _config
from src.core import evaluate as _evaluate
from src.core import train as _train
from src.core import train_state as _train_state
from src.utilities import logging

CONFIG = absl_flags.DEFINE_fiddle_config(
    name="experiment",
    default=None,
    help_string="Experiment configuration.",
    required=True,
)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store the experiment results.",
    required=True,
)


# toggle off GPU/TPU for TensorFlow
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
assert not tf.config.experimental.get_visible_devices("GPU")


def main(_: typing.List[str]) -> int:
    r"""Main entry point for training and evaluate generative models."""
    del _  # unused.

    # Log the current platform
    logging.rank_zero_info("Running on platform: %s", platform.node())

    # Setup JAX runtime
    logging.rank_zero_info("Running on JAX backend: %s", jax.default_backend())
    logging.rank_zero_info(
        "Running on JAX process: %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    logging.rank_zero_info("Running on JAX devices: %r", jax.devices())

    clu_platform.work_unit().set_task_status(
        "process_index: %d, process_count: %d"
        % (jax.process_index() + 1, jax.process_count()),
    )
    clu_platform.work_unit().create_artifact(
        clu_platform.ArtifactType.DIRECTORY,
        FLAGS.work_dir,
        "Working directory.",
    )

    # Setup Experiment
    exp_config = CONFIG.value
    if not isinstance(exp_config, _config.ExperimentConfig):
        logging.rank_zero_error(
            "Expect configuration to be of type `ExperimentConfig`, got %s.",
            type(exp_config),
        )
        return 1
    logging.rank_zero_info("Experiment Configuration:\n%s", exp_config)

    rng = jax.random.PRNGKey(exp_config.seed)
    log_dir = os.path.join(
        FLAGS.work_dir,
        exp_config.name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    writer = metric_writers.create_default_writer(
        logdir=log_dir,
        just_logging=(jax.process_index() > 0),
    )

    logging.rank_zero_info("Building dataset...")
    rng, data_rng = jax.random.split(rng, num=2)
    p_datamodule = fdl.build(exp_config.data.module)
    datamodule = p_datamodule(
        batch_size=exp_config.data.batch_size,
        deterministic=exp_config.data.deterministic,
        drop_remainder=exp_config.data.drop_remainder,
        num_workers=exp_config.data.num_workers,
        rng=data_rng,
    )
    logging.rank_zero_info(
        "Building dataset %s... DONE!",
        datamodule.__class__.__name__,
    )

    logging.rank_zero_info("Building model...")
    rng, init_rng = jax.random.split(rng, num=2)
    p_model = fdl.build(exp_config.model)
    model = p_model(
        dtype=exp_config.dtype,
        param_dtype=exp_config.param_dtype,
        precision=exp_config.precision,
    )
    params = model.init(batch=None, rngs=init_rng)  # NOTE: use dummy batch
    logging.rank_zero_info(
        "Building model %s... DONE!",
        model.__class__.__name__,
    )

    logging.rank_zero_info("Building train state...")
    lr_scheduler = fdl.build(exp_config.optimizer.lr_schedule)
    p_optimizer = fdl.build(exp_config.optimizer.optimizer)
    tx = p_optimizer(learning_rate=lr_scheduler)
    if exp_config.optimizer.grad_clip_method == "norm":
        tx = optax.chain(
            optax.clip_by_global_norm(exp_config.optimizer.grad_clip_value),
            tx,
        )
    elif exp_config.optimizer.grad_clip_method == "value":
        tx = optax.chain(
            optax.clip(exp_config.optimizer.grad_clip_value),
            tx,
        )
    elif exp_config.optimizer.grad_clip_method is not None:
        logging.rank_zero_error(
            "Unknown grad clip method: %s",
            exp_config.optimizer.grad_clip_method,
        )
        return 1
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=exp_config.optimizer.ema_rate,
    )
    logging.rank_zero_info("Building train state... DONE!")

    checkpoint_manager = checkpoint.MultihostCheckpoint(
        os.path.join(log_dir, "checkpoints"),
        max_to_keep=max(2, exp_config.trainer.max_checkpoints_to_keep),
    )
    if exp_config.trainer.checkpoint_dir is not None:
        logging.rank_zero_error("Resuming from checkpoint not implemented.")
        return 1

    if exp_config.mode == "train":
        _train.run(
            model=model,
            state=state,
            datamodule=datamodule,
            num_train_steps=exp_config.trainer.num_train_steps,
            checkpoint_manager=checkpoint_manager,
            writer=writer,
            work_dir=log_dir,
            rng=rng,
            log_every_n_steps=exp_config.trainer.log_every_n_steps,
            eval_every_n_steps=exp_config.trainer.eval_every_n_steps,
            profile=exp_config.trainer.profile,
        )
    elif exp_config.mode == "evaluate":
        _evaluate.run(
            model=model,
            datamodule=datamodule,
            params=params,
            writer=writer,
            work_dir=log_dir,
            rng=rng,
        )
    else:
        logging.rank_zero_error("Mode %s not implemented.", exp_config.mode)
        return 1

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
