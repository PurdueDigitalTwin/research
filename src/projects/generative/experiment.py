import dataclasses
import functools
import pathlib
import platform
import typing

import fiddle as fdl
import jax
from jax import numpy as jnp
import jaxtyping
import optax
from orbax import checkpoint as ocp
import tensorflow as tf
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging
import wandb
from wandb import sdk as wandb_sdk

from src.core import config as _config
from src.core import model as _model
from src.core import train as _train
from src.core import train_state as _train_state
from src.projects.generative.tools import fid
from src.utilities import logging
from src.utilities import visualization

PyTree = jaxtyping.PyTree


# toggle off GPU/TPU for TensorFlow
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
assert not tf.config.experimental.get_visible_devices("GPU")


# ==============================================================================
# Helper Functions
def init_wandb(
    config: _config.ExperimentConfig,
    work_dir: str,
    resume: bool = False,
) -> None:
    r"""Initializes the Weights & Biases logging.

    Args:
        config (ExperimentConfig): The experiment configuration.
        work_dir (str): The working directory for experiment outputs.
        resume (bool, optional): Whether to resume from an existing wandb run.
            Default is `False`.

    Raises:
        FileNotFoundError: If resuming and checkpoint directory does not exist.
        RuntimeError: If wandb run initialization fails.
    """

    ckpt_dir = config.trainer.checkpoint_dir
    if resume:
        if not pathlib.Path(str(ckpt_dir)).exists():
            raise FileNotFoundError(
                f"Checkpoint directory {ckpt_dir} does not exist for resuming."
            )
        run_id = pathlib.Path(str(ckpt_dir), "wandb.txt").read_text().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project=config.project_name,
            dir=work_dir,
            group=config.exp_name,
            job_type="coordinator" if jax.process_index() == 0 else "worker",
        )
    else:
        wandb.init(
            name="_".join([config.exp_name, str(jax.process_index())]),
            config=dataclasses.asdict(config),
            project=config.project_name,
            dir=work_dir,
            group=config.exp_name,
            job_type="coordinator" if jax.process_index() == 0 else "worker",
        )
        _run = wandb.run
        if not isinstance(_run, wandb_sdk.wandb_run.Run):
            raise RuntimeError("Failed to initialize wandb run.")
        pathlib.Path(work_dir, "wandb.txt").write_text(_run.id)


def evaluate(
    rngs: jax.Array,
    model: _model.Model,
    params: PyTree,
    batch: typing.Dict[str, typing.Any],
    fid_metric: fid.FrechetInceptionDistance,
    **kwargs,
) -> _model.StepOutputs:
    r"""Conduct a single evaluation step and compute metrics."""
    local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))

    def _generate(params: PyTree, rng: jax.Array) -> jax.Array:
        r"""Generate samples from the model."""
        outputs = model.forward(
            rngs=jax.random.fold_in(rng, jax.lax.axis_index("batch")),
            params=params,
            deterministic=True,
            batch=batch,
            **kwargs,
        )
        assert isinstance(outputs, _model.StepOutputs)
        assert outputs.output is not None
        img = jnp.clip(outputs.output * 0.5 + 0.5, 0.0, 1.0)
        img = jnp.floor(img * 255.0).astype(jnp.uint8)

        return img

    generate_fn = jax.pmap(_generate, axis_name="batch")
    with tqdm_logging.logging_redirect_tqdm():
        images, count = [], 0
        if jax.process_index() == 0:
            pbar = tqdm.tqdm(total=50_000, unit="sample")
        else:
            pbar = None

        while count < 50_000:
            local_rng, step_rng = jax.random.split(local_rng)
            out = generate_fn(params=params, rng=step_rng)
            out = jnp.reshape(out, (-1,) + out.shape[-3:])
            _slice = min(50_000 - count, out.shape[0])
            images.append(out[:_slice])
            count += _slice
            if pbar is not None:
                pbar.update(_slice)
    if pbar is not None:
        pbar.close()

    outputs = _model.StepOutputs()
    images = jnp.concatenate(images, axis=0)

    fid_score = fid_metric(images=jax.device_get(images[0:50_000]))
    outputs.scalars = {"fid": fid_score}

    img_grid = visualization.make_grid(
        images[0:32],
        n_rows=4,
        n_cols=8,
        padding=2,
    )
    outputs.images = {"sampled images": img_grid}

    return outputs


def training_step(
    rngs: jax.Array,
    model: _model.Model,
    state: _train_state.TrainState,
    batch: typing.Dict[str, typing.Any],
    **kwargs,
) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
    r"""Conduct a single training step and update train state."""
    local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
    local_rng = jax.random.fold_in(local_rng, state.step)

    def loss_fn(params: PyTree) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        loss, outputs = model.compute_loss(
            rngs=local_rng,
            params=params,
            deterministic=False,
            batch=batch,
            **kwargs,
        )
        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (_, outputs), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    return new_state, outputs


def train_and_evaluate(
    exp_config: _config.ExperimentConfig,
    work_dir: str,
) -> int:
    r"""Main entry point for training and evaluate generative models."""

    rng = jax.random.PRNGKey(exp_config.seed)
    log_dir = tf.io.gfile.join(
        work_dir,
        exp_config.project_name,
        exp_config.exp_name,
    )
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
    init_wandb(
        config=exp_config,
        work_dir=log_dir,
        resume=exp_config.trainer.checkpoint_dir is not None,
    )

    # Log the current platform
    logging.rank_zero_info("Running on platform: %s", platform.node())
    logging.rank_zero_info("Running on JAX backend: %s", jax.default_backend())
    logging.rank_zero_info(
        "Running on JAX process: %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    logging.rank_zero_info("Running on JAX devices: %r", jax.devices())

    # Setup Experiment
    if not isinstance(exp_config, _config.ExperimentConfig):
        logging.rank_zero_error(
            (
                "Expect configuration to be of an "
                "`ExperimentConfig` instance, but got %s."
            ),
            type(exp_config),
        )
        return 1
    logging.rank_zero_info("Experiment Configuration:\n%s", exp_config)

    logging.rank_zero_info("Building dataset...")
    rng, data_rng = jax.random.split(rng, num=2)
    p_datamodule = fdl.build(exp_config.data.module)
    # properly handle batch size in distributed setting
    _global_batch_size = exp_config.data.batch_size
    if _global_batch_size % jax.process_count() != 0:
        logging.rank_zero_warning(
            (
                "Global batch size %d is not evenly divisible by process count %d; "
                "per-process batch size will be truncated to %d (effective global "
                "batch size %d)."
            ),
            _global_batch_size,
            jax.process_count(),
            _global_batch_size // jax.process_count(),
            (_global_batch_size // jax.process_count()) * jax.process_count(),
        )
    _per_process_batch_size = int(_global_batch_size // jax.process_count())
    datamodule = p_datamodule(
        batch_size=_per_process_batch_size,
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
    jax.block_until_ready(state)
    logging.rank_zero_info("Building train state... DONE!")

    checkpoint_manager = ocp.CheckpointManager(
        directory=tf.io.gfile.join(log_dir, "checkpoints"),
        item_handlers={
            "state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=exp_config.trainer.max_checkpoints_to_keep,
            create=True,
            enable_async_checkpointing=False,
            cleanup_tmp_directories=True,
        ),
    )
    if exp_config.trainer.checkpoint_every_n_steps is not None:
        checkpoint_every_n_steps = exp_config.trainer.checkpoint_every_n_steps
    else:
        checkpoint_every_n_steps = exp_config.trainer.eval_every_n_steps
    if exp_config.trainer.checkpoint_dir is not None:
        # TODO (juanwu): support loading from custom checkpoint dir
        logging.rank_zero_error("Resuming from checkpoint not implemented.")
        return 1

    p_training_step = functools.partial(training_step, model=model)
    fid_metric = fdl.build(exp_config.metric)
    if not isinstance(fid_metric, fid.FrechetInceptionDistance):
        logging.rank_zero_error(
            (
                "Expect metric to be of an `FrechetInceptionDistance` "
                "instance, but got %s."
            ),
            type(fid_metric),
        )
        return 1
    evaluation_fn = functools.partial(
        evaluate,
        model=model,
        rngs=rng,
        batch=next(datamodule.eval_dataloader()),
        fid_metric=fid_metric,
    )
    if exp_config.mode == "train":
        status = _train.run(
            state=state,
            datamodule=datamodule,
            training_step=p_training_step,
            evaluation_fn=evaluation_fn,
            num_train_steps=exp_config.trainer.num_train_steps,
            checkpoint_manager=checkpoint_manager,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            rng=rng,
            log_every_n_steps=exp_config.trainer.log_every_n_steps,
            eval_every_n_steps=exp_config.trainer.eval_every_n_steps,
        )
    elif exp_config.mode == "evaluate":
        logging.rank_zero_error("Evaluation mode not implemented.")
        status = 1
    else:
        logging.rank_zero_error("Mode %s not implemented.", exp_config.mode)
        status = 1

    # properly shutdown WandB
    wandb.finish(exit_code=status)

    return status
