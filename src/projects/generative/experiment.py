import collections
import dataclasses
import functools
import platform
import traceback
import typing

import fiddle as fdl
from flax import jax_utils
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
import optax
from orbax import checkpoint as ocp
from orbax.checkpoint import utils as ocp_utils
import tensorflow as tf
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging
import wandb

from src.core import config as _config
from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.generative.tools import fid
from src.utilities import logging
from src.utilities import training
from src.utilities import visualization

# Type aliases
PyTree = jaxtyping.PyTree


# Toggle off GPU/TPU for TensorFlow
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")


# ==============================================================================
# Helper Functions
def _log_step_outputs(
    outputs: _model.StepOutputs,
    prefix: str,
    step: int,
    suffix: str = "",
) -> None:
    """Log a StepOutputs object to W&B under ``{prefix}/{key}{suffix}``."""
    if outputs.scalars is not None:
        wandb.log(
            {f"{prefix}/{k}{suffix}": v for k, v in outputs.scalars.items()},
            step=step,
        )
    if outputs.images is not None:
        wandb.log(
            {
                f"{prefix}/{k}": wandb.Image(np.asarray(v))
                for k, v in outputs.images.items()
            },
            step=step,
        )
    if outputs.histograms is not None:
        wandb.log(
            {
                f"{prefix}/{k}": wandb.Histogram(list(v))
                for k, v in outputs.histograms.items()
            },
            step=step,
        )


def evaluate(
    params: PyTree,
    rngs: jax.Array,
    model: _model.Model,
    batch: typing.Dict[str, typing.Any],
    fid_metric: fid.FrechetInceptionDistance,
    **kwargs,
) -> _model.StepOutputs:
    r"""Conduct a single evaluation step and compute metrics.

    Args:
        params (PyTree): The model parameters.
        rngs (jax.Array): Random key for sampling.
        model (Model): The generative model to be evaluated.
        batch (Dict[str, Any]): An example batch of evaluation data.
        fid_metric (FrechetInceptionDistance): The FID metric instance.

    Returns:
        The evaluation outputs including metrics and generated images.
    """

    def _generate(
        params: PyTree,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        step_rngs: jax.Array,
    ) -> jax.Array:
        local_rng = jax.random.fold_in(step_rngs, jax.lax.axis_index("batch"))
        outputs = model.forward(
            rngs=local_rng,
            params=params,
            shape=shape,
            deterministic=True,
            batch=batch,
            **kwargs,
        )
        assert isinstance(outputs, _model.StepOutputs)
        assert outputs.output is not None
        img = jnp.clip(outputs.output * 0.5 + 0.5, 0.0, 1.0)
        img = jnp.floor(img * 255.0).astype(jnp.uint8)

        return img

    shape = batch["image"].shape
    p_generate = functools.partial(_generate, shape=shape)
    p_generate = jax.pmap(p_generate, axis_name="batch")
    with tqdm_logging.logging_redirect_tqdm():
        images, count = [], 0
        if jax.process_index() == 0:
            pbar = tqdm.tqdm(
                total=50_000,
                desc="Generating samples for FID evaluation...",
                unit="sample",
            )
        else:
            pbar = None

        while count < 50_000:
            step_rng = jax.random.fold_in(rngs, count)
            step_rng = jax.random.split(step_rng, jax.local_device_count())
            out = p_generate(params=params, step_rngs=step_rng)
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

    if jax.process_index() == 0:
        # NOTE: only compute FID metric on process 0
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


def train_and_evaluate(
    exp_config: _config.ExperimentConfig,
    work_dir: str,
) -> int:
    r"""Main entry point for training and evaluate generative models."""
    _status = 0

    rng = jax.random.PRNGKey(exp_config.seed)
    log_dir = tf.io.gfile.join(
        work_dir,
        exp_config.project_name,
        exp_config.exp_name,
    )
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
    checkpoint_dir = exp_config.trainer.checkpoint_dir
    logging.init_wandb(
        config=dataclasses.asdict(exp_config),
        project_name=str(exp_config.project_name),
        experiment_name=str(exp_config.exp_name),
        work_dir=log_dir,
        resume=checkpoint_dir is not None,
        checkpoint_dir=checkpoint_dir,
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
    _local_batch_size = exp_config.data.batch_size * jax.local_device_count()
    datamodule = p_datamodule(
        batch_size=_local_batch_size,
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
    params, _ = model.init(batch=None, rngs=init_rng)  # NOTE: use dummy batch
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

    if exp_config.mode == "train":
        logging.rank_zero_info("Compiling training step functions...")
        rng, train_key, eval_key = jax.random.split(rng, num=3)
        p_train_step = functools.partial(model.training_step, rngs=train_key)
        p_train_step = jax.pmap(p_train_step, axis_name="batch")
        evaluation_fn = functools.partial(
            evaluate,
            model=model,
            rngs=eval_key,
            batch=next(datamodule.eval_dataloader()),
            fid_metric=fid_metric,
        )

        state: _train_state.TrainState = jax_utils.replicate(state)
        step = int(jax.device_get(state.step)[0])
        if jax.process_index() == 0:
            pbar = tqdm.tqdm(
                initial=step,
                total=exp_config.trainer.num_train_steps,
                desc="Training",
                leave=False,
                position=0,
                unit="step",
            )
        else:
            pbar = None

        try:
            while step < exp_config.trainer.num_train_steps:
                train_metrics = collections.defaultdict(list)
                for train_batch in datamodule.train_dataloader():
                    if (
                        step % exp_config.trainer.eval_every_n_steps == 0
                        or step == exp_config.trainer.num_train_steps
                    ):
                        logging.rank_zero_info("Running evaluation...")
                        outputs = evaluation_fn(params=state.ema_params)
                        logging.rank_zero_info("Evaluation done.")
                        _log_step_outputs(
                            outputs=outputs,
                            prefix="eval",
                            step=step,
                        )
                        if outputs.scalars is not None and pbar is not None:
                            scalar_str = ", ".join(
                                f"{k}={jax.device_get(v).mean():.4f}"
                                for k, v in outputs.scalars.items()
                            )
                            pbar.write(f"[eval end]: {scalar_str}")

                    train_batch = training.shard(train_batch)
                    with jax.profiler.StepTraceAnnotation(
                        name="train",
                        step_num=step,
                    ):
                        state, outputs = p_train_step(
                            state=state,
                            batch=train_batch,
                        )
                    if not isinstance(outputs, _model.StepOutputs):
                        raise RuntimeError(
                            "FATAL: Output from `training_step` is not "
                            "a `StepOutputs` object."
                        )
                    if outputs.scalars is not None:
                        for k, v in outputs.scalars.items():
                            train_metrics[k].append(jax.device_get(v).mean())

                    if step % exp_config.trainer.log_every_n_steps == 0:
                        _log_step_outputs(
                            outputs=outputs,
                            prefix="train",
                            step=step,
                            suffix="_step",
                        )

                    # update step and progress bar
                    step += 1
                    if pbar is not None:
                        pbar.update(1)

                    # checkpointing
                    if (
                        step % checkpoint_every_n_steps == 0
                        or step >= exp_config.trainer.num_train_steps
                    ):
                        logging.rank_zero_info("Checkpointing...")
                        with jax.profiler.StepTraceAnnotation(
                            name="checkpoint",
                            step_num=step,
                        ):
                            state_to_save = jax.tree_util.tree_map(
                                ocp_utils.fully_replicated_host_local_array_to_global_array,
                                state,
                            )

                            if hasattr(state_to_save, "ema_params"):
                                params = state_to_save.ema_params
                                state_to_save = dataclasses.replace(
                                    state_to_save,
                                    ema_params={},
                                )
                            else:
                                params = state_to_save.params
                                state_to_save = dataclasses.replace(
                                    state_to_save,
                                    params={},
                                )
                            checkpoint_manager.save(
                                step=state_to_save.step,
                                items={
                                    "state": state_to_save,
                                    "params": params,
                                },
                            )

                    # break outer loop if reach max steps
                    if step >= exp_config.trainer.num_train_steps:
                        break

                # logging on the end of epoch
                scalar_output = {
                    f"train/{k}_epoch": sum(v) / len(v)
                    for k, v in train_metrics.items()
                }
                wandb.log(data=scalar_output, step=step)
                scalar_str = ", ".join(
                    [
                        f"{k}={sum(v) / len(v):.4f}"
                        for k, v in train_metrics.items()
                    ]
                )
                if pbar is not None:
                    pbar.write(f"[epoch end at step={step:d}]: {scalar_str:s}")

        except Exception as e:
            logging.rank_zero_error(
                "Exception encountered during training: %s", e
            )
            error_trace = traceback.format_exc()
            logging.rank_zero_error(error_trace)
            _status = 1
        finally:
            state = jax_utils.unreplicate(state)
            checkpoint_manager.wait_until_finished()
            if pbar is not None:
                pbar.close()
            logging.rank_zero_info(
                "Training finished. Final step: %d. Exit with code %d.",
                state.step,
                _status,
            )

    elif exp_config.mode == "evaluate":
        logging.rank_zero_error("Evaluation mode not implemented.")
        _status = 1
    else:
        logging.rank_zero_error("Mode %s not implemented.", exp_config.mode)
        _status = 1

    # properly shutdown WandB
    wandb.finish(exit_code=_status)

    return _status
