import collections
import functools
import os
import traceback
import typing

from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
import jax
import jaxtyping

from src.core import datamodule as _data
from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging

EVAL_STEP_OUTPUT = _model.StepOutputs
TRAIN_STEP_OUTPUT = typing.Tuple[_train_state.TrainState, _model.StepOutputs]


def _shard(tree: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Helper function for `jax.pmap` to shard a pytree onto local devices.

    Args:
        tree (PyTree): The pytree to shard.

    Returns:
        A `PyTree` with an added leading dimension for local devices.
    """
    _shape_prefix = (jax.local_device_count(), -1)
    return jax.tree_util.tree_map(
        lambda x: (
            x.reshape(_shape_prefix + x.shape[1:])
            if hasattr(x, "reshape")
            else x
        ),
        tree=tree,
    )


def run(
    state: _train_state.TrainState,
    datamodule: _data.DataModule,
    training_step: typing.Callable[..., TRAIN_STEP_OUTPUT],
    evaluation_step: typing.Callable[..., EVAL_STEP_OUTPUT],
    num_train_steps: int,
    writer: metric_writers.MetricWriter,
    work_dir: str,
    rng: typing.Any,
    checkpoint_every_n_steps: typing.Optional[int] = None,
    log_every_n_steps: int = 50,
    eval_every_n_steps: int = 1_000,
    profile: bool = False,
) -> int:
    """Runs training and evaluation loop with given model and dataloaders.

    Args:
        datamodule (DataModule): The data module for loading data.
        training_step (Callable): The training step function.
        evaluation_step (Callable): The evaluation step function.
        num_train_steps (int): Number of training steps.
        checkpoint_manager (Checkpoint): The checkpoint manager.
        writer (MetricWriter): The metric writer for logging.
        work_dir (str): The working directory for saving checkpoints and logs.
        rng (Any): The random number generator.
        checkpoint_every_n_steps (Optional[int]): Frequency of checkpointing.
            If `None`, defaults to `eval_every_n_steps`.
        log_every_n_steps (int): Frequency of logging. Default is `50`.
        eval_every_n_steps (int): Frequency of evaluation. Default is `1000`.
        profile (bool): Whether to enable profiling.

    Returns:
        Integer status code.
    """
    _status = 0

    if checkpoint_every_n_steps is None:
        checkpoint_every_n_steps = eval_every_n_steps

    logging.rank_zero_info("Compiling training step function...")
    rng, train_rng = jax.random.split(rng, num=2)
    p_training_step = functools.partial(training_step, rngs=train_rng)
    p_training_step = jax.pmap(p_training_step, axis_name="batch")
    logging.rank_zero_info("Compiling training step function... DONE!")

    logging.rank_zero_info("Compiling evaluation step function...")
    rng, eval_rng = jax.random.split(rng, num=2)
    p_evaluation_step = functools.partial(evaluation_step, rngs=eval_rng)
    p_evaluation_step = jax.pmap(p_evaluation_step, axis_name="batch")
    logging.rank_zero_info("Compiling evaluation step function... DONE!")

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_train_steps,
        writer=writer,
    )
    if jax.process_index() == 0:
        hooks.append(report_progress)
        if profile:
            hooks.append(
                periodic_actions.Profile(
                    logdir=work_dir,
                    num_profile_steps=5,
                )
            )
    step = state.step
    state = jax_utils.replicate(state)
    logging.rank_zero_info("Training...")
    with metric_writers.ensure_flushes(writer):
        try:
            train_metrics = collections.defaultdict(list)
            while True:
                for batch in datamodule.train_dataloader():
                    # evaluation and sanity check running
                    if (
                        step % eval_every_n_steps == 0
                        or step == num_train_steps
                    ):
                        logging.rank_zero_info("Running evaluation...")
                        eval_metrics = collections.defaultdict(list)
                        outputs = None
                        for batch in datamodule.eval_dataloader():
                            batch = _shard(batch)
                            outputs = p_evaluation_step(
                                params=state.params,
                                batch=batch,
                            )
                            if not isinstance(outputs, _model.StepOutputs):
                                raise RuntimeError(
                                    "FATAL: Output from `evaluation_step` is "
                                    "not a `StepOutputs` object."
                                )
                            if outputs.scalars is not None:
                                for k, v in outputs.scalars.items():
                                    eval_metrics[k].append(
                                        jax.device_get(v).mean()
                                    )
                        logging.rank_zero_info("Evaluation done.")

                        if isinstance(outputs, _model.StepOutputs):
                            writer.write_scalars(
                                step=step,
                                scalars={
                                    f"eval/{k}": sum(v) / len(v)
                                    for k, v in eval_metrics.items()
                                },
                            )
                            if outputs.images is not None:
                                writer.write_images(
                                    step=step,
                                    images={
                                        f"eval/{k}": v
                                        for k, v in outputs.images.items()
                                    },
                                )

                    batch = _shard(batch)
                    with jax.profiler.StepTraceAnnotation(
                        name="train",
                        step_num=step,
                    ):
                        state, outputs = p_training_step(
                            state=state,
                            batch=batch,
                        )
                    if not isinstance(outputs, _model.StepOutputs):
                        raise RuntimeError(
                            "FATAL: Output from `training_step` is not "
                            "a `StepOutputs` object."
                        )
                    if outputs.scalars is not None:
                        for k, v in outputs.scalars.items():
                            train_metrics[k].append(jax.device_get(v).mean())
                    for hook in hooks:
                        hook(step)
                    if step % log_every_n_steps == 0:
                        if outputs.scalars is not None:
                            writer.write_scalars(
                                step=step,
                                scalars={
                                    f"train/{k}_step": sum(v) / len(v)
                                    for k, v in outputs.scalars.items()
                                },
                            )
                        if outputs.images is not None:
                            writer.write_images(
                                step=step,
                                images=outputs.images,
                            )
                    step += 1

                    # checkpointing
                    if step % checkpoint_every_n_steps == 0:
                        logging.rank_zero_info("Checkpointing...")
                        if jax.process_index() == 0:
                            with report_progress.timed("checkpoint"):
                                filepath = checkpoints.save_checkpoint(
                                    ckpt_dir=os.path.join(
                                        work_dir,
                                        "checkpoints",
                                    ),
                                    target=jax_utils.unreplicate(state),
                                    keep=3,
                                    overwrite=True,
                                    prefix="ckpt-",
                                    step=step,
                                )
                            logging.rank_zero_info(
                                "Checkpoint saved to %s",
                                filepath,
                            )

                # logging on the end of epoch
                scalar_output = {
                    f"train/{k}_epoch": sum(v) / len(v)
                    for k, v in train_metrics.items()
                }
                writer.write_scalars(
                    step=step,
                    scalars=scalar_output,
                )

        except Exception as e:
            logging.rank_zero_error(
                "Exception occurred during training: %s", e
            )
            error_trace = traceback.format_exc()
            logging.rank_zero_error(error_trace)
            _status = 1
        finally:
            state = jax_utils.unreplicate(state)
            logging.rank_zero_info(
                "Training finished. Final step: %d. Exit with code %d.",
                state.step,
                _status,
            )

    return _status
