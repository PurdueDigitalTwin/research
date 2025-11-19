import collections
import functools
import typing

from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import jax
import jaxtyping

from src.core import datamodule as _datamodule
from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging


def _create_step_fn(
    model: _model.Model,
    rng: typing.Any,
) -> typing.Tuple[jax.Array, typing.Callable, typing.Callable]:
    """Creates the step functions for training and evaluation."""
    # create training step function
    rng, train_rng = jax.random.split(rng, num=2)
    p_training_step = functools.partial(model.training_step, rngs=train_rng)
    p_training_step = jax.pmap(p_training_step, axis_name="batch")

    rng, eval_rng = jax.random.split(rng, num=2)
    p_evaluation_step = functools.partial(model.evaluation_step, rngs=eval_rng)
    p_evaluation_step = jax.pmap(p_evaluation_step, axis_name="batch")

    return rng, p_training_step, p_evaluation_step


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
    model: _model.Model,
    state: _train_state.TrainState,
    datamodule: _datamodule.DataModule,
    num_train_steps: int,
    checkpoint_manager: checkpoint.Checkpoint,
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
        model (Model): The model to run.
        train_dataloader (Any): The training dataloaders.
        eval_dataloader (Any): The evaluation dataloaders.
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
    logging.rank_zero_debug(f"running {model.__class__.__name__} fit stage...")

    if checkpoint_every_n_steps is None:
        checkpoint_every_n_steps = eval_every_n_steps
    rng, p_training_step, p_evaluation_step = _create_step_fn(
        model=model,
        rng=rng,
    )

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
    step, epoch = state.step, 0
    state = jax_utils.replicate(state)
    logging.rank_zero_info("Training...")
    with metric_writers.ensure_flushes(writer):
        try:
            train_metrics = collections.defaultdict(list)
            while True:
                for batch in datamodule.train_dataloader():
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
                    step += 1
                    for hook in hooks:
                        hook(step)
                    if step % log_every_n_steps == 0:
                        if outputs.scalars is not None:
                            scalar_output = {
                                f"train/{k.replace('_', ' ')}_step": sum(v)
                                / len(v)
                                for k, v in outputs.scalars.items()
                            }
                            writer.write_scalars(
                                step=step,
                                scalars=scalar_output,
                            )
                        if outputs.images is not None:
                            writer.write_images(
                                step=step,
                                images=outputs.images,
                            )

                    # evaluation
                    if (
                        step % eval_every_n_steps == 0
                        or step == num_train_steps
                    ):
                        logging.rank_zero_info("Running evaluation...")
                        eval_metrics = collections.defaultdict(list)
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
                        writer.write_scalars(
                            step=step,
                            scalars={
                                f"eval/{k.replace('_', ' ')}": sum(v) / len(v)
                                for k, v in eval_metrics.items()
                            },
                        )
                        if outputs.images is not None:
                            writer.write_images(
                                step=step,
                                images=outputs.images,
                            )

                    # checkpointing
                    if step % checkpoint_every_n_steps == 0:
                        logging.rank_zero_info("Checkpointing...")
                        # TODO (juanwulu): resolve the error (no __enter__)
                        with report_progress.timed("checkpoint"):
                            filepath = checkpoint_manager.save(
                                state=jax_utils.unreplicate(state)
                            )
                        logging.rank_zero_info(
                            "Checkpoint saved to %s",
                            filepath,
                        )

                # logging on the end of epoch
                logging.rank_zero_info("Epoch %d done.", epoch)
                scalar_output = {
                    f"train/{k.replace('_', ' ')}_epoch": sum(v) / len(v)
                    for k, v in train_metrics.items()
                }
                writer.write_scalars(
                    step=epoch,
                    scalars=scalar_output,
                )
                epoch += 1

        except Exception as e:
            logging.rank_zero_error(
                "Exception occurred during training: %s", e
            )
            _status = 1
        finally:
            state = jax_utils.unreplicate(state)
            logging.rank_zero_info(
                "Training finished. Final step: %d. Exit with code %d.",
                state.step,
                _status,
            )

    return _status
