import collections
import functools
import traceback
import typing

from flax import jax_utils
import jax
import jaxtyping
import numpy as np
from orbax import checkpoint as ocp
from tqdm import auto as tqdm
import wandb

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
    num_train_steps: int,
    checkpoint_manager: ocp.CheckpointManager,
    checkpoint_every_n_steps: int,
    rng: typing.Any,
    evaluation_step: typing.Optional[
        typing.Callable[..., EVAL_STEP_OUTPUT]
    ] = None,
    evaluation_fn: typing.Optional[
        typing.Callable[..., EVAL_STEP_OUTPUT]
    ] = None,
    log_every_n_steps: int = 50,
    eval_every_n_steps: int = 1_000,
    profile: bool = False,
) -> int:
    """Runs training and evaluation loop with given model and dataloaders.

    Args:
        datamodule (DataModule): The data module for loading data.
        training_step (Callable): The training step function.
        num_train_steps (int): Number of training steps.
        checkpoint_manager (CheckpointManager): The checkpoint manager.
        checkpoint_every_n_steps (int): Frequency of checkpointing.
        rng (Any): The random number generator.
        evaluation_step (Optional[Callable]): Optional evaluation step function.
        evaluation_fn (Optional[Callable]): Optional evaluation function.
        log_every_n_steps (int): Frequency of logging. Default is `50`.
        eval_every_n_steps (int): Frequency of evaluation. Default is `1000`.
        profile (bool): Whether to enable profiling.

    Returns:
        Integer status code.
    """
    _status = 0

    logging.rank_zero_info("Compiling training step function...")
    rng, train_rng = jax.random.split(rng, num=2)
    p_training_step = functools.partial(training_step, rngs=train_rng)
    p_training_step = jax.pmap(p_training_step, axis_name="batch")
    logging.rank_zero_info("Compiling training step function... DONE!")

    if evaluation_step is not None:
        logging.rank_zero_info("Compiling evaluation step function...")
        rng, eval_rng = jax.random.split(rng, num=2)
        p_evaluation_step = functools.partial(evaluation_step, rngs=eval_rng)
        p_evaluation_step = jax.pmap(p_evaluation_step, axis_name="batch")
        logging.rank_zero_info("Compiling evaluation step function... DONE!")
    else:
        p_evaluation_step = None

    step = state.step
    pbar = tqdm.tqdm(
        initial=step,
        total=num_train_steps,
        desc="Training",
        leave=False,
        position=0,
        unit="step",
    )
    state = jax_utils.replicate(state)
    logging.rank_zero_info("Training...")
    try:
        while step < num_train_steps:
            train_metrics = collections.defaultdict(list)
            for train_batch in datamodule.train_dataloader():
                # evaluation and sanity check running
                if step % eval_every_n_steps == 0 or step == num_train_steps:
                    logging.rank_zero_info("Running evaluation...")
                    eval_metrics = collections.defaultdict(list)
                    if p_evaluation_step is not None:
                        outputs = None
                        for eval_batch in datamodule.eval_dataloader():
                            eval_batch = _shard(eval_batch)
                            outputs = p_evaluation_step(
                                params=state.ema_params,
                                batch=eval_batch,
                            )
                            if not isinstance(outputs, _model.StepOutputs):
                                raise RuntimeError(
                                    "FATAL: Output from `evaluation_step` "
                                    "is not a `StepOutputs` object."
                                )
                            if outputs.scalars is not None:
                                for k, v in outputs.scalars.items():
                                    eval_metrics[k].append(
                                        jax.device_get(v).mean()
                                    )
                    elif evaluation_fn is not None:
                        outputs = evaluation_fn(params=state.ema_params)
                        if outputs.scalars is not None:
                            for k, v in outputs.scalars.items():
                                eval_metrics[k] = [jax.device_get(v).mean()]
                    else:
                        logging.rank_zero_error(
                            "No evaluation step or function provided. "
                            "Skipping evaluation..."
                        )
                        outputs = None
                    logging.rank_zero_info("Evaluation done.")

                    if isinstance(outputs, _model.StepOutputs):
                        wandb.log(
                            data={
                                f"eval/{k}": sum(v) / len(v)
                                for k, v in eval_metrics.items()
                            },
                            step=step,
                        )
                        if outputs.images is not None:
                            wandb.log(
                                data={
                                    f"eval/{k}": wandb.Image(np.asarray(v))
                                    for k, v in outputs.images.items()
                                },
                                step=step,
                            )
                        if outputs.histograms is not None:
                            wandb.log(
                                data={
                                    f"eval/{k}": wandb.Histogram(list(v))
                                    for k, v in outputs.histograms.items()
                                },
                                step=step,
                            )

                train_batch = _shard(train_batch)
                with jax.profiler.StepTraceAnnotation(
                    name="train",
                    step_num=step,
                ):
                    state, outputs = p_training_step(
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

                if step % log_every_n_steps == 0:
                    if outputs.scalars is not None:
                        wandb.log(
                            data={
                                f"train/{k}_step": sum(v) / len(v)
                                for k, v in outputs.scalars.items()
                            },
                            step=step,
                        )
                    if outputs.images is not None:
                        wandb.log(
                            data={
                                f"train/{k}": wandb.Image(np.asarray(v))
                                for k, v in outputs.images.items()
                            },
                        )
                    if outputs.histograms is not None:
                        wandb.log(
                            data={
                                f"train/{k}": wandb.Histogram(list(v))
                                for k, v in outputs.histograms.items()
                            },
                            step=step,
                        )
                step += 1
                pbar.update(1)

                # checkpointing
                if step % checkpoint_every_n_steps == 0:
                    logging.rank_zero_info("Checkpointing...")
                    if jax.process_index() == 0:
                        with jax.profiler.StepTraceAnnotation(
                            name="checkpoint",
                            step_num=step,
                        ):
                            state_to_save = jax_utils.unreplicate(state)
                            checkpoint_manager.save(
                                step=state_to_save.step,
                                items={
                                    "state": state_to_save,
                                    "params": state_to_save.ema_params,
                                },
                            )

            # logging on the end of epoch
            scalar_output = {
                f"train/{k}_epoch": sum(v) / len(v)
                for k, v in train_metrics.items()
            }
            wandb.log(data=scalar_output, step=step)

            # break outer loop if reach max steps
            if step >= num_train_steps:
                break

    except Exception as e:
        logging.rank_zero_error("Exception occurred during training: %s", e)
        error_trace = traceback.format_exc()
        logging.rank_zero_error(error_trace)
        _status = 1

    finally:
        state = jax_utils.unreplicate(state)
        checkpoint_manager.wait_until_finished()
        logging.rank_zero_info(
            "Training finished. Final step: %d. Exit with code %d.",
            state.step,
            _status,
        )

    return _status
