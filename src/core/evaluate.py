import collections
import functools
import traceback
import typing

import jax
import jaxtyping
import wandb

from src.core import datamodule as _datamodule
from src.core import model as _model
from src.utilities import logging


def run(
    datamodule: _datamodule.DataModule,
    evaluation_step: typing.Callable[..., _model.StepOutputs],
    params: jaxtyping.PyTree,
    rng: typing.Any,
) -> int:
    """Runs evaluation loop with the given model and datamodule.

    Args:
        datamodule (DataModule): The datamodule providing the evaluation data.
        evaluation_step (Callable): The pmapped evaluation step function.
        params (PyTree): The model parameters to use for evaluation.
        rng (Any): Random key generator.

    Returns:
        Integer status code (0 for success).
    """
    _status = 0

    logging.rank_zero_info("Compiling evaluation step...")
    p_evaluation_step = functools.partial(evaluation_step, rng=rng)
    p_evaluation_step = jax.pmap(p_evaluation_step, axis_name="batch")
    logging.rank_zero_info("Compiling evaluation step...DONE!")

    step: int = 0
    eval_metrics = collections.defaultdict(list)
    logging.rank_zero_info("Evaluating...")
    try:
        for batch in datamodule.eval_dataloader():
            batch = jax.tree_util.tree_map(
                lambda x: (
                    x.reshape(
                        (jax.local_device_count(), -1, *x.shape[1:]),
                    )
                    if hasattr(x, "reshape") and hasattr(x, "shape")
                    else x
                ),
                batch,
            )
            with jax.profiler.StepTraceAnnotation(
                name="evaluation",
                step_num=step,
            ):
                outputs = p_evaluation_step(
                    params=params,
                    batch=batch,
                )
            if not isinstance(outputs, _model.StepOutputs):
                raise ValueError(
                    "FATAL: Output from `evaluation_step` is not "
                    "a `StepOutputs` object."
                )
            step += 1

            # logging at the end of batch
            if outputs.scalars is not None:
                wandb.log(
                    data={
                        f"eval/{k}_step": sum(v) / len(v)
                        for k, v in outputs.scalars.items()
                    },
                    step=step,
                )
            if outputs.images is not None:
                wandb.log(
                    data={
                        f"eval/{k}_step": wandb.Image(v)
                        for k, v in outputs.images.items()
                    },
                    step=step,
                )
            if outputs.histograms is not None:
                wandb.log(
                    data={
                        f"eval/{k}_step": wandb.Histogram(list(v))
                        for k, v in outputs.histograms.items()
                    },
                    step=step,
                )

        # logging at the end of evaluation
        logging.rank_zero_info("Evaluation done.")
        scalar_output = {
            f"eval/{k.replace('_', ' ')}_epoch": sum(v) / len(v)
            for k, v in eval_metrics.items()
        }
        wandb.log(
            data=scalar_output,
            step=step,
        )

    except Exception as e:
        logging.rank_zero_error("Exception occurred during evaluation: %s", e)
        error_trace = traceback.format_exc()
        logging.rank_zero_error("Stack trace:\n%s", error_trace)
        _status = 1
    finally:
        logging.rank_zero_info(
            "Evaluation done. Exit with code %d.",
            _status,
        )

    return _status
