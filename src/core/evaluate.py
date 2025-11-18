import collections
import functools
import typing

from clu import metric_writers
from clu import periodic_actions
import jax
import jaxtyping

from src.core import data as _data
from src.core import model as _model
from src.utilities import logging


def run(
    model: _model.Model,
    datamodule: _data.DataModule,
    params: jaxtyping.PyTree,
    writer: metric_writers.MetricWriter,
    work_dir: str,
    rng: typing.Any,
    profile: bool = False,
) -> int:
    """Runs evaluation loop with the given model and datamodule.

    Args:
        model (Model): The model to evaluate.
        datamodule (DataModule): The datamodule providing the evaluation data.
        params (PyTree): The model parameters to use for evaluation.
        writer (MetricWriter): The metric writer for logging evaluation metrics.
        work_dir (str): The working directory for saving outputs.
        rng (Any): Random key generator.
        profile (bool): Whether to enable profiling.

    Returns:
        Integer status code (0 for success).
    """
    _status = 0
    logging.rank_zero_debug(f"running {model.__class__.__name__} eval...")

    eval_rng = jax.random.fold_in(rng, jax.process_index())
    p_evaluation_step = functools.partial(model.evaluation_step, rng=eval_rng)
    p_evaluation_step = jax.pmap(p_evaluation_step, axis_name="batch")

    hooks = []
    if jax.process_index() == 0:
        if profile:
            hooks.append(
                periodic_actions.Profile(
                    logdir=work_dir,
                    num_profile_steps=5,
                )
            )

    step: int = 0
    eval_metrics = collections.defaultdict(list)
    logging.rank_zero_info("Evaluating...")
    with metric_writers.ensure_flushes(writer):
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
                    name="train",
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
                    _scalars = {}
                    for k, v in outputs.scalars.items():
                        eval_metrics[k].append(jax.device_get(v).mean())
                        _scalars[f"eval/{k.replace('_', ' ')}"] = (
                            jax.device_get(v).mean()
                        )
                    writer.write_scalars(
                        step=step + 1,
                        scalars=_scalars,
                    )
                if outputs.images is not None:
                    writer.write_images(
                        step=step + 1,
                        images=outputs.images,
                    )

            # logging at the end of evaluation
            logging.rank_zero_info("Evaluation done.")
            scalar_output = {
                f"eval/{k.replace('_', ' ')}": sum(v) / len(v)
                for k, v in eval_metrics.items()
            }
            writer.write_scalars(
                step=step,
                scalars=scalar_output,
            )
        except Exception as e:
            logging.rank_zero_error(
                "Exception occurred during evaluation: %s", e
            )
            _status = 1
        finally:
            logging.rank_zero_info(
                "Evaluation done. Exit with code %d.",
                _status,
            )
    return _status
