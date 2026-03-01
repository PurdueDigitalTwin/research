import os.path as osp
import typing

from absl import logging
import jax
import tensorflow as tf
import wandb
from wandb import sdk as wandb_sdk


def init_wandb(
    config: typing.Dict[str, typing.Any],
    project_name: str,
    experiment_name: str,
    work_dir: str,
    resume: bool = False,
    checkpoint_dir: typing.Optional[str] = None,
) -> None:
    r"""Initializes the Weights and Biases logging.

    Args:
        config (Dict[str, typing.Any]): The experiment configuration to log.
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
        work_dir (str): The working directory for experiment outputs.
        resume (bool, optional): Whether to resume from an existing wandb run.
            Default is `False`.
        checkpoint_dir (Optional[str], optional): Directory of the checkpoint
            file to resume from. Default is `None`.

    Raises:
        FileNotFoundError: If resuming and checkpoint directory does not exist.
        RuntimeError: If wandb run initialization fails.
    """

    if resume and (checkpoint_dir is not None):
        if not tf.io.gfile.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory {checkpoint_dir} does not exist."
            )
        fp = osp.join(checkpoint_dir, "wandb.txt")
        with tf.io.gfile.GFile(fp, "r") as f:
            run_id = f.read().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project=project_name,
            dir=work_dir,
            group=experiment_name,
            job_type="coordinator" if jax.process_index() == 0 else "worker",
        )
    else:
        wandb.init(
            name="_".join([experiment_name, str(jax.process_index())]),
            config=config,
            project=project_name,
            dir=work_dir,
            group=experiment_name,
            job_type="coordinator" if jax.process_index() == 0 else "worker",
        )
        _run = wandb.run
        if not isinstance(_run, wandb_sdk.wandb_run.Run):
            raise RuntimeError("Failed to initialize wandb run.")
        fp = osp.join(work_dir, "wandb.txt")
        with tf.io.gfile.GFile(fp, "w") as f:
            f.write(_run.id)


def rank_zero_log_first_n(
    level: int,
    msg: str,
    n: int,
    *args,
) -> None:
    """Logs a message the first n times only from the process with rank zero.

    Args:
        level (int): The logging level at which to log the message.
        msg (str): The message to be logged.
        n (int): The number of times this should be called before logging.
        *args: Additional positional arguments passed to the logging function.
    """
    if jax.process_index() == 0:
        logging.log_first_n(level=level, msg=msg, n=n, *args)


def rank_zero_log_every_n_seconds(
    level: int,
    msg: str,
    n_seconds: int,
    *args,
) -> None:
    """Logs a message every n seconds only from the process with rank zero.

    Args:
        level (int): The logging level at which to log the message.
        msg (str): The message to be logged.
        n_seconds (int): The number of seconds this should be called before logging.
        *args: Additional positional arguments passed to the logging function.
    """
    if jax.process_index() == 0:
        logging.log_every_n_seconds(
            level=level,
            msg=msg,
            n_seconds=n_seconds,
            *args,
        )


def rank_zero_log_every_n(
    level: int,
    msg: str,
    n: int,
    *args,
) -> None:
    """Logs a message every n times only from the process with rank zero.

    Args:
        level (int): The logging level at which to log the message.
        msg (str): The message to be logged.
        n (int): The number of times this should be called before logging.
        *args: Additional positional arguments passed to the logging function.
    """
    if jax.process_index() == 0:
        logging.log_every_n(level=level, msg=msg, n=n, *args)


def rank_zero_debug(msg: str, *args, **kwargs) -> None:
    """Logs a debug message only from the process with rank zero."""
    if jax.process_index() == 0:
        logging.debug(msg, *args, **kwargs)


def rank_zero_error(msg: str, *args, **kwargs) -> None:
    """Logs an error message only from the process with rank zero."""
    if jax.process_index() == 0:
        logging.error(msg, *args, **kwargs)


def rank_zero_fatal(msg: str, *args, **kwargs) -> None:
    """Logs a fatal message only from the process with rank zero."""
    if jax.process_index() == 0:
        logging.fatal(msg, *args, **kwargs)


def rank_zero_info(msg: str, *args, **kwargs) -> None:
    """Logs an info message only from the process with rank zero."""
    if jax.process_index() == 0:
        logging.info(msg, *args, **kwargs)


def rank_zero_warning(msg: str, *args, **kwargs) -> None:
    """Logs a warning message only from the process with rank zero."""
    if jax.process_index() == 0:
        logging.warning(msg, *args, **kwargs)


def rank_zero_exception(msg: str, *args, **kwargs) -> None:
    """Raises an exception only from the process with rank zero."""
    if jax.process_index() == 0:
        rank_zero_error(msg, *args, **kwargs)
