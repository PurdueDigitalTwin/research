import functools
import typing

from absl import logging
import jax


def rank_zero_only(fn: typing.Callable) -> typing.Callable:
    """A decorator to wrap a function for only executing it on rank zero.

    This is useful in distributed training scenarios where you want to avoid
    duplicate logging or printing from multiple processes.

    Args:
        fn (Callable): The function to be decorated.

    Returns:
        The decorated function that only runs on rank zero.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if jax.process_index() == 0:
            return fn(*args, **kwargs)

    return wrapper


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
