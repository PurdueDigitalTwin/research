import functools
import typing

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
        return None

    return wrapper
