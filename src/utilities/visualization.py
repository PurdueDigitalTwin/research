import typing

import jax
from jax import numpy as jnp

from src.utilities import logging


def make_grid(
    images: jax.Array,
    n_rows: int = 8,
    n_cols: int = 8,
    padding: int = 2,
) -> jax.Array:
    r"""Convert a batch of images into a grid for visualization.

    Args:
        images (jax.Array): Batch of images with shape `(B, H, W, C)`.
        n_rows (int): Number of rows in grid. Default is :math:`8`.
        n_cols (int): Number of columns in grid. Default is :math:`8`.
        padding (int, optional): Number of pixels between pair of images.
            Default is :math:`2`.

    Returns:
        The array containing a grid of input images.
    """
    images = jnp.reshape(images, (-1,) + images.shape[-3:])
    _, h, w, c = images.shape
    shape = (
        h * n_rows + padding * (n_rows - 1),
        w * n_cols + padding * (n_cols - 1),
        c,
    )
    out = jnp.zeros(shape, dtype=images.dtype)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        top = row * (h + padding)
        left = col * (w + padding)
        out = out.at[top : top + h, left : left + w].set(img)

        if idx + 1 >= n_rows * n_cols:
            logging.rank_zero_warning(
                "Number of images exceed grid capacity; "
                + "only the first %d images are used.",
                n_rows * n_cols,
            )
            break

    return out
