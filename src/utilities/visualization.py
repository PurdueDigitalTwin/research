import typing

import numpy as np
import numpy.typing as npt


def make_grid(
    images: typing.Union[typing.Any, npt.NDArray],
    n_rows: int = 8,
    padding: int = 2,
) -> npt.NDArray:
    r"""Convert a batch of images into a grid for visualization.

    Args:
        images (Any | NDArray): Batch of images with shape `(B, H, W, C)`.
        n_rows (int, optional): Number of rows in grid. Default is :math:`8`.
        padding (int, optional): Number of pixels between pair of images.
            Default is :math:`2`.

    Returns:
        The array containing a grid of input images.
    """
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(
            "Input images must be a 4-D numpy array with shape (B, H, W, C). "
            f"But got {images}."
        )
    bz, h, w, c = images.shape
    n_cols = int(np.ceil(bz / n_rows))
    shape = (
        h * n_rows + padding * (n_rows - 1),
        w * n_cols + padding * (n_cols - 1),
        c,
    )
    out = np.zeros(shape, dtype=images.dtype)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        top = row * (h + padding)
        left = col * (w + padding)
        out[top : top + h, left : left + w] = img

    return out
