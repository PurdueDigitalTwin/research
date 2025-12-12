import chex
import jax
from jax import numpy as jnp
from jax import typing as jxt
import numpy as np
from numpy import typing as npt
from scipy import linalg as splin

from src.utilities import logging


def _frechet_distance(
    mu_left: jxt.ArrayLike,
    cov_left: jxt.ArrayLike,
    mu_right: jxt.ArrayLike,
    cov_right: jxt.ArrayLike,
    eps: float = 1e-6,
) -> npt.NDArray[np.float_]:
    r"""Computes the Fréchet Distance between two multivariate Gaussians.

    Args:
        mu_left (ArrayLike): Mean vector of the first Gaussian of shape `(D,)`.
        cov_left (ArrayLike): Covariance matrix of the first Gaussian with
            a shape of `(D, D)`.
        mu_right (ArrayLike): Mean of the second Gaussian of shape `(D,)`.
        cov_right (jax.Array): Covariance matrix of the second Gaussian with
            a shape of `(D, D)`.
        eps (float, optional): Small value to add to the diagonal for numerical
            stability. Default is `0.000001`.

    Returns:
        The Fréchet Distance between the two Gaussians as a scalar array.
    """
    # sanity checks
    chex.assert_equal_shape([mu_left, mu_right])
    chex.assert_equal_shape([cov_left, cov_right])

    mu_left = np.atleast_1d(np.array(mu_left))
    mu_right = np.atleast_1d(np.array(mu_right))
    cov_left = np.atleast_2d(np.array(cov_left))
    cov_right = np.atleast_2d(np.array(cov_right))

    diff = mu_left - mu_right
    covmean, _ = splin.sqrtm(cov_left @ cov_right, disp=False)
    if not np.isfinite(covmean).all():
        logging.rank_zero_warning(
            "Singular product detected during FID calculation. "
            "Adding %s to diagonal of covariance estimations.",
            eps,
        )
        offset = np.eye(cov_left.shape[0]) * eps
        covmean, _ = splin.sqrtm(
            (cov_left + offset) @ (cov_right + offset),
            disp=False,
        )

    if np.iscomplexobj(covmean):
        logging.rank_zero_warning(
            "Complex component detected in covmean during FID calculation. "
            "Taking only the real part.",
        )
        covmean = covmean.real

    trm = np.trace(covmean)
    out = diff @ diff + np.trace(cov_left) + np.trace(cov_right) - 2 * trm

    return out
