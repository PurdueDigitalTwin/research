import sys

import numpy as np
import pytest

from src.projects.generative.tools import fid


def test_frechet_distance() -> None:
    r"""Test frechet distance computation."""
    dim: int = 128

    # test identity
    mu = np.random.randn(dim)
    cov = np.diag(np.random.rand(dim) + 0.1)
    dist = fid._frechet_distance(mu, cov, mu, cov)
    np.testing.assert_allclose(dist, 0.0, atol=1e-5)

    # test mean shift only
    mu2 = mu + np.random.randn(dim) * 0.5
    dist = fid._frechet_distance(mu, cov, mu2, cov)
    expected_dist = np.sum((mu - mu2) ** 2)
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)

    # test covariance shift only
    cov1 = np.eye(dim)
    cov2 = 4 * np.eye(dim)
    dist = fid._frechet_distance(mu, cov1, mu, cov2)
    expected_dist = float(dim)
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)

    # test singular covariance
    cov1 = np.zeros((dim, dim))
    cov2 = np.eye(dim)
    dist = fid._frechet_distance(mu, cov1, mu, cov2)
    expected_dist = float(dim)
    assert np.all(np.isfinite(dist))
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)

    # test imaginary component
    cov1 = np.array([[1.0, 0.99], [0.99, 1.0]])
    cov2 = np.array([[1.0, 0.99], [0.99, 1.0]])
    dist = fid._frechet_distance(mu[:2], cov1, mu[:2], cov2)
    assert np.all(np.isreal(dist))
    np.testing.assert_allclose(dist, 0.0, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
