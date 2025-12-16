import sys

import chex
import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.pipeline import augment


def test_matrix() -> None:
    r"""Tests the matrix constructor."""
    mat = augment.matrix(
        jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (3, 3))
    test_output = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)


def test_translate_2d() -> None:
    r"""Tests the 2D translation transformation."""
    tx, ty = 5.0, -3.0
    mat = augment.translate2d(tx, ty)
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (3, 3))
    test_output = jnp.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)
    inv_mat = augment.translate2d_inv(tx=tx, ty=ty)
    chex.assert_trees_all_close(inv_mat @ mat, jnp.eye(3, dtype=jnp.float32))


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
