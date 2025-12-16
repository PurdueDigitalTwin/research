import sys

import chex
import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.pipeline import augment


def test_matrix() -> None:
    r"""Tests the matrix constructor."""
    mat = augment.matrix(
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (3, 3))
    test_output = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
