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


def test_translate() -> None:
    r"""Tests the 2D and 3D translation transformation."""
    # test 2D translation
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

    # test 3D translation
    tx, ty, tz = 2.0, -4.0, 7.0
    mat = augment.translate3d(tx, ty, tz)
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (4, 4))
    test_output = jnp.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)
    inv_mat = augment.translate3d(tx=-tx, ty=-ty, tz=-tz)
    chex.assert_trees_all_close(inv_mat @ mat, jnp.eye(4, dtype=jnp.float32))


def test_scale() -> None:
    r"""Tests the 2D and 3D scaling transformation."""
    sx, sy, sz = 2.0, 3.0, 4.0
    # test 2D scaling
    mat = augment.scale2d(sx, sy)
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (3, 3))
    test_output = jnp.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)
    inv_mat = augment.scale2d_inv(sx=sx, sy=sy)
    chex.assert_trees_all_close(inv_mat @ mat, jnp.eye(3, dtype=jnp.float32))

    # test 3D scaling
    mat = augment.scale3d(sx, sy, sz)
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (4, 4))
    test_output = jnp.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)
    inv_mat = augment.scale3d(sx=1.0 / sx, sy=1.0 / sy, sz=1.0 / sz)
    chex.assert_trees_all_close(inv_mat @ mat, jnp.eye(4, dtype=jnp.float32))


def test_rotate_2d() -> None:
    r"""Tests the 2D rotation transformation."""
    # test 2D rotation
    theta = jnp.pi / 4  # 45 degrees
    mat = augment.rotate2d(theta)
    assert isinstance(mat, jax.Array)
    chex.assert_shape(mat, (3, 3))
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    test_output = jnp.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    chex.assert_trees_all_close(mat, test_output)
    inv_mat = augment.rotate2d_inv(theta)
    chex.assert_trees_all_close(inv_mat @ mat, jnp.eye(3, dtype=jnp.float32))


def test_augmentor() -> None:
    r"""Tests the EDM augmentation pipeline."""
    augmentor = augment.EDMAugmentor(
        p=0.12,
        xflip=1e8,
        yflip=0,
        scale=1,
        rotate_frac=0,
        aniso=1,
        translate_frac=1,
    )
    test_input = jnp.ones((2, 32, 32, 3), dtype=jnp.float32)
    test_output, test_labels = augmentor.apply(
        variables={},
        images=test_input,
        rngs=jax.random.PRNGKey(0),
    )
    assert isinstance(test_output, jax.Array)
    chex.assert_shape(test_output, (2, 32, 32, 3))
    assert isinstance(test_labels, jax.Array)
    chex.assert_shape(test_labels, (2, 1))


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
