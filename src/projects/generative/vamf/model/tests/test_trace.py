import sys
import typing

import jax
from jax import numpy as jnp
from jax import random as jrnd
import pytest

from src.projects.generative.meanflow.model import trace


def _linear_u_fn(matrix: jax.Array) -> typing.Callable:
    r"""``u(z, r, t) = z @ matrix.T + (t - r)[..., None]``.

    Jacobian w.r.t. ``z`` equals ``matrix`` (independent of ``z``), so per
    sample ``B = (t - r) * matrix - I``.
    """

    def u_fn(z: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        return z @ matrix.T + (t - r)[..., None]

    return u_fn


def _nonlinear_u_fn(z: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
    r"""Nonlinear ``u_fn`` whose Jacobian depends on ``z``."""
    return jnp.tanh(z) * (1.0 + t[..., None]) + jnp.sin(z) * r[..., None]


def _linear_expected(
    matrix: jax.Array,
    r: jax.Array,
    t: jax.Array,
) -> jax.Array:
    r"""Closed-form ``||(t-r) A - I||_F^2`` per sample."""
    matrix = matrix.astype(jnp.float32)
    delta = (t - r).astype(jnp.float32)
    dim = matrix.shape[-1]
    a_frob = jnp.sum(matrix**2)
    a_trace = jnp.trace(matrix)
    return delta**2 * a_frob - 2.0 * delta * a_trace + dim


def _dense_per_sample_trace(
    u_fn: typing.Callable,
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
) -> jax.Array:
    r"""Reference per-sample ``||(t-r) J - I||_F^2`` via dense Jacobian."""

    def single(zi: jax.Array, ri: jax.Array, ti: jax.Array) -> jax.Array:
        j = jax.jacobian(lambda zz: u_fn(zz[None], ri[None], ti[None])[0])(zi)
        b = (ti - ri) * j - jnp.eye(zi.shape[-1], dtype=j.dtype)
        return jnp.sum(b**2)

    return jax.vmap(single)(z, r, t)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_exact_trace_linear(dtype: typing.Any) -> None:
    """``exact_trace`` matches the closed-form per-sample ``||B||_F^2``."""
    batch, dim = 4, 3
    key = jrnd.PRNGKey(0)
    matrix = (0.1 * jrnd.normal(key, (dim, dim))).astype(dtype)
    z = jrnd.normal(jrnd.fold_in(key, 1), (batch, dim), dtype=dtype)
    r = jrnd.uniform(
        jrnd.fold_in(key, 2), (batch,), dtype=dtype, minval=0.0, maxval=0.4
    )
    t = jrnd.uniform(
        jrnd.fold_in(key, 3), (batch,), dtype=dtype, minval=0.6, maxval=1.0
    )

    out = trace.exact_trace(_linear_u_fn(matrix), z, r, t)
    expected = _linear_expected(matrix, r, t)

    atol = 5e-2 if dtype == jnp.bfloat16 else 1e-4
    assert out.shape == (batch,)
    assert out.dtype == dtype
    assert jnp.allclose(
        out.astype(jnp.float32), expected, atol=atol, rtol=atol
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_exact_trace_matches_dense_jacobian(dtype: typing.Any) -> None:
    """``exact_trace`` matches the dense per-sample reference."""
    batch, dim = 3, 4
    key = jrnd.PRNGKey(1)
    z = jrnd.normal(key, (batch, dim), dtype=dtype)
    r = jrnd.uniform(
        jrnd.fold_in(key, 2), (batch,), dtype=dtype, minval=0.0, maxval=0.4
    )
    t = jrnd.uniform(
        jrnd.fold_in(key, 3), (batch,), dtype=dtype, minval=0.6, maxval=1.0
    )

    out = trace.exact_trace(_nonlinear_u_fn, z, r, t)
    expected = _dense_per_sample_trace(_nonlinear_u_fn, z, r, t)

    atol = 5e-2 if dtype == jnp.bfloat16 else 1e-4
    assert out.shape == (batch,)
    assert jnp.allclose(
        out.astype(jnp.float32),
        expected.astype(jnp.float32),
        atol=atol,
        rtol=atol,
    )


def test_exact_trace_handles_extra_batch_dims() -> None:
    """``exact_trace`` works with multiple leading batch dims."""
    b1, b2, dim = 2, 3, 3
    key = jrnd.PRNGKey(4)
    matrix = 0.1 * jrnd.normal(key, (dim, dim))
    z = jrnd.normal(jrnd.fold_in(key, 1), (b1, b2, dim))
    r = jrnd.uniform(jrnd.fold_in(key, 2), (b1, b2), minval=0.0, maxval=0.4)
    t = jrnd.uniform(jrnd.fold_in(key, 3), (b1, b2), minval=0.6, maxval=1.0)

    out = trace.exact_trace(_linear_u_fn(matrix), z, r, t)

    delta = t - r
    a_frob = jnp.sum(matrix**2)
    a_trace = jnp.trace(matrix)
    expected = delta**2 * a_frob - 2.0 * delta * a_trace + dim

    assert out.shape == (b1, b2)
    assert jnp.allclose(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_hutchinson_trace_converges_to_exact(dtype: typing.Any) -> None:
    """Hutchinson estimate converges to ``exact_trace`` with many probes."""
    batch, dim = 4, 3
    key = jrnd.PRNGKey(2)
    matrix = (0.1 * jrnd.normal(key, (dim, dim))).astype(dtype)
    z = jrnd.normal(jrnd.fold_in(key, 1), (batch, dim), dtype=dtype)
    r = jrnd.uniform(
        jrnd.fold_in(key, 2), (batch,), dtype=dtype, minval=0.0, maxval=0.4
    )
    t = jrnd.uniform(
        jrnd.fold_in(key, 3), (batch,), dtype=dtype, minval=0.6, maxval=1.0
    )
    u_fn = _linear_u_fn(matrix)

    exact = trace.exact_trace(u_fn, z, r, t)
    estimate = trace.hutchinson_trace(
        jrnd.fold_in(key, 4), u_fn, z, r, t, n_probes=1024
    )

    atol = 2e-1 if dtype == jnp.bfloat16 else 1e-1
    assert estimate.shape == exact.shape
    assert estimate.dtype == dtype
    assert jnp.allclose(
        estimate.astype(jnp.float32),
        exact.astype(jnp.float32),
        atol=atol,
        rtol=atol,
    )


def test_hutchinson_trace_is_deterministic_for_fixed_key() -> None:
    """Same key produces identical estimates; different keys differ."""
    batch, dim = 2, 4
    key = jrnd.PRNGKey(3)
    matrix = 0.1 * jrnd.normal(key, (dim, dim))
    z = jrnd.normal(jrnd.fold_in(key, 1), (batch, dim))
    r = jnp.zeros(batch)
    t = jnp.ones(batch)
    u_fn = _linear_u_fn(matrix)

    a_key, b_key = jrnd.split(jrnd.PRNGKey(42))
    a = trace.hutchinson_trace(a_key, u_fn, z, r, t, n_probes=4)
    a_repeat = trace.hutchinson_trace(a_key, u_fn, z, r, t, n_probes=4)
    b = trace.hutchinson_trace(b_key, u_fn, z, r, t, n_probes=4)

    assert jnp.array_equal(a, a_repeat)
    assert not jnp.allclose(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
