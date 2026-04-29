import typing

import jax
from jax import numpy as jnp
from jax import random as jrnd


def _gap_broadcast(gap: jax.Array, ndim: int) -> jax.Array:
    """Reshape a per-sample scalar to broadcast against ``z`` of rank ``ndim``."""
    return jnp.reshape(gap, (-1,) + (1,) * (ndim - 1))


def _data_axes(ndim: int) -> typing.Tuple[int, ...]:
    """All non-batch axes for a tensor of rank ``ndim`` (axis 0 is batch)."""
    return tuple(range(1, ndim))


def exact_trace(
    u_fn: typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
) -> jax.Array:
    r"""Evaluate the exact trace term :math:`\mathrm{tr}(BB^{\top})`,
    where :math:`B = (t-r) J - I` and :math:`J = \partial_z u`.

    Iterates over each scalar coordinate of ``z`` (treating axis 0 as the
    batch axis) and accumulates :math:`\| B e_i \|^2`. Tractable only for
    very low total data dimension; use ``hutchinson_trace`` for images.

    Args:
        u_fn: Velocity field ``u_fn(z, r, t)``; output has the same shape as ``z``.
        z: Noisy input of shape ``(B,) + data_dims``.
        r: Start timestamps of shape ``(B,)``.
        t: End timestamps of shape ``(B,)``.

    Returns:
        Per-sample trace estimate of shape ``(B,)``.
    """
    flat_dim = 1
    for s in z.shape[1:]:
        flat_dim *= int(s)
    out = jnp.zeros(z.shape[:1], dtype=jnp.float32)
    gap = _gap_broadcast(t - r, z.ndim)
    sum_axes = _data_axes(z.ndim)

    def _body_fn(i: int, val: jax.Array) -> jax.Array:
        e_i_flat = jax.nn.one_hot(i, flat_dim, dtype=z.dtype)
        e_i = e_i_flat.reshape(z.shape[1:])
        e_i = jnp.broadcast_to(e_i[None], z.shape)
        _, jv = jax.jvp(lambda x: u_fn(x, r, t), (z,), (e_i,))
        bv = gap * jv - e_i
        return val + jnp.sum(jnp.square(bv), axis=sum_axes).astype(val.dtype)

    out = jax.lax.fori_loop(
        lower=0,
        upper=flat_dim,
        body_fun=_body_fn,
        init_val=out,
    )
    return out.astype(z.dtype)


def hutchinson_trace(
    key: typing.Any,
    u_fn: typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
    n_probes: int = 1,
) -> jax.Array:
    r"""Hutchinson estimate of :math:`\mathrm{tr}(BB^{\top})`,
    where :math:`B = (t-r) J - I` and :math:`J = \partial_z u`.

    Estimates :math:`\mathrm{tr}(BB^{\top}) \approx \tfrac{1}{N}\sum_{i}
    \| B v_i \|_2^2` with i.i.d. Rademacher probes :math:`v_i` of the same
    shape as ``z``. Treats axis 0 as the batch axis and sums squares over
    all remaining axes, so it works for both flat ``(B, d)`` and image
    ``(B, H, W, C)`` inputs.

    Args:
        key: PRNG key for probe sampling.
        u_fn: Velocity field ``u_fn(z, r, t)``; output has the same shape as ``z``.
        z: Noisy input of shape ``(B,) + data_dims``.
        r: Start timestamps of shape ``(B,)``.
        t: End timestamps of shape ``(B,)``.
        n_probes: Number of Rademacher probes. Default ``1``.

    Returns:
        Per-sample trace estimate of shape ``(B,)``.
    """
    out = jnp.zeros(z.shape[:1], dtype=jnp.float32)
    gap = _gap_broadcast(t - r, z.ndim)
    sum_axes = _data_axes(z.ndim)

    def _body_fn(i: int, val: jax.Array) -> jax.Array:
        local_key = jrnd.fold_in(key, i)
        v = jrnd.rademacher(local_key, z.shape, dtype=z.dtype)
        _, jv = jax.jvp(lambda z_: u_fn(z_, r, t), (z,), (v,))
        bv = gap * jv - v
        return val + jnp.sum(jnp.square(bv), axis=sum_axes).astype(val.dtype)

    out = jax.lax.fori_loop(
        lower=0,
        upper=n_probes,
        body_fun=_body_fn,
        init_val=out,
    )
    return (out / n_probes).astype(z.dtype)
