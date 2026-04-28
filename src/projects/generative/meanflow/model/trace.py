import typing

import jax
from jax import numpy as jnp
from jax import random as jrnd


def exact_trace(
    u_fn: typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
) -> jax.Array:
    r"""Evaluate the exact trace term :math:`\mathrm{tr}(BB^{\top})`.

    Args:
        u_fn (Callable): Average velocity field function ``u_fn(z, r, t)``.
        z (jax.Array): Noisy input at time step ``t`` of shape ``(*, d)``.
        r (jax.Array): Start time step of shape ``(*,)``.
        t (jax.Array): End time step of shape ``(*,)``.

    Returns:
        Calculated trace term of shape ``(*,)``.
    """
    dim = z.shape[-1]
    out = jnp.zeros_like(z[..., 0])

    def _body_fn(i: int, val: jax.Array) -> jax.Array:
        e_i = jnp.zeros_like(z).at[..., i].set(1.0)
        _, jv = jax.jvp(lambda x: u_fn(x, r, t), (z,), (e_i,))
        jv = (t - r)[..., None] * jv - e_i

        return val + jnp.sum(jnp.square(jv), axis=-1)

    out = jax.lax.fori_loop(
        lower=0,
        upper=dim,
        body_fun=_body_fn,
        init_val=out,
    )

    return out


def hutchinson_trace(
    key: typing.Any,
    u_fn: typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
    n_probes: int = 1,
) -> jax.Array:
    r"""Computes the Huntchinson's estimate of :math:`\mathrm{tr}(BB^{\top})`.

    .. note::

        The trace is estimated by
        :math:`\\frac{1}{N}\\sum_{i}\|(t-r)Jv_{i}-v_{i}\|_{2}^{2}`.

    Args:
        key (Any): Random generator key for reproducibility.
        u_fn (Callable): Average velocity field function ``u_fn(z, r, t)``.
        z (jax.Array): Noisy input at time step ``t`` of shape ``(*, d)``.
        r (jax.Array): Start time step of shape ``(*,)``.
        t (jax.Array): End time step of shape ``(*,)``.
        n_probes (int, optional): Number of Rademacher probes ``v_{i}``.
            Default is :math:`1`.

    Returns:
        Calculated trace term of shape ``(*,)``.
    """
    # NOTE: enforce the accumulator to use ``float32`` for numerical stability
    out = jnp.zeros(z.shape[:-1], dtype=jnp.float32)

    def _body_fn(i: int, val: jax.Array) -> jax.Array:
        local_key = jrnd.fold_in(key, i)
        v = jrnd.rademacher(local_key, z.shape, dtype=z.dtype)
        _, jv = jax.jvp(lambda z_: u_fn(z_, r, t), (z,), (v,))
        jv = (t - r)[..., None] * jv - v

        return val + jnp.sum(jnp.square(jv), axis=-1).astype(val.dtype)

    out = jax.lax.fori_loop(
        lower=0,
        upper=n_probes,
        body_fun=_body_fn,
        init_val=out,
    )

    return (out / n_probes).astype(z.dtype)
