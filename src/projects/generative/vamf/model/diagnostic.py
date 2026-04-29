"""Diagnostic primitives for MeanFlow variance analysis."""

import typing

import jax
from jax import numpy as jnp
from jax import random as jrnd
import numpy as np

from src.projects.generative.vamf.model import trace as _trace

# ----- Type aliases ----------------------------------------------------------
UFn = typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
Sampler = typing.Callable[[jax.Array, int], jax.Array]


# ----- Helpers ---------------------------------------------------------------
def _sum_data_axes(x: jax.Array) -> jax.Array:
    r"""Sum squared norms over all axes after the batch axis."""
    return jnp.sum(jnp.square(x), axis=tuple(range(1, x.ndim)))


def _gap_broadcast(gap: jax.Array, ndim: int) -> jax.Array:
    r"""Reshape a per-sample scalar to broadcast against an ndim tensor."""
    return jnp.reshape(gap, (-1,) + (1,) * (ndim - 1))


# ----- Experiment 1: variance amplification ----------------------------------
def _exp1_probe(
    u_fn: UFn, x0: jax.Array, e: jax.Array, t_val: jax.Array, r_val: jax.Array
) -> typing.Tuple[jax.Array, jax.Array]:
    r"""Computes per-batch variance ratio at fixed ``(t, r)``."""
    n = x0.shape[0]
    t = jnp.broadcast_to(t_val, (n,))
    r = jnp.broadcast_to(r_val, (n,))
    gap = _gap_broadcast(t - r, x0.ndim)
    z = (1.0 - _gap_broadcast(t, x0.ndim)) * x0 + _gap_broadcast(
        t, x0.ndim
    ) * e
    v_cond = e - x0
    zeros_n, ones_n = jnp.zeros(n), jnp.ones(n)

    # Stochastic tangent: vanilla MeanFlow uses v_cond.
    u_rt, dudt_s = jax.jvp(u_fn, (z, r, t), (v_cond, zeros_n, ones_n))
    v_pred_s = u_rt + gap * dudt_s
    L_s = _sum_data_axes(v_pred_s - v_cond)

    # Deterministic tangent: VaMF replaces v_cond with the model's own
    # prediction at the boundary r=t.
    v_tang_d = jax.lax.stop_gradient(u_fn(z, t, t))
    _, dudt_d = jax.jvp(u_fn, (z, r, t), (v_tang_d, zeros_n, ones_n))
    v_pred_d = u_rt + gap * dudt_d
    L_d = _sum_data_axes(v_pred_d - v_cond)

    return jnp.var(L_s), jnp.var(L_d)


def variance_amplification(
    u_fn: UFn,
    sample_x0: Sampler,
    key: jax.Array,
    n_samples: int,
    *,
    t_probes: typing.Sequence[float] = tuple(np.arange(0.1, 1.0, 0.1)),
    fixed_gap: float = 0.25,
    log_fn: typing.Callable[..., None] | None = None,
) -> dict[str, dict[str, float]]:
    r"""Compute variance ratio at each given ``t``."""
    probe = jax.jit(_exp1_probe, static_argnames=())

    # Bind u_fn into a JIT'd specialization. We close over u_fn here
    # because ``static_argnums`` cannot accept a Python callable.
    @jax.jit
    def _probe(x0, e, t_val, r_val):
        return _exp1_probe(u_fn, x0, e, t_val, r_val)

    results = {}
    for t_val in t_probes:
        k_data, k_noise, key = jrnd.split(key, 3)
        x0 = sample_x0(k_data, n_samples)
        e = jrnd.normal(k_noise, x0.shape, dtype=x0.dtype)
        r_val = max(float(t_val) - fixed_gap, 1e-4)
        var_s, var_d = _probe(
            x0,
            e,
            jnp.float32(t_val),
            jnp.float32(r_val),
        )
        var_s, var_d = float(var_s), float(var_d)
        ratio = var_s / max(var_d, 1e-12)
        results[f"t={t_val:.1f}"] = {
            "variance_ratio": ratio,
            "stochastic_var": var_s,
            "deterministic_var": var_d,
        }
        if log_fn is not None:
            log_fn(
                "  t=%.1f: Var(stoch)=%.4f  Var(det)=%.4f  ratio=%.2f",
                t_val,
                var_s,
                var_d,
                ratio,
            )
    return results


# ----- Experiment 2: curvature gap -------------------------------------------
def _exp2_probe(
    u_fn: UFn, x0: jax.Array, e: jax.Array, t_val: jax.Array, r_val: jax.Array
) -> jax.Array:
    r"""Computes mean :math:`||u(z,r,t) - v_cond||^2` of a fixed ``(x0, e)``."""
    n = x0.shape[0]
    t = jnp.broadcast_to(t_val, (n,))
    r = jnp.broadcast_to(r_val, (n,))
    z = (1.0 - _gap_broadcast(t, x0.ndim)) * x0 + _gap_broadcast(
        t, x0.ndim
    ) * e
    v_cond = e - x0
    u = u_fn(z, r, t)
    return jnp.mean(_sum_data_axes(u - v_cond))


def curvature_gap(
    u_fn: UFn,
    sample_x0: Sampler,
    key: jax.Array,
    n_samples: int,
    *,
    t_probes: typing.Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    n_gaps: int = 12,
    max_gap_cap: float = 0.45,
    log_fn: typing.Callable[..., None] | None = None,
) -> dict[str, list[dict[str, float]]]:
    r"""Computes :math:`||u(z,r,t) - v_cond||^{2}` vs (t-r) for Theorem 3."""

    @jax.jit
    def _probe(x0, e, t_val, r_val):
        return _exp2_probe(u_fn, x0, e, t_val, r_val)

    results = {}
    for t_val in t_probes:
        max_g = min(float(t_val) - 1e-4, max_gap_cap)
        gaps_list = []
        for g in np.linspace(0.01, max_g, n_gaps):
            k_data, k_noise, key = jrnd.split(key, 3)
            x0 = sample_x0(k_data, n_samples)
            e = jrnd.normal(k_noise, x0.shape, dtype=x0.dtype)
            mse = float(
                _probe(
                    x0,
                    e,
                    jnp.float32(t_val),
                    jnp.float32(t_val - g),
                )
            )
            gaps_list.append(
                {
                    "t_minus_r": float(g),
                    "gap_sq_mean": mse,
                }
            )
        results[f"t={t_val}"] = gaps_list
        if log_fn is not None:
            log_fn(
                "  t=%.1f: gap [%.2f, %.2f], MSE [%.4f, %.4f]",
                t_val,
                gaps_list[0]["t_minus_r"],
                gaps_list[-1]["t_minus_r"],
                gaps_list[0]["gap_sq_mean"],
                gaps_list[-1]["gap_sq_mean"],
            )
    return results


# ----- Experiment 4: jacobian norm -------------------------------------------
def jacobian_norm(
    u_fn: UFn,
    sample_x0: Sampler,
    key: jax.Array,
    n_samples: int,
    *,
    t_probes: typing.Sequence[float] = tuple(np.arange(0.1, 1.0, 0.1)),
    fixed_gap: float = 0.25,
    n_probes: int = 1,
    exact: bool = False,
    log_fn: typing.Callable[..., None] | None = None,
) -> dict[str, dict[str, float]]:
    r"""Compute the Frobenius norm :math:`||(t-r) J_z - I||_{F}`"""

    @jax.jit
    def _probe(x0, e, t_val, r_val, probe_key):
        n = x0.shape[0]
        t = jnp.broadcast_to(t_val, (n,))
        r = jnp.broadcast_to(r_val, (n,))
        z = (1.0 - _gap_broadcast(t, x0.ndim)) * x0 + _gap_broadcast(
            t, x0.ndim
        ) * e
        if exact:
            tr_bbt = _trace.exact_trace(u_fn, z, r, t)
        else:
            tr_bbt = _trace.hutchinson_trace(
                probe_key,
                u_fn,
                z,
                r,
                t,
                n_probes=n_probes,
            )
        # ``trace`` returns ||B||_F^2 = ||(t-r) J - I||_F^2 directly.
        J_norm = jnp.sqrt(tr_bbt)
        return jnp.mean(J_norm), jnp.std(tr_bbt)

    results = {}
    for t_val in t_probes:
        k_data, k_noise, k_probe, key = jrnd.split(key, 4)
        x0 = sample_x0(k_data, n_samples)
        e = jrnd.normal(k_noise, x0.shape, dtype=x0.dtype)
        r_val = max(float(t_val) - fixed_gap, 1e-4)
        J_mean, J_sq_std = _probe(
            x0,
            e,
            jnp.float32(t_val),
            jnp.float32(r_val),
            k_probe,
        )
        results[f"t={t_val:.1f}"] = {
            "J_norm_mean": float(J_mean),
            "J_norm_sq_std": float(J_sq_std),
        }
        if log_fn is not None:
            log_fn(
                "  t=%.1f: ||(t-r)J - I||_F = %.4f",
                t_val,
                float(J_mean),
            )
    return results
