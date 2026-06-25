"""Host-side tangent-mixing coefficient schedule beta(step) for VAMF B1.

The MeanFlow JVP tangent in ``vamf_tmix`` is

    v_tang = (1 - beta) * v_cond + beta * u_bar(x_t, t, t).

B1 anneals ``beta`` over training from ``beta_start`` (high-beta corner,
strong noise cancellation) to ``beta_end`` (the FID-optimal corner,
typically 0). The schedule is evaluated in plain Python on the training
driver and the resulting scalar is passed into the jitted train / probe
step. Keeping it host-side means beta can change every step WITHOUT
triggering JAX recompilation, and the exact same function plugs into the
DiT trainer (JAX or PyTorch) unchanged.

Design decisions locked with the author:
  * beta_start = 1.0  (existing EMA-tangent corner; ~= the measured 0.94)
  * beta_end   = 0.0  (FID-optimal corner)
  * only the *tangent* beta is annealed; the regression target stays
    v_cond, and any r=t flow-matching auxiliary loss weight stays CONSTANT.
"""

import math

try:
    from jax import numpy as jnp
except ImportError:  # pragma: no cover – pure-Python fallback
    jnp = None  # type: ignore[assignment]

_SHAPES = ("constant", "linear", "cosine", "step")


def beta_at_step(
    step: int,
    total_steps: int,
    *,
    shape: str = "constant",
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    s0: float = 0.0,
    s1: float = 0.6,
) -> float:
    """Return the tangent-mixing coefficient beta in [0, 1] at ``step``.

    Args:
        step: current training step (0-indexed).
        total_steps: total number of training steps (the schedule horizon).
        shape: one of:
            "constant" -- beta_start for all steps; recovers the static
                           --tangent_beta behavior (fully backward-compatible).
            "linear"   -- hold beta_start until s0*T, linearly interpolate to
                           beta_end by s1*T, then hold beta_end.
            "cosine"   -- same window as "linear" but a half-cosine
                           interpolation (flat tangents at both ends).
            "step"     -- hold beta_start, then drop to beta_end at s1*T
                           (s0 is ignored).
        beta_start: beta at step 0 (anneal source).
        beta_end: beta after the anneal window (target corner).
        s0: anneal-window start as a fraction of total_steps, in [0, 1].
        s1: anneal-window end as a fraction of total_steps, in [0, 1], s1 >= s0.

    Returns:
        A Python float beta(step). Wrap with ``jnp.asarray(..., jnp.float32)``
        before passing into a jitted step so it is traced, not static.
    """
    if shape not in _SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {_SHAPES}")
    if not (0.0 <= s0 <= s1 <= 1.0):
        raise ValueError(f"require 0 <= s0 <= s1 <= 1, got s0={s0}, s1={s1}")
    if total_steps <= 0 or shape == "constant":
        return float(beta_start)

    frac = step / float(total_steps)  # progress in [0, 1)

    if shape == "step":
        return float(beta_start if frac < s1 else beta_end)

    # linear / cosine: windowed progress p in [0, 1]
    if s1 <= s0:  # degenerate window -> instant drop at s0
        p = 0.0 if frac < s0 else 1.0
    else:
        p = (frac - s0) / (s1 - s0)
        p = min(1.0, max(0.0, p))  # clamp to the hold regions

    if shape == "linear":
        return float(beta_start + (beta_end - beta_start) * p)
    # cosine: p=0 -> beta_start, p=1 -> beta_end, monotone non-increasing
    return float(beta_end + (beta_start - beta_end) * 0.5 * (1.0 + math.cos(math.pi * p)))


def jax_beta_at_step(
    step,
    total_steps,
    *,
    shape: str = "constant",
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    s0: float = 0.0,
    s1: float = 0.6,
):
    """JAX-traceable ``beta_at_step`` for use inside pmap/jit.

    Unlike ``beta_at_step`` (pure Python, host-side), this version
    uses ``jnp`` arithmetic so ``step`` can be a traced integer
    (e.g. ``state.step`` inside a pmap'd training_step). The
    ``shape`` string is resolved at trace time (Python if/elif)
    so it does not cause retracing as long as the model attribute
    is constant across calls (which it always is).
    """
    if shape not in _SHAPES:
        raise ValueError(
            f"unknown shape {shape!r}; expected one of {_SHAPES}"
        )
    if jnp is None:
        raise RuntimeError("jax_beta_at_step requires JAX")

    if shape == "constant":
        return jnp.float32(beta_start)

    frac = jnp.float32(step) / jnp.float32(total_steps)

    if shape == "step":
        return jnp.where(frac < s1, beta_start, beta_end)

    # linear / cosine: windowed progress p in [0, 1]
    if s1 <= s0:
        p = jnp.where(frac < s0, 0.0, 1.0)
    else:
        p = (frac - s0) / (s1 - s0)
        p = jnp.clip(p, 0.0, 1.0)

    if shape == "linear":
        return jnp.float32(beta_start + (beta_end - beta_start) * p)
    # cosine
    return jnp.float32(
        beta_end
        + (beta_start - beta_end) * 0.5 * (1.0 + jnp.cos(jnp.pi * p))
    )


# ----------------------------------------------------------------------------
# Smoke test / curve inspection. Run:  python beta_schedule.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    T = 200_000
    s1 = 0.6
    fracs = [i / 10 for i in range(11)]  # 0.0 .. 1.0

    print(f"beta(step) curves  (T={T:,}, s0=0.0, s1={s1}, start=1.0, end=0.0)\n")
    header = "frac   " + "".join(f"{sh:>10}" for sh in ("linear", "cosine", "step"))
    print(header)
    print("-" * len(header))
    for fr in fracs:
        step = int(fr * T)
        row = f"{fr:>4.1f}  "
        for sh in ("linear", "cosine", "step"):
            row += f"{beta_at_step(step, T, shape=sh, s1=s1):>10.3f}"
        print(row)

    for sh in ("linear", "cosine", "step"):
        assert abs(beta_at_step(0, T, shape=sh, s1=s1) - 1.0) < 1e-9, sh
        assert abs(beta_at_step(int(0.7 * T), T, shape=sh, s1=s1) - 0.0) < 1e-9, sh
        prev = 2.0
        for i in range(0, T + 1, T // 50):
            b = beta_at_step(i, T, shape=sh, s1=s1)
            assert 0.0 - 1e-9 <= b <= 1.0 + 1e-9, (sh, i, b)
            assert b <= prev + 1e-9, f"{sh} not monotone non-increasing at step {i}"
            prev = b
    assert beta_at_step(123, T, shape="constant", beta_start=0.4) == 0.4
    assert beta_at_step(0, T, shape="constant", beta_start=0.4) == 0.4
    mid = int(0.3 * T)
    assert abs(beta_at_step(mid, T, shape="linear", s1=s1) - 0.5) < 1e-9
    assert abs(beta_at_step(mid, T, shape="cosine", s1=s1) - 0.5) < 1e-9

    print("\nAll assertions passed: boundaries, [0,1] range, monotone non-increasing, "
          "constant backward-compat.")
