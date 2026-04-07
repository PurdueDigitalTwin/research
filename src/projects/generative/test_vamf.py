"""Tests for ``VAMeanFlowUNetModule`` / ``VAMeanFlowUNetModel``.

Regression tests for the P0/P1/P2 fixes from
``docs/generative/vamf/reviews/nll-audit-2026-04-06.md`` Revision 1.

The core guarantees these tests enforce:

1. The variance head is per-pixel-per-channel ``(B, H, W, C)``. The
   audit's original concern was a broadcast-shape crash in the NLL
   loss; the fix keeps the head per-pixel and rewrites the loss to a
   proper diagonal-Gaussian per-pixel sum. These tests lock that shape.
2. Running the full ``training_step`` through ``jax.pmap`` with
   ``predict_variance=True`` past ``nll_warmup_steps`` yields finite
   losses. Prior to the Revision 1 fix this would raise
   ``ValueError: Incompatible shapes for broadcasting: shapes=[(B,),
   (B,H,W,C)]``.
3. The MSE variant (``predict_variance=False``) is unchanged — the
   variance head is absent and the SNR-weighted MSE path still runs.
"""

from __future__ import annotations

import functools
import sys

from flax import jax_utils
import jax
from jax import numpy as jnp
import optax
import pytest

from src.core import train_state as _train_state
from src.projects.generative import meanflow

# Small test dimensions — CIFAR-shaped but downscaled to keep CPU
# test runtime reasonable. ``features`` must remain divisible by 32
# for the internal GroupNorm in SongNetwork.
_B = 2
_H = 16
_W = 16
_C = 3
_FEATURES = 64


def _build_model(
    *,
    predict_variance: bool,
    nll_warmup_steps: int = 3,
    no_fm_anchor: bool = False,
    boundary_tangent: bool = False,
) -> meanflow.VAMeanFlowUNetModel:
    """Construct a small ``VAMeanFlowUNetModel`` for tests."""
    return meanflow.VAMeanFlowUNetModel(
        in_channels=_C,
        image_size=_H,
        features=_FEATURES,
        dropout_rate=0.0,
        predict_variance=predict_variance,
        variance_floor=1e-4,
        nll_warmup_steps=nll_warmup_steps,
        nll_ramp_steps=2,
        fm_anchor_weight=0.5,
        fm_anchor_delta_min=1e-4,
        fm_anchor_delta_max=0.01,
        timestamp_overlap_rate=0.5,
        no_fm_anchor=no_fm_anchor,
        boundary_tangent=boundary_tangent,
    )


def _init_state(
    model: meanflow.VAMeanFlowUNetModel,
    rng: jax.Array,
) -> _train_state.TrainState:
    """Initialize model params and wrap in a replicated ``TrainState``."""
    params, _ = model.init(batch=None, rngs=rng)
    tx = optax.adam(learning_rate=1e-4)
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=0.999,
    )
    return jax_utils.replicate(state)


def _make_batch() -> dict:
    """Build a single-device sharded dummy batch."""
    return {
        "image": 0.5
        * jnp.ones(
            (1, _B, _H, _W, _C),
            dtype=jnp.float32,
        )
    }


def test_nll_perpixel_variance_shape() -> None:
    """Variance head must output per-pixel-per-channel ``(B, H, W, C)``.

    Regression guard for the P0 diagnosis. Prior to the fix the audit mistakenly described this
    head as a per-sample scalar; this test pins down the correct spatial heteroscedastic shape.
    """
    module = meanflow.VAMeanFlowUNetModule(
        features=_FEATURES,
        dropout_rate=0.0,
        epsilon=1e-6,
        skip_scale=1.0,
        predict_variance=True,
    )
    inputs = jnp.ones((_B, _H, _W, _C), dtype=jnp.float32)
    timestamps = (
        jnp.zeros((_B,), dtype=jnp.float32),
        jnp.zeros((_B,), dtype=jnp.float32),
    )
    variables = module.init(
        rngs={"params": jax.random.PRNGKey(0)},
        inputs=inputs,
        timestamps=timestamps,
        deterministic=True,
    )
    u, log_var = module.apply(
        variables,
        inputs=inputs,
        timestamps=timestamps,
        deterministic=True,
    )
    assert u.shape == (_B, _H, _W, _C)
    assert log_var is not None
    assert log_var.shape == (_B, _H, _W, _C), (
        "expected per-pixel-per-channel variance of shape "
        f"(B,H,W,C) = ({_B},{_H},{_W},{_C}), got {log_var.shape}"
    )


def test_mse_variant_has_no_variance_head() -> None:
    """``predict_variance=False`` must not emit a log_var output.

    Regression guard: the MSE variant of VaMF already works in
    production (run 9k3bt7aa), so after the NLL fix we must leave it
    untouched — no variance head, ``log_var is None``.
    """
    module = meanflow.VAMeanFlowUNetModule(
        features=_FEATURES,
        dropout_rate=0.0,
        epsilon=1e-6,
        skip_scale=1.0,
        predict_variance=False,
    )
    inputs = jnp.ones((_B, _H, _W, _C), dtype=jnp.float32)
    timestamps = (
        jnp.zeros((_B,), dtype=jnp.float32),
        jnp.zeros((_B,), dtype=jnp.float32),
    )
    variables = module.init(
        rngs={"params": jax.random.PRNGKey(0)},
        inputs=inputs,
        timestamps=timestamps,
        deterministic=True,
    )
    u, log_var = module.apply(
        variables,
        inputs=inputs,
        timestamps=timestamps,
        deterministic=True,
    )
    assert u.shape == (_B, _H, _W, _C)
    assert log_var is None


@pytest.mark.parametrize("predict_variance", [False, True])
def test_training_step_finite(predict_variance: bool) -> None:
    """End-to-end ``training_step`` must compile and yield finite loss.

    Regression test for the P0 broadcast crash: prior to the fix this
    test would raise
    ``ValueError: Incompatible shapes for broadcasting: shapes=[(B,),
    (B,H,W,C)]`` the moment the NLL branch of ``_loss_fn`` was hit.

    We force ``nll_warmup_steps=3`` and ``nll_ramp_steps=2`` so the
    per-pixel NLL branch is exercised on the very first step
    (``alpha = max((0 - 1) / 2, 0) = 0``) and on a later step where
    ``alpha > 0``. For the NLL variant we also run a second step
    with an artificially incremented ``state.step`` to cover the
    ``alpha = 1`` post-warmup regime.
    """
    model = _build_model(predict_variance=predict_variance, nll_warmup_steps=3)
    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(0))
    state = _init_state(model, init_rng)
    batch = _make_batch()

    p_step = jax.pmap(
        functools.partial(model.training_step, rngs=train_rng),
        axis_name="batch",
    )

    new_state, outputs = p_step(state=state, batch=batch)

    for key in ("loss", "mf_loss", "fm_anchor_loss", "velocity_loss"):
        val = jnp.asarray(outputs.scalars[key])
        assert jnp.all(
            jnp.isfinite(val)
        ), f"{key} is not finite on step 0: {val}"

    if predict_variance:
        # The NLL branch publishes extra scalars only when
        # ``predict_variance=True``. Verify they are finite.
        for key in ("sigma_sq_mean", "log_var_std", "fm_mf_ratio"):
            val = jnp.asarray(outputs.scalars[key])
            assert jnp.all(
                jnp.isfinite(val)
            ), f"{key} is not finite on step 0: {val}"

        # Advance past the warmup end (step=3) so alpha clamps to 1
        # and the loss is pure per-pixel NLL. This is the regime
        # where the original P0 bug would have crashed.
        bumped = new_state.replace(step=new_state.step + 5)
        _, post_outputs = p_step(state=bumped, batch=batch)
        for key in (
            "loss",
            "mf_loss",
            "fm_anchor_loss",
            "sigma_sq_mean",
            "log_var_std",
        ):
            val = jnp.asarray(post_outputs.scalars[key])
            assert jnp.all(
                jnp.isfinite(val)
            ), f"{key} is not finite after warmup: {val}"


@pytest.mark.parametrize(
    "no_fm_anchor,boundary_tangent",
    [
        (True, False),  # R5a/R5d-style: FM anchor disabled
        (False, True),  # R5b-style: boundary tangent (no EMA)
        (True, True),  # both flags on (sanity check)
    ],
)
def test_ablation_flags_finite(
    no_fm_anchor: bool, boundary_tangent: bool
) -> None:
    """Ablation flags must compile and produce finite losses.

    Smoke test for the R5a/R5b/R5d ablations from the submission
    plan: ``no_fm_anchor=True`` (skip the FM anchor branch entirely)
    and ``boundary_tangent=True`` (use the current model's own
    boundary prediction as the JVP tangent instead of the EMA model).
    Both flags target the MSE variant; we exercise them on the MSE
    variant since the NLL variant adds orthogonal complexity already
    covered by ``test_training_step_finite``.

    When ``no_fm_anchor=True`` we additionally assert that
    ``fm_anchor_loss`` is exactly zero, since the branch should
    short-circuit before any forward pass.
    """
    model = _build_model(
        predict_variance=False,
        no_fm_anchor=no_fm_anchor,
        boundary_tangent=boundary_tangent,
    )
    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(0))
    state = _init_state(model, init_rng)
    batch = _make_batch()

    p_step = jax.pmap(
        functools.partial(model.training_step, rngs=train_rng),
        axis_name="batch",
    )

    _, outputs = p_step(state=state, batch=batch)

    for key in ("loss", "mf_loss", "fm_anchor_loss", "velocity_loss"):
        val = jnp.asarray(outputs.scalars[key])
        assert jnp.all(
            jnp.isfinite(val)
        ), f"{key} is not finite with flags={no_fm_anchor},{boundary_tangent}: {val}"

    if no_fm_anchor:
        fm_val = jnp.asarray(outputs.scalars["fm_anchor_loss"])
        assert jnp.all(
            fm_val == 0.0
        ), f"fm_anchor_loss must be zero under no_fm_anchor, got {fm_val}"
        # When the anchor is disabled total_loss == mf_loss exactly.
        loss_val = jnp.asarray(outputs.scalars["loss"])
        mf_val = jnp.asarray(outputs.scalars["mf_loss"])
        assert jnp.allclose(loss_val, mf_val), (
            "total loss must equal mf_loss under no_fm_anchor; "
            f"got loss={loss_val}, mf_loss={mf_val}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
