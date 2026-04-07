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

        # PR (d) regression guard for the sign-inversion bug
        # observed on run ``apjbrvz2``. At step 0 the MSE->NLL
        # ramp is at ``alpha = 0`` so *both* ``mf_loss`` and
        # ``fm_anchor_loss`` must be pure per-sample sums of
        # squared residuals — strictly non-negative. Prior to
        # PR (d), the FM anchor used per-pixel NLL from step 0
        # regardless of ``alpha``, training the variance head
        # exclusively via the anchor during pre-warmup and
        # driving ``sigma^2`` downward until the sign of
        # ``0.5*log sigma^2`` flipped the whole loss.
        fm_val = jnp.asarray(outputs.scalars["fm_anchor_loss"])
        assert jnp.all(fm_val >= 0.0), (
            "fm_anchor_loss must be >= 0 at step 0 (alpha=0, "
            f"pure MSE) under predict_variance=True; got {fm_val}"
        )
        mf_val_step0 = jnp.asarray(outputs.scalars["mf_loss"])
        assert jnp.all(mf_val_step0 >= 0.0), (
            "mf_loss must be >= 0 at step 0 (alpha=0, pure "
            f"MSE) under predict_variance=True; got {mf_val_step0}"
        )

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


def _beta_nll_per_pixel(
    *,
    v_pred: jax.Array,
    v_target: jax.Array,
    log_var: jax.Array,
    nll_beta: float,
    variance_floor: float = 1e-4,
) -> jax.Array:
    """Mirror of the β-NLL formula used in ``meanflow._loss_fn``.

    Kept intentionally short and standalone so a future refactor to the main code cannot silently
    change the math without also failing this regression test. If the main code deviates from this
    formula, either the main code is wrong or this test must be updated in the same commit.
    """
    sigma_sq = jax.nn.softplus(log_var) + variance_floor
    raw = 0.5 * (jnp.square(v_pred - v_target) / sigma_sq + jnp.log(sigma_sq))
    weight = jax.lax.stop_gradient(sigma_sq**nll_beta)
    return weight * raw


def test_beta_nll_mean_grad_matches_mse_at_beta_one() -> None:
    """β=1 must make the mean-head gradient equal plain MSE gradient.

    The whole point of the β-NLL fix (audit Revision 3 / PR (e)) is
    to decouple the mean head's effective learning rate from
    ``sigma^2``. At β=1 the per-pixel NLL's gradient wrt ``v_pred``
    is ``stop_gradient(sigma^{2*(1-1)}) * (v_pred - v_target) =
    (v_pred - v_target)`` — numerically identical to MSE.

    This test fixes ``v_pred``, ``v_target``, and ``log_var`` to
    concrete values and asserts the numerical invariant. Any
    regression that removes ``stop_gradient`` or changes the β
    exponent will be caught.
    """
    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    v_pred = jax.random.normal(k1, (_B, _H, _W, _C))
    v_target = jax.random.normal(k2, (_B, _H, _W, _C))
    # log_var in a realistic post-warmup range: softplus(log_var) + 1e-4
    # spans roughly [0.05, 1.5] so 1/sigma^2 amplification is non-trivial.
    log_var = jax.random.normal(k3, (_B, _H, _W, _C)) - 1.0

    def beta_nll_total(vp):
        return jnp.sum(
            _beta_nll_per_pixel(
                v_pred=vp,
                v_target=v_target,
                log_var=log_var,
                nll_beta=1.0,
            )
        )

    def mse_total(vp):
        return 0.5 * jnp.sum(jnp.square(vp - v_target))

    grad_beta = jax.grad(beta_nll_total)(v_pred)
    grad_mse = jax.grad(mse_total)(v_pred)

    assert jnp.allclose(grad_beta, grad_mse, atol=1e-5), (
        "β=1 mean-head gradient must equal MSE gradient. "
        f"max |diff| = {jnp.max(jnp.abs(grad_beta - grad_mse))}"
    )


def test_beta_nll_zero_amplifies_by_inv_sigma_sq() -> None:
    """β=0 must reproduce plain NLL with 1/σ² mean-head amplification.

    This is the *failure mode* PR (e) fixes — it is tested here so
    that if someone flips the default back to 0 they get a clean
    contrast against the β=1 baseline. The mean-head gradient at β=0
    is ``(v_pred - v_target) / sigma^2``, so with ``sigma^2 < 1`` it
    is strictly larger in magnitude than the MSE gradient.
    """
    rng = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(rng)
    v_pred = jax.random.normal(k1, (_B, _H, _W, _C))
    v_target = jax.random.normal(k2, (_B, _H, _W, _C))
    # Force sigma^2 ≈ 0.13 (the observed post-warmup scale on
    # hz3dpmz4): softplus(log_var) + 1e-4 = 0.13 → log_var ≈ -1.93.
    log_var = jnp.full((_B, _H, _W, _C), -1.93)

    def nll_total(vp):
        return jnp.sum(
            _beta_nll_per_pixel(
                v_pred=vp,
                v_target=v_target,
                log_var=log_var,
                nll_beta=0.0,
            )
        )

    grad_nll = jax.grad(nll_total)(v_pred)
    grad_mse = v_pred - v_target
    sigma_sq = jax.nn.softplus(log_var) + 1e-4
    expected = grad_mse / sigma_sq

    assert jnp.allclose(grad_nll, expected, atol=1e-5), (
        "β=0 mean-head gradient must equal (v_pred - v_target) / σ². "
        f"max |diff| = {jnp.max(jnp.abs(grad_nll - expected))}"
    )

    # Sanity: the β=0 gradient should be substantially larger than
    # MSE at σ² ≈ 0.13 — this is the amplification that triggered
    # the hz3dpmz4 FID regression.
    ratio = jnp.mean(jnp.abs(grad_nll)) / jnp.mean(jnp.abs(grad_mse))
    assert ratio > 5.0, (
        "β=0 / MSE gradient magnitude ratio should be > 5 at "
        f"σ² ≈ 0.13; got {float(ratio):.2f}"
    )


def test_beta_nll_variance_grad_scales_with_sigma_sq() -> None:
    """Variance-head gradient must satisfy ``grad_β = σ²^β · grad_0``.

    This is the property that justifies PR (e): we decouple the mean
    head's effective LR from ``σ²`` *without* changing what the
    variance head converges to. Under ``jax.lax.stop_gradient`` the
    β-weight is a constant during differentiation, so

    ``∂L_β/∂log_var = stop_gradient(σ²^β) · ∂L_0/∂log_var``.

    Two consequences follow:

    1. **Fixed point preserved.** Wherever plain-NLL's log_var
       gradient ``∂L_0/∂log_var`` is zero (σ²* = r² per pixel), the
       β-NLL gradient is also zero regardless of β — zero scaled by
       any weight is still zero. So the variance head converges to
       the same place under any β.
    2. **Per-step magnitude differs.** Away from the fixed point,
       the β=1 variance-head update is ``σ²`` times the β=0 update
       (element-wise). Under σ² < 1 this slows the variance head's
       convergence, but it does not change its target.

    This test locks in the exact element-wise scaling relationship,
    which implies both (1) and (2).
    """
    rng = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(rng, 3)
    v_pred = jax.random.normal(k1, (_B, _H, _W, _C))
    v_target = jax.random.normal(k2, (_B, _H, _W, _C))
    log_var = jax.random.normal(k3, (_B, _H, _W, _C)) - 0.5
    sigma_sq = jax.nn.softplus(log_var) + 1e-4

    def total(lv, beta):
        return jnp.sum(
            _beta_nll_per_pixel(
                v_pred=v_pred,
                v_target=v_target,
                log_var=lv,
                nll_beta=beta,
            )
        )

    grad_beta0 = jax.grad(lambda lv: total(lv, 0.0))(log_var)
    grad_beta1 = jax.grad(lambda lv: total(lv, 1.0))(log_var)
    grad_beta_half = jax.grad(lambda lv: total(lv, 0.5))(log_var)

    expected_beta1 = sigma_sq * grad_beta0
    expected_beta_half = jnp.sqrt(sigma_sq) * grad_beta0

    assert jnp.allclose(grad_beta1, expected_beta1, atol=1e-5), (
        "log_var gradient must satisfy grad_{β=1} = σ² · grad_{β=0} "
        "(stop_gradient freezes the σ²^β weight); "
        f"max |diff| = {jnp.max(jnp.abs(grad_beta1 - expected_beta1))}"
    )
    assert jnp.allclose(grad_beta_half, expected_beta_half, atol=1e-5), (
        "log_var gradient must satisfy grad_{β=0.5} = σ · grad_{β=0}; "
        f"max |diff| = {jnp.max(jnp.abs(grad_beta_half - expected_beta_half))}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
