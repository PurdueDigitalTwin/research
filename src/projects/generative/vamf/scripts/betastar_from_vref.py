#!/usr/bin/env python3
"""Bias-corrected beta*(t) from an INDEPENDENT v_ref  (VAMF A1, Step 3).

Replaces the paper's self-referential EMA-proxy ||b||^2 with an independent
flow-matching reference v_ref. Ratio-based (self-normalizing), so the absolute
units / t-distribution of the paper's 8.2e3 are irrelevant.

For each t, over many REAL-latent batches at NULL class (index 1000):
    v_cond = e - x_0
    x_t    = (1-t) x_0 + t e
    u_MF   = MeanFlow_boundary(x_t, t, t; null)    # proxy whose bias we measure
    v_ref  = FM_reference(x_t, t; null)            # independent marginal-velocity estimate
    bias_sq = ||u_MF - v_ref||^2    (sum over dims)
    noise   = ||v_cond - v_ref||^2  (sum over dims)  ~ sigma^2(t) d
    ratio(t)     = mean(bias_sq) / mean(noise) = ||b(t)||^2 / (sigma^2(t) d)
    shrinkage(t) = 1 / (1 + ratio(t))
    beta*(t)     = beta_no_bias * shrinkage(t)

Headline question: does beta*(t) collapse toward 0  <=>  is ratio(t) >> 1 ?

CRITICAL — beta_no_bias:
    The paper's matrix-form no-bias bound is ~0.94, i.e. kappa/(kappa+1) ~ 0.94
    => kappa ~ 15.7. DO NOT use 0.5 (that is kappa=1, inconsistent with the
    paper's own 0.94). Default below is 0.94; pass --beta_no_bias to override or
    to use a t-resolved value if you have one.

This file has two parts:
  * STATS CORE — unit-tested. Run:  python betastar_from_vref.py --self-test
  * TPU HARNESS — loads checkpoints / runs DiT forward; run on the VM. The harness
    takes two apply-callables and a latent iterator so the math is testable with
    fakes; wire the callables to the real models (see __main__).
"""
import argparse
import numpy as np


# ----------------------------- STATS CORE (tested) -----------------------------
def ratio_stats(bias_sq, noise, beta_no_bias=0.94, n_boot=2000, seed=0):
    """Pooled per-sample bias_sq and noise (1-D arrays) -> ratio, shrinkage,
    beta*, each with a bootstrap 95% CI. ratio = mean(bias_sq)/mean(noise) is a
    ratio-of-means estimator; with large N its finite-sample bias is small and
    the bootstrap CI reflects the (denominator) noise."""
    bias_sq = np.asarray(bias_sq, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if bias_sq.shape != noise.shape or bias_sq.ndim != 1:
        raise ValueError("bias_sq and noise must be 1-D arrays of equal length")
    if np.any(noise <= 0):
        raise ValueError("noise must be positive (it is a squared norm)")
    n = bias_sq.size
    ratio = bias_sq.mean() / noise.mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = bias_sq[idx].mean() / noise[idx].mean()
    r_lo, r_hi = np.percentile(boots, [2.5, 97.5])
    shrink = lambda r: 1.0 / (1.0 + r)
    return {
        "n": int(n),
        "ratio": ratio, "ratio_ci": (r_lo, r_hi),
        "shrinkage": shrink(ratio), "shrinkage_ci": (shrink(r_hi), shrink(r_lo)),
        "beta_star": beta_no_bias * shrink(ratio),
        "beta_star_ci": (beta_no_bias * shrink(r_hi), beta_no_bias * shrink(r_lo)),
        "noise_mean": noise.mean(), "noise_cv": noise.std() / noise.mean(),
        "bias_mean": bias_sq.mean(),
    }


def print_table(per_t, beta_no_bias):
    print(f"\n=== bias-corrected beta*(t) via independent v_ref "
          f"(beta_no_bias={beta_no_bias}) ===")
    print("ratio(t)=||b||^2/sigma^2d ; shrinkage=1/(1+ratio) ; beta*=beta_no_bias*shrinkage\n")
    h = (f"{'t':>6}{'N':>8}{'ratio':>9}{'ratio95%CI':>20}"
         f"{'shrink':>9}{'beta*':>9}{'beta*95%CI':>18}{'noiseCV':>9}")
    print(h); print("-" * len(h))
    for t in sorted(per_t):
        s = per_t[t]
        print(f"{t:>6.2f}{s['n']:>8}{s['ratio']:>9.3f}"
              f"{('['+format(s['ratio_ci'][0],'.2f')+','+format(s['ratio_ci'][1],'.2f')+']'):>20}"
              f"{s['shrinkage']:>9.3f}{s['beta_star']:>9.3f}"
              f"{('['+format(s['beta_star_ci'][0],'.2f')+','+format(s['beta_star_ci'][1],'.2f')+']'):>18}"
              f"{s['noise_cv']:>9.2f}")
    betas = [per_t[t]["beta_star"] for t in per_t]
    print(f"\nbeta* range over t: [{min(betas):.3f}, {max(betas):.3f}]")
    if max(betas) < 0.15:
        print("--> beta* COLLAPSES toward 0 at all t: ||b||^2 >> sigma^2 d (high-d bias dominance).")
        print("    Reconciles FID's preference for beta=0 with the gradient-MSE optimum. Thesis supported.")
    elif min(betas) > 0.7:
        print("--> beta* stays near beta_no_bias: bias is small vs noise. The high-d bias-dominance")
        print("    story is NOT supported by v_ref — report this honestly; the paper's claim needs revisiting.")
    else:
        print("--> beta* is intermediate / t-dependent. Report the full curve; no clean collapse.")


# ----------------------------- TPU HARNESS (run on VM) -------------------------
def run_eval(get_latent_batch, mf_apply, vref_apply, t_grid, n_batches,
             beta_no_bias=0.94, seed=0):
    """get_latent_batch() -> x_0 real-latent array [B,4,32,32];
    mf_apply(z,t)/vref_apply(z,t) -> velocity arrays [B,4,32,32] at NULL class,
    boundary r=t. Accumulates per-sample bias_sq/noise across batches per t."""
    rng = np.random.default_rng(seed)
    per_t = {}
    for t in t_grid:
        bias_all, noise_all = [], []
        for _ in range(n_batches):
            x0 = np.asarray(get_latent_batch(), dtype=float)
            e = rng.standard_normal(x0.shape)
            xt = (1.0 - t) * x0 + t * e
            v_cond = e - x0
            u_mf = np.asarray(mf_apply(xt, t), dtype=float)
            v_ref = np.asarray(vref_apply(xt, t), dtype=float)
            ax = tuple(range(1, x0.ndim))  # sum over all but batch
            bias_all.append(np.sum((u_mf - v_ref) ** 2, axis=ax))
            noise_all.append(np.sum((v_cond - v_ref) ** 2, axis=ax))
        per_t[t] = ratio_stats(np.concatenate(bias_all), np.concatenate(noise_all),
                               beta_no_bias=beta_no_bias)
    return per_t


# --------------------------------- self-test -----------------------------------
def _self_test():
    rng = np.random.default_rng(1)
    NOISE = 3090.0  # ~ the v_ref FM-loss floor reported overnight

    def synth(ratio_target, n=6000):
        # noise ~ positive with realistic spread; bias = ratio_target*noise * lognormal jitter
        noise = NOISE * np.exp(rng.normal(0, 0.25, n))
        bias = ratio_target * NOISE * np.exp(rng.normal(0, 0.30, n))
        return bias, noise

    # Regime A: high-d bias dominance (ratio ~ 10) -> beta* collapses
    b, nz = synth(10.0)
    A = ratio_stats(b, nz)
    assert 8 < A["ratio"] < 12, A["ratio"]
    assert A["beta_star"] < 0.15, A["beta_star"]
    assert A["ratio_ci"][0] < A["ratio"] < A["ratio_ci"][1]

    # Regime B: small bias (ratio ~ 0.1) -> beta* stays near 0.94
    b, nz = synth(0.1)
    B = ratio_stats(b, nz)
    assert 0.08 < B["ratio"] < 0.12, B["ratio"]
    assert B["beta_star"] > 0.80, B["beta_star"]

    # beta_no_bias must be 0.94 not 0.5: with ratio->0, beta*->beta_no_bias
    z = ratio_stats(np.full(1000, 1e-9), np.full(1000, NOISE))
    assert abs(z["beta_star"] - 0.94) < 1e-3, z["beta_star"]

    # end-to-end harness plumbing check with fake models (v_ref = u_mf = 0)
    def get_b():
        return rng.standard_normal((64, 4, 8, 8))
    def vref_apply(xt, t):
        return np.zeros_like(xt)                      # v_ref = 0
    def mf_apply(xt, t):
        # with v_ref=0: noise=||v_cond||^2, bias=||u_mf||^2; set u_mf so ratio~4
        return np.zeros_like(xt)                      # bias=0 -> ratio 0 (sanity path)
    per_t = run_eval(get_b, mf_apply, vref_apply, t_grid=[0.3, 0.5, 0.7], n_batches=3)
    for t in per_t:
        assert per_t[t]["ratio"] < 0.05               # u_mf=v_ref=0 -> bias 0
        assert abs(per_t[t]["beta_star"] - 0.94) < 0.02

    print("self-test OK:")
    print(f"  regime A (bias dominance): ratio={A['ratio']:.2f} beta*={A['beta_star']:.3f} "
          f"CI={tuple(round(x,3) for x in A['beta_star_ci'])}")
    print(f"  regime B (small bias):     ratio={B['ratio']:.2f} beta*={B['beta_star']:.3f}")
    print(f"  beta_no_bias check (ratio->0): beta*={z['beta_star']:.3f} (==0.94)")
    print(f"  harness (u_mf=v_ref=0):    beta*~0.94 at all t, ratio~0  [plumbing OK]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--beta_no_bias", type=float, default=0.94)
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        raise SystemExit(
            "Wire mf_apply / vref_apply / get_latent_batch to the real models, then call "
            "run_eval(...) and print_table(...). See the eval interface in the overnight report."
        )
