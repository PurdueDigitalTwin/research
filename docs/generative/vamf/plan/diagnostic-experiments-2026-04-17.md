# Diagnostic Experiments for VaMF Paper

Date: 2026-04-17
Status: In progress

## Objective

Provide empirical evidence for the paper's three theorems without requiring
reproduction of SOTA FID. These experiments measure the *mechanisms* the
theory predicts, not end-to-end generation quality.

## Experiment 1: Jacobian Variance Amplification (Theorem 1)

**Claim:** Per-sample gradient variance is amplified by Tr(J Sigma_v' J^T)
when using stochastic v_cond as JVP tangent, and drops to the irreducible
Tr(Sigma_v') when using a deterministic tangent.

**Method:** Load trained MeanFlow checkpoint (3zhzf3qm, step 800k). For
N=1000 batches of CIFAR-10 data:
1. Sample (x0, x1, t, r) with fixed (t, r) grid: t in {0.1, 0.3, 0.5, 0.7, 0.9}, r=0
2. Compute x_t = (1-t)x0 + t*x1
3. Compute per-sample loss with **stochastic tangent** (v_cond in JVP)
4. Compute per-sample loss with **deterministic tangent** (u_ema(x_t, t, t))
5. Record per-sample loss values

**Output:** For each t, plot histogram of per-sample loss under both tangents.
Report Var[ell_stoch] / Var[ell_determ] ratio. Theorem 1 predicts ratio >> 1,
growing with t (since J grows with (t-r) and Sigma_v' is larger at mid-t).

**Checkpoint:** `gs://pdt_gen_ai/juanwu/meanflow/meanflow_unet_cifar_10_20260412_191003/checkpoints/800000/`

## Experiment 2: Curvature Gap vs Interval Length (Theorem 3)

**Claim:** ||Delta(x_t, r, t)|| ~ (t-r)/2 * ||Dv/Dt||, i.e., the curvature
gap is approximately linear in the interval length.

**Method:** Load same checkpoint. For a fixed batch of 256 CIFAR-10 images:
1. Fix t = 0.5 (mid-trajectory)
2. Vary r in linspace(0, t, 20) to sweep (t-r) from 0 to 0.5
3. Compute u(z, r, t) and v_cond for each (r, t)
4. Compute curvature gap: ||u(z, r, t) - v_cond||^2 (averaged over batch)

**Output:** Plot ||Delta||^2 vs (t-r). Should be approximately quadratic
(since ||Delta|| ~ (t-r), ||Delta||^2 ~ (t-r)^2). Fit a polynomial and
report the leading coefficient as an empirical estimate of ||Dv/Dt||/2.

Also sweep t in {0.2, 0.4, 0.6, 0.8} with r from 0 to t to show the
relationship holds at different trajectory positions.

## Experiment 3: Loss Non-Monotonicity (Semi-Gradient Gap, Theorem 2)

**Claim:** The semi-gradient gap blinds the optimizer to Tr(J Sigma J^T)
growth, causing the total loss to be non-decreasing even as the mean-field
residual decreases.

**Method:** Extract from wandb:
- train/loss trajectory for vanilla MF runs (yekq7lnu, 3zhzf3qm)
- train/velocity_loss trajectory (measures the compound prediction quality)
- train/grad_norm trajectory

**Output:** Multi-panel figure showing:
(a) Loss curve: non-decreasing (as paper claims)
(b) Gradient norm: high variance
(c) FID: decreasing (model IS learning despite non-decreasing loss)

This shows the loss is dominated by the uncontrolled variance term.

## Experiment 4: Jacobian Norm Growth (Supporting Theorem 1)

**Claim:** ||J|| = ||(t-r) d_z u_theta - I|| grows with (t-r), amplifying
the noise term.

**Method:** From the same checkpoint, for a batch of 256 images:
1. Fix t in {0.2, 0.4, 0.6, 0.8}
2. For each t, set r = 0
3. Estimate ||d_z u_theta||_F via Hutchinson trace estimator with 10 random vectors
4. Compute ||(t-r) * estimated_jacobian_norm - d||

**Output:** Plot estimated ||J||_F vs t. Should grow with t.

## Experiment 5: Variance-Diversity Tradeoff (New Hypothesis)

**Claim:** The stochastic JVP tangent implicitly regularizes the Jacobian
and promotes generation diversity. Removing it (as VaMF does) may reduce
diversity.

**Method:** If time permits, from VaMF MSE checkpoint (9k3bt7aa, 800k):
1. Generate 50k samples with EMA params
2. Compute precision and recall (P&R) metrics
3. Compare with vanilla MF samples

This tests whether FID gap is coverage-driven.

## Resource Plan

- **VM:** tpu-v4-32-us-central2-b (4 workers, v4-32)
- **Experiments 1,2,4:** ~30 min each (forward passes only, no training)
- **Experiment 3:** Local (wandb extraction)
- **Experiment 5:** ~2 hours (sample generation)

## Checkpoints Available

| Run | ID | GCS Path | Steps | FID |
|-----|-----|----------|-------|-----|
| Vanilla MF v0 | 3zhzf3qm | meanflow_unet_cifar_10_20260412_191003 | 800k | 5.67 |
| Vanilla MF v1 | 8vcc42dj | meanflow_new_sample_t_and_r_unet_cifar_10_20260415_022837 | 800k | 5.75 |
| Old baseline | yekq7lnu | unet_cifar10_20260103_070257 | 800k | 7.05 |
| VaMF MSE | 9k3bt7aa | vamf_with_snr_unet_cifar_10_20260406_131122 | 800k | 22.17 |
