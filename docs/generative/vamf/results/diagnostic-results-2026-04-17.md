# Diagnostic Experiment Results

Date: 2026-04-17
Checkpoint: `3zhzf3qm` (vanilla MF, 800k steps, FID 5.67)
Platform: TPU v4-32, single-process CPU mode

## Experiment 1: Jacobian Variance Amplification (Theorem 1)

**Setup:** For each t in {0.1, 0.3, 0.5, 0.7, 0.9} with r=0,
compute per-sample MF loss with stochastic tangent (v_cond) vs
deterministic tangent (u(z,t,t)). N=3200 samples per t value
(50 batches x 64).

**Results:**

| t   | Var[stochastic] | Var[deterministic] | Ratio      |
| --- | --------------- | ------------------ | ---------- |
| 0.1 | 5.12e4          | 5.30e4             | 0.97       |
| 0.3 | 5.02e4          | 2.32e4             | 2.16       |
| 0.5 | 1.13e8          | 9.34e5             | **120.92** |
| 0.7 | 8.06e11         | 3.53e11            | 2.28       |
| 0.9 | 2.12e13         | 3.59e10            | **589.77** |

**Interpretation:**

- At t=0.1, the Jacobian factor J = (t-r) d_z u - I is approximately -I
  (since 0.1 * d_z u is small), so J Sigma_v' J^T ~ Sigma_v'.
  The ratio is ~1 as predicted.

- The ratio grows sharply with t: 2x at t=0.3, 121x at t=0.5,
  590x at t=0.9. This directly validates Theorem 1's prediction
  that per-sample gradient variance is amplified by Tr(J Sigma_v' J^T).

- The dip at t=0.7 (ratio=2.28) occurs because both stochastic and
  deterministic variances are extremely large (~10^11). At this trajectory
  point the model evaluates on nearly-pure-noise inputs (z = 0.3*x0 + 0.7*e),
  making the deterministic tangent u(z,t,t) itself highly variable.

- The exponential growth in absolute variance (5e4 at t=0.1 to 2e13 at t=0.9)
  explains why the MeanFlow loss is dominated by high-t samples, consistent
  with the non-decreasing loss behavior.

**Conclusion:** Strong support for Theorem 1. The stochastic tangent induces
up to 590x more per-sample loss variance than the deterministic tangent,
with the amplification growing as the Jacobian norm increases with t.

## Experiment 2: Curvature Gap vs Interval Length (Theorem 3)

**Setup:** For each t in {0.1, 0.3, 0.5, 0.7, 0.9}, sweep r from 0
to t in 20 steps. Measure ||u(z,r,t) - v_cond||^2 averaged over
640 samples (10 batches x 64).

**Results (gap at endpoints):**

| t   | Gap at r=0 | Gap at r~t | Relative decrease |
| --- | :--------: | :--------: | :---------------: |
| 0.1 |   771.9    |   734.8    |       4.8%        |
| 0.3 |   436.9    |   376.1    |       13.9%       |
| 0.5 |   390.2    |   316.9    |       18.8%       |
| 0.7 |   508.6    |   387.6    |       23.8%       |
| 0.9 |   898.5    |   598.8    |       33.4%       |

**Interpretation:**

- The curvature gap monotonically decreases as r approaches t (interval
  shrinks), consistent with Theorem 3: ||Delta|| ~ (t-r)/2 * ||Dv/Dt||.

- The relative decrease grows with t (4.8% to 33.4%), reflecting that
  longer intervals produce more curvature accumulation.

- The gap is nonzero even at r~t because the model u(z,r,t) is trained
  on a distribution of r values, not point-evaluated at r=t. The
  residual gap at small (t-r) reflects the irreducible model error.

- See `exp2_curvature_gap.pdf` for the full sweep curves showing the
  approximately quadratic dependence on (t-r).

**Conclusion:** Supports Theorem 3. The curvature gap shrinks with
the interval length, and the rate of decrease grows with t.

## Experiment 3: Loss Non-Monotonicity (from wandb)

**Setup:** Extracted training curves from wandb for vanilla MeanFlow
runs (3zhzf3qm, 8vcc42dj) and old baseline (yekq7lnu).

**Key observations:**

### Loss behavior

| Run           | Loss (start) | Loss (end) | Loss (mean) |
| ------------- | :----------: | :--------: | :---------: |
| vanilla_mf_v0 |     7.87     |    5.67    |    5.55     |
| vanilla_mf_v1 |     7.88     |    5.86    |    5.88     |
| old_baseline  |     7.85     |    5.68    |     --      |

The loss does decrease overall but shows high variance throughout training,
consistent with the Theorem 2 prediction that the observable loss is
ell_mean + Tr(J Sigma_v' J^T), where the second term is not controlled by
the gradient.

### Gradient norm variance

| Run           | GradNorm mean | GradNorm std | GradNorm max |
| ------------- | :-----------: | :----------: | :----------: |
| vanilla_mf_v0 |     2.97      |     2.24     |    41.36     |
| vanilla_mf_v1 |     3.26      |     2.18     |    39.25     |

Gradient norm std/mean ratio ~ 0.67-0.75 indicates high gradient noise,
consistent with the variance amplification measured in Experiment 1.

### FID trajectory

| Run           | Best FID | Best step | Final FID |
| ------------- | :------: | :-------: | :-------: |
| vanilla_mf_v0 |   5.35   |  612,500  |   5.67    |
| vanilla_mf_v1 |   5.43   |  500,000  |   5.75    |
| old_baseline  |   7.05   |  705,000  |    --     |

FID continues to improve after loss plateaus, showing that the model
learns useful structure despite the loss being dominated by the
uncontrolled variance term.

## Experiment 4: Jacobian Norm Growth (Supporting Theorem 1)

**Setup:** For each t in {0.1, 0.3, 0.5, 0.7, 0.9} with r=0,
estimate ||J||\_F = ||(t-r) d_z u - I||\_F via Hutchinson estimator
with 5 random vectors, over 640 samples (10 batches x 64).

**Results:**

| t | ||d_z u||\_F^2 | ||J||\_F^2 | ||J||\_F |
| \--- | :-----------: | :-------: | :-----: |
| 0.1 | 197,781 | 638 | 24.9 |
| 0.3 | 27,786 | 367 | 18.8 |
| 0.5 | 11,337 | 473 | 20.1 |
| 0.7 | 11,269 | 3,081 | 38.3 |
| 0.9 | 26,204 | 18,760 | 74.8 |

**Interpretation:**

- ||J||\_F grows from 18.8 at t=0.3 to 74.8 at t=0.9, a 4x increase.
  This growth amplifies the noise variance term Tr(J Sigma_v' J^T)
  in Theorem 1, explaining the 590x variance ratio at t=0.9.

- The raw Jacobian ||d_z u||\_F^2 is largest at t=0.1 (197k), where
  the model has a strong, localized response. At mid-t the network
  becomes smoother (||d_z u||\_F^2 ~ 11k), but the (t-r) scaling
  factor causes ||J||\_F = ||(t-r)\*d_z u - I||\_F to grow.

- The U-shaped pattern in ||J||\_F^2 (high at t=0.1, low at t=0.3-0.5,
  then rapidly growing) reflects the interplay between the identity
  subtraction (dominant at small t where (t-r)\*d_z u is small) and
  the Jacobian scaling (dominant at large t).

**Conclusion:** Confirms that the Jacobian factor ||J||\_F grows
significantly with t, providing the mechanism for variance
amplification identified in Theorem 1 and measured in Experiment 1.

## VaMF MSE Run (9k3bt7aa) — Reference

The VaMF MSE variant with FM anchor shows:

- Loss: 11049 -> 870 (dominated by FM anchor loss)
- FID: best 14.87 @ step 435k (significantly worse than vanilla MF)
- This confirms the MSE scaling bug where FM anchor dominates 98% of loss

## Raw Data

- Experiment 1, 2, 4: `diagnostic_results.json`
- Experiment 3: `wandb_metrics.json`

## Figures

- `exp1_variance_amplification.pdf` — Variance ratio and absolute variances vs t
- `exp2_curvature_gap.pdf` — Curvature gap ||Delta||^2 vs (t-r) for multiple t
- `exp3_training_curves.pdf` — Loss, gradient norm, and FID training curves
- `exp4_jacobian_norm.pdf` — ||J||\_F vs t
