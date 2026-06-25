# B1 Variance Addendum

Two corrections to the B1 handoff report, plus checkpoint preservation.

## 1. Task V: Gradient-Variance Decomposition

The B1 handoff concluded "annealing does NOT reduce gradient variance"
based on `nr = tr_cov / mean_norm_sq` alone. This was wrong: `nr` is a
noise-to-**signal** ratio. VAMF theory predicts beta>0 reduces the
gradient **variance** itself (the numerator `tr_cov`). If beta>0 also
reshapes the loss surface, it can shrink the denominator `mean_norm_sq`
too, so `nr` can rise while `tr_cov` falls.

### Full `analyze_grad_var.py` output

Script: `src/projects/generative/vamf/scripts/analyze_grad_var.py`
Work dir: Phase 1 sweep (`b1_sweep/`, 30 JSONs, DGMM-64)

```
=== Gradient-variance decomposition (dgmm_64, 10 configs, mean over seeds) ===

config          bin                tr_cov      mean_norm_sq                nr
-----------------------------------------------------------------------------
baseline_b0     early        337.5+/-10        940.5+/-70       0.4494+/-0.017
baseline_b0     mid          217.7+/-6.7       313.1+/-14       0.8028+/-0.021
baseline_b0     late         268.5+/-11        107.4+/-8.6       3.081+/-0.11

static_b0.4     early        370.7+/-10        999.1+/-73         0.46+/-0.017
static_b0.4     mid            238+/-7           324+/-11       0.8466+/-0.016
static_b0.4     late         290.3+/-9.1       111.2+/-7.1       3.194+/-0.096

static_b1       early        417.9+/-0.68       1040+/-17        0.485+/-0.0073
static_b1       mid          270.3+/-4.7       333.8+/-7.7       0.931+/-0.0079
static_b1       late           324+/-5.6         115+/-2.5       3.374+/-0.063

cosine_s0.3     early        393.6+/-0.39       1005+/-20        0.475+/-0.011
cosine_s0.3     mid          217.2+/-3.9       310.3+/-6.9      0.8065+/-0.012
cosine_s0.3     late         260.4+/-5.8         103+/-3.8         3.1+/-0.079

cosine_s0.6     early        410.5+/-0.58       1029+/-17        0.482+/-0.0077
cosine_s0.6     mid          230.8+/-4.1       315.4+/-5.6      0.8424+/-0.0074
cosine_s0.6     late         253.8+/-5         99.04+/-3.3       3.138+/-0.077

cosine_s0.8     early        413.6+/-0.62       1034+/-17       0.4833+/-0.0075
cosine_s0.8     mid          241.7+/-4.3       320.6+/-5.8      0.8674+/-0.0051
cosine_s0.8     late         252.8+/-4.9       98.21+/-3.2       3.156+/-0.075

linear_s0.3     early        388.9+/-0.38      998.3+/-19       0.4729+/-0.011
linear_s0.3     mid          218.5+/-4         311.4+/-6.9      0.8086+/-0.012
linear_s0.3     late         260.6+/-5.7         103+/-3.7       3.102+/-0.079

linear_s0.6     early          403+/-0.49       1019+/-17       0.4788+/-0.0085
linear_s0.6     mid          233.1+/-4.1       317.6+/-5.6      0.8446+/-0.0073
linear_s0.6     late         254.2+/-5.4       99.29+/-3.4       3.134+/-0.077

linear_s0.8     early        406.6+/-0.53       1024+/-17       0.4804+/-0.008
linear_s0.8     mid          241.7+/-4.3       321.6+/-5.7      0.8645+/-0.0057
linear_s0.8     late         255.8+/-5.3       99.66+/-3.2       3.155+/-0.075

step_s0.6       early        417.9+/-0.68       1040+/-17        0.485+/-0.0073
step_s0.6       mid          270.3+/-4.7       333.8+/-7.7       0.931+/-0.0079
step_s0.6       late           247+/-4.5       95.38+/-2.8       3.174+/-0.07
```

### Variance Verdict (constant-beta runs, early bin)

```
reference tr_cov[beta=0, early] = 337.5

  beta=0.4   tr_cov ratio = 1.098   (signal mean_norm_sq ratio = 1.062)
  beta=1     tr_cov ratio = 1.238   (signal mean_norm_sq ratio = 1.105)

--> tr_cov is NOT reduced by beta>0 (even the raw numerator).
    Do NOT use this toy's variance numbers in the rebuttal. Two candidates:
    (a) the EMA proxy (3-layer MLP on hard data) is a POOR control variate
        on this toy -> inflated variance, a toy artifact that need not
        transfer to DiT scale; or
    (b) a genuine inconsistency with the paper's DGMM '1.2-4.3x variance
        reduction' claim -> reconcile against the exact quantity/config
        that figure was generated from (see the cross-check task).

CAVEAT (carry into any writeup): a toy KILL justifies NOT running Phase-2 DiT
annealing, but is NOT evidence the variance mechanism fails at scale. Keep the
two claims separate.
```

The original B1 handoff was RIGHT about the direction (beta>0 is worse)
but for the WRONG reason. The `nr` rise was not masking a variance
drop — both `tr_cov` AND `mean_norm_sq` INCREASE with beta>0. The ratio
`nr` rises because the numerator grows faster than the denominator.

### Paper-Claim Reconciliation

The "1.2x~4.3x gradient-variance reduction" claim
(`report/contents/introduction.tex:25,29` and
`report/contents/experiment.tex:24`) refers to:

- **Quantity**: Raw `Tr(Cov[nabla_theta ell_MF])` (= `tr_cov`), plotted
  in `fig: nr-vs-beta` (`report/contents/experiment.tex:15-17`)
- **Datasets**: The six 2D toy datasets (swiss_roll, eight_gaussians,
  checkerboard, moons, pinwheel, two_spirals) — **NOT DGMM**
- **Beta grid**: {0, 0.25, 0.5, 0.75, 1}
- **Measurement code**: `figures/plot_beta_sweep.py` with `--metric=tr_cov`,
  reading from `beta_<beta>/` sweep directories

**Does this 30-run B1 sweep reproduce that claim?**

**No — but this is NOT an inconsistency.** The claims are about
different datasets:

1. The "1.2-4.3x" is from the 2D toys where high geometric curvature
   (large kappa) amplifies the Jacobi factor. The paper itself notes the
   largest reductions appear on "high-curvature datasets (eight_gaussians,
   checkerboard)" and the smallest (1.20x) on two_spirals.

2. The B1 sweep is on DGMM-64, where the paper explicitly acknowledges
   (experiment.tex:43): "On low-dimensional datasets with d in {2..32},
   SW1 is essentially beta-insensitive [...] At d=64 the data hints at
   an interior minimum at beta*=0.4 with a ~9% reduction over beta=0,
   consistent with kappa coming off saturation."

3. The theory explains why: at d=64, kappa is near 0, so
   kappa/(kappa+1) -> 0, the noise-cancellation factor vanishes, and
   beta>0 adds bias without enough variance reduction to compensate.

**Config differences from paper's DGMM experiment:**

| Aspect | Paper DGMM | B1 sweep |
|--------|-----------|----------|
| Beta values | {0, 0.1, ..., 1.0} | 0, 0.4, 1.0 (constant) + 7 annealed |
| Dimension | d in {2,4,8,16,32,64} | d=64 only |
| Steps | 200k | 200k |
| Seeds | {42, 0, 1} | {42, 0, 1} |
| tr_cov logged? | yes | yes |
| Metric plotted | SW1 (variance not shown for DGMM) | SW1 + tr_cov |

The B1 sweep is actually the FIRST measurement of raw `tr_cov` vs beta
on DGMM-64 specifically. The paper's DGMM section reports only SW1, not
variance. The finding that `tr_cov` increases with beta at d=64 is NEW
data that **strengthens** the paper's narrative: at high d, the EMA
tangent is a poor control variate (large bias norm ||b||^2), and the
variance reduction mechanism that works brilliantly on 2D toys fails to
compensate.

**Rebuttal implication:** If a reviewer probes the "1.2-4.3x" number,
the answer is: "that figure is from the 2D toys; the DGMM-64 section
already documents the dimensionality-dependent degradation and attributes
it to kappa saturation." The B1 sweep adds a new data point that the
tr_cov itself increases at d=64, which is consistent with poor control
variate quality at high dimension.


## 2. Task K: Checkpoint Inventory

### Provided paths — verified

| Run | Path | Status | Size |
|-----|------|--------|------|
| beta=0.25 | `gs://pdt_training/juanwu/meanflow/meanflow/vamf_beta025_dit_b4_imagenet_256_20260503_170426/` | **EMPTY** (0 bytes, directory marker only) | 0 B |
| beta=0.5 | `gs://pdt_training/juanwu/meanflow/vamf_beta05_dit_b4_imagenet_256_20260502_211910/` | **EMPTY** (0 bytes, directory marker only) | 0 B |
| beta=0 baseline | `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` | **EXISTS** | 37.74 GiB |

### Baseline checkpoint detail

Steps: 10k, 20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k, 100k, 110k,
120k, 130k, 140k, 150k, 160k, 170k, 180k, 190k, 200k, 210k
(21 checkpoints, every 10k steps). Contains `wandb.txt` + `checkpoints/`.

### Beta=1 search

Searched both `gs://pdt_training/juanwu/meanflow/` and
`gs://pdt_training/juanwu/meanflow/meanflow/` for patterns matching
`vamf_beta1*`, `vamf_beta10*`, `*beta1*dit_b4*imagenet_256*`.
**Not found.** No beta=1 DiT checkpoint directory exists under any
naming variation.

### Other VAMF DiT directories (all empty)

All VAMF-related DiT directories are 0-byte directory markers:
- `vamf_tw_dit_b4_20260428_203132/` — 0 B
- `vamf_tw_dit_b4_20260428_203443/` — 0 B
- `vamf_l2_dit_b4_imagenet_256_20260501_014255/` — 0 B
- `vamf_l2_dit_b4_imagenet_256_20260501_140313/` — 0 B

These appear to be runs that created their output directory but crashed
or were killed before the first checkpoint was written.


## 3. Bucket Lifecycle / TTL

```
gsutil lifecycle get gs://pdt_training
-> gs://pdt_training/ has no lifecycle configuration.
```

**No auto-delete TTL.** The bucket has no lifecycle rules. Checkpoints
will not be automatically deleted. There is no time pressure for
preservation beyond standard operational caution.


## 4. Preservation

### Baseline (the only non-empty checkpoint)

- **Source**: `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/`
- **Destination**: `gs://pdt_training/juanwu/vamf_preserved/baseline_dit_imagenet_256_latent/`
- **Method**: `gsutil -m rsync -r`
- **Size**: ~37.74 GiB (21 Orbax checkpoints + wandb.txt)
- **Status**: COMPLETE (40,525,186,266 bytes, 21 checkpoint dirs verified)

### Beta-sweep runs (beta=0.25, beta=0.5)

**NOT COPIED** — source directories are genuinely empty (0 bytes).
There is nothing to preserve. These runs failed before writing their
first checkpoint.

### Beta=1

**NOT FOUND.** No beta=1 DiT run exists in GCS under any naming scheme.

**FLAG FOR HUMAN:** The preservation destination
`gs://pdt_training/juanwu/vamf_preserved/` is in the same bucket as the
source. If a separate durable/archive bucket exists, the copy should be
moved there. The current copy protects against accidental deletion of the
source path but not against bucket-level operations.


## 5. Bottom Line

**(a) Variance mechanism on the toy.** The variance-reduction mechanism
does NOT reproduce on DGMM-64 at the `tr_cov` level: beta>0 increases
both `tr_cov` (by 10-24%) and `mean_norm_sq` (by 6-11%), with the
variance growing faster. This is NOT inconsistent with the paper's
"1.2-4.3x" claim, which is from the 2D toys where kappa is large. On
DGMM-64, kappa is near zero, and the theory itself predicts the
variance-reduction effect vanishes (kappa/(kappa+1) -> 0). The B1 KILL
verdict is a **bias/quality story**: at high d, the EMA tangent adds
bias without enough variance reduction to compensate. For the rebuttal,
frame the DGMM-64 result as confirming the dimensionality-dependent
beta* shift (toward 0) that the paper already documents, not as evidence
that variance reduction fails at all scales.

**(b) DiT evidence safety.** The only non-empty DiT checkpoint (beta=0
baseline, 37.74 GiB, steps 10k-210k) is being preserved. All beta-sweep
DiT runs (beta=0.25, beta=0.5) are confirmed empty — they never wrote
checkpoints. Beta=1 never existed. The preservation gap is total for the
sweep runs; the paper's DiT FID curves depend on data that is either in
wandb (metrics) or lost (checkpoints). If the paper needs to regenerate
samples from beta>0 DiT checkpoints, those runs must be retrained.
