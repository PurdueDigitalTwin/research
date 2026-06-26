# FID Ordering Evidence, Dual-Baseline Reconciliation, and Authoritative Run Map

Audit date: 2026-06-26. Zero TPU, zero training, no edits to paper.


## 1. Per-Step FID Ordering (Task 1)

Script: `src/projects/generative/vamf/scripts/fid_ordering.py`
Input: wandb FID histories from the canonical FID-logging worker per job.

### Run with `fidpdet7` as beta=0 (the only viable baseline)

Matched-step horizon: 59 steps at 5k intervals (5k-295k), all 4 betas present.

**IMPORTANT: wandb vs paper discrepancy.** The wandb-logged FID values
for beta=0 (`fidpdet7`) differ from the paper's Table S.4 (`dit-fid.tex`)
by up to +1.87 FID (wandb values consistently higher). Beta>0 values
match the paper almost exactly (< 0.05 FID difference). See Section 2
for analysis. The per-step table below uses the **wandb-logged values**.

```
=== Per-step 4-point FID ordering (59 matched steps) ===
predicted (paper) ordering: beta 0 < 0.25 < 0.5 < 1

    step     b=0.0    b=0.25     b=0.5     b=1.0   ordering             strict?
-------------------------------------------------------------------------------
    5000   355.322   348.442   349.192   347.655   1.0<0.25<0.5<0.0     NO
   10000   363.961   502.252   525.643   528.849   0.0<0.25<0.5<1.0     yes
   15000   303.982   353.005   366.584   358.361   0.0<0.25<1.0<0.5     NO
   20000   213.297   236.889   241.451   239.350   0.0<0.25<1.0<0.5     NO
   25000   159.956   174.539   181.384   176.346   0.0<0.25<1.0<0.5     NO
   30000   126.529   138.466   145.170   148.809   0.0<0.25<0.5<1.0     yes
   35000    98.833   109.635   116.219   123.901   0.0<0.25<0.5<1.0     yes
   40000    78.432    88.390    94.638   104.077   0.0<0.25<0.5<1.0     yes
   45000    64.294    72.691    78.694    90.125   0.0<0.25<0.5<1.0     yes
   50000    55.077    61.563    66.879    79.886   0.0<0.25<0.5<1.0     yes
   55000    48.398    53.203    57.731    71.861   0.0<0.25<0.5<1.0     yes
   60000    43.103    46.683    50.810    65.691   0.0<0.25<0.5<1.0     yes
   65000    39.064    41.355    45.147    60.725   0.0<0.25<0.5<1.0     yes
   70000    35.879    37.024    40.598    56.474   0.0<0.25<0.5<1.0     yes
   75000    32.925    33.245    36.744    52.929   0.0<0.25<0.5<1.0     yes
   80000    30.587    30.396    33.667    49.896   0.25<0.0<0.5<1.0     NO
   85000    28.408    28.092    31.121    47.264   0.25<0.0<0.5<1.0     NO
   90000    26.908    26.373    29.034    45.144   0.25<0.0<0.5<1.0     NO
   95000    25.271    24.733    27.124    43.379   0.25<0.0<0.5<1.0     NO
  100000    23.868    23.334    25.546    41.714   0.25<0.0<0.5<1.0     NO
  105000    22.736    22.274    24.215    40.419   0.25<0.0<0.5<1.0     NO
  110000    21.705    21.218    23.068    39.164   0.25<0.0<0.5<1.0     NO
  115000    20.776    20.322    22.175    38.007   0.25<0.0<0.5<1.0     NO
  120000    20.040    19.568    21.250    36.891   0.25<0.0<0.5<1.0     NO
  125000    19.289    18.821    20.496    36.003   0.25<0.0<0.5<1.0     NO
  130000    18.696    18.256    19.728    35.209   0.25<0.0<0.5<1.0     NO
  135000    18.250    17.700    19.129    34.296   0.25<0.0<0.5<1.0     NO
  140000    17.714    17.183    18.578    33.447   0.25<0.0<0.5<1.0     NO
  145000    17.201    16.668    18.066    32.714   0.25<0.0<0.5<1.0     NO
  150000    16.731    16.209    17.599    32.084   0.25<0.0<0.5<1.0     NO
  155000    16.221    15.803    17.021    31.425   0.25<0.0<0.5<1.0     NO
  160000    15.743    15.461    16.625    30.756   0.25<0.0<0.5<1.0     NO
  165000    15.355    15.090    16.250    30.364   0.25<0.0<0.5<1.0     NO
  170000    15.042    14.787    15.941    29.777   0.25<0.0<0.5<1.0     NO
  175000    14.707    14.500    15.657    29.347   0.25<0.0<0.5<1.0     NO
  180000    14.482    14.208    15.410    29.058   0.25<0.0<0.5<1.0     NO
  185000    14.220    13.960    15.165    28.723   0.25<0.0<0.5<1.0     NO
  190000    13.956    13.755    14.976    28.399   0.25<0.0<0.5<1.0     NO
  195000    13.687    13.589    14.672    27.981   0.25<0.0<0.5<1.0     NO
  200000    13.525    13.344    14.399    27.605   0.25<0.0<0.5<1.0     NO
  205000    13.310    13.213    14.195    27.280   0.25<0.0<0.5<1.0     NO
  210000    13.125    13.056    13.947    27.165   0.25<0.0<0.5<1.0     NO
  215000    13.047    12.858    13.862    26.971   0.25<0.0<0.5<1.0     NO
  220000    12.802    12.708    13.757    26.633   0.25<0.0<0.5<1.0     NO
  225000    12.671    12.545    13.537    26.302   0.25<0.0<0.5<1.0     NO
  230000    12.536    12.409    13.434    26.024   0.25<0.0<0.5<1.0     NO
  235000    12.466    12.327    13.376    25.700   0.25<0.0<0.5<1.0     NO
  240000    12.361    12.312    13.335    25.470   0.25<0.0<0.5<1.0     NO
  245000    12.315    12.226    13.267    25.256   0.25<0.0<0.5<1.0     NO
  250000    12.275    12.114    13.165    25.101   0.25<0.0<0.5<1.0     NO
  255000    12.061    11.940    13.024    24.726   0.25<0.0<0.5<1.0     NO
  260000    11.982    11.860    12.939    24.474   0.25<0.0<0.5<1.0     NO
  265000    11.919    11.834    12.863    24.299   0.25<0.0<0.5<1.0     NO
  270000    11.807    11.797    12.742    24.185   0.25<0.0<0.5<1.0     NO
  275000    11.774    11.743    12.678    24.106   0.25<0.0<0.5<1.0     NO
  280000    11.671    11.765    12.608    23.933   0.0<0.25<0.5<1.0     yes
  285000    11.582    11.763    12.537    23.811   0.0<0.25<0.5<1.0     yes
  290000    11.571    11.730    12.505    23.553   0.0<0.25<0.5<1.0     yes
  295000    11.533    11.764    12.515    23.356   0.0<0.25<0.5<1.0     yes
```

**Strict ordering: 15/59 (25%).** Within the paper's claimed window
(30k-295k, 54 steps): **14/54 strict.**

### Crossover verdict

```
longest window where 0.25<0 : steps 80000..275000  (40 consecutive matched steps)
mean advantage in window     : 0.275 FID   (max 0.550)
local noise scale (sigma)    : 20.734 FID   -> advantage/noise = 0.01x

--> AMBIGUOUS: sustained but within the noise scale.
```

The script's verdict is AMBIGUOUS because the rolling-median noise
estimator is dominated by the steep early-training FID decay (from
~350 to ~12), yielding sigma=20.7 — far too large for judging a 0.3
FID gap. **The noise scale is an artifact of the estimator, not the
physics.** To interpret correctly:

1. The beta=0.25 advantage over beta=0 is **real and sustained**: 40
   consecutive matched steps, with the gap slowly shrinking from 0.55
   (step 90k) to 0.01 (step 270k), then reversing at step 280k.
2. The gap magnitude (0.1-0.5 FID) is **within typical FID evaluation
   noise** for a single 50k-sample evaluation, so the per-step
   advantage at any INDIVIDUAL step is not statistically significant.
3. The 40-step SUSTAINED window is too long to be random jitter —
   it's a systematic trend.
4. **Interpretation consistent with the bias-variance account:** at
   beta=0.25, variance reduction gives a slight mid-training FID
   advantage. At convergence (280k+), the accumulated bias dominates
   and beta=0 retakes the lead. This is exactly the paper's own
   prediction.

### Does the verdict depend on which beta=0 baseline?

The first baseline (`77ncnhsm`, crashed at ~215k, FID=60 at 200k) is
a diverged run. Using it makes the script say "SYSTEMATIC" but that's
meaningless — it's comparing against a broken run. **Only `fidpdet7`
is usable.** The verdict does NOT change with baseline choice because
there is no viable alternative.


## 2. Dual-Baseline Reconciliation (Task 2)

### Side-by-side inventory

| Attribute | First baseline | Second baseline |
|-----------|---------------|-----------------|
| GCS path | `.../dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` | `.../dit_b4_latent_20260427_045409/` |
| Recursive size | 37.74 GiB | 54.07 GiB |
| Step range | 10k-210k | 10k-300k |
| #Checkpoints | 21 | 30 |
| wandb.txt | `f5fffn99` | `fidpdet7` |
| wandb state | crashed | finished |
| FID-logging worker | `77ncnhsm` (crashed, FID to 215k only) | `fidpdet7` (finished, FID to 295k) |
| Final FID | 59.84 (at 215k) | 11.53 (at 295k) |
| tangent_beta | default (0.0) | default (0.0) |
| exp_name | `dit_imagenet_256_latent` | `dit_b4_latent_20260427_045409` |
| #wandb runs | 52 (all crashed/failed) | 4 (all finished) |

### Which prior reports referred to which

- The **B1 handoff** and **B1 variance addendum** cited 37.74 GiB / 21
  checkpoints — this is the **first** baseline.
- The **checkpoint audit** cited 54.07 GiB / 30 checkpoints — this is
  the **second** baseline.
- The checkpoint audit's **preservation** copied the first baseline
  (37.74 GiB) to `vamf_preserved/baseline_dit_imagenet_256_latent/`.

### Which baseline did the paper use?

The paper's FID table (`dit-fid.tex`) reports beta=0 at step 295k =
11.37 and all 4 runs completing 300k steps. Only the second baseline
(`fidpdet7`) reaches step 295k. The first baseline crashed long before.

**However**, the paper's beta=0 FID values do NOT exactly match the
`fidpdet7` wandb log:

| Step | Paper beta=0 | wandb fidpdet7 | Difference |
|------|-------------|----------------|------------|
| 30k | 128.9 | 126.5 | -2.4 |
| 50k | 54.5 | 55.1 | +0.6 |
| 100k | 22.0 | 23.9 | +1.9 |
| 150k | 15.5 | 16.7 | +1.2 |
| 200k | 13.0 | 13.5 | +0.5 |
| 250k | 11.9 | 12.3 | +0.4 |
| 295k | 11.37 | 11.53 | +0.2 |

For beta=0.25/0.5/1.0, the paper matches wandb within 0.05 FID. The
beta=0 divergence (wandb consistently higher, especially mid-training)
suggests the paper's beta=0 FID values were produced by a **separate
offline re-evaluation** of the same checkpoints, not by reading the
wandb in-training log directly. Both use 50k samples (`experiment.py`
line 200), but different random seeds for sample generation would
produce this level of variation.

**This matters because the paper's "every matched-step" claim depends
on the beta=0 values being LOWER than beta=0.25 at every step.** With
the wandb-logged values, beta=0.25 beats beta=0 at 40/54 steps in the
paper's window. With the paper's published values, the strict ordering
holds. The source of the paper's beta=0 values must be identified.

### Recommendation

The **second baseline** (`fidpdet7`, `dit_b4_latent_20260427_045409`)
is the canonical beta=0. It is the only finished run with 300k steps.
But the paper's FID table uses re-evaluated values for beta=0 that
differ from this run's wandb log. **The human must identify where the
paper's FID JSON came from** — it was likely generated by a separate
offline evaluation script and saved to a local file that was passed to
`plot_dit_fid_curves.py --fid_json <path>`.


## 3. Authoritative Run Map (Task 3)

All cross-verified bidirectionally (wandb config -> GCS path, GCS
`wandb.txt` -> wandb run ID).

### beta=0 (second baseline, CANONICAL)

| Field | Value |
|-------|-------|
| GCS path | `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/` |
| Preserved at | `gs://pdt_training/juanwu/vamf_preserved/baseline_dit_imagenet_256_latent/` (first baseline copy only) |
| wandb.txt | `fidpdet7` |
| Config beta | default (0.0), no tangent_beta, no ema_tangent |
| Method | MeanFlowDiTModel (vanilla MeanFlow) |
| exp_name | `dit_b4_latent_20260427_045409` |
| Size | 54.07 GiB |
| Steps | 10k-300k (30 checkpoints) |
| All workers | `fidpdet7` (w0, FID), `rwxjrbh8` (w1), `qacd8zvy` (w2), `0yf65ai6` (w3) |
| FID worker | `fidpdet7` |
| Final FID | 11.53 (wandb) / 11.37 (paper table) |

### beta=0 (first baseline, CRASHED)

| Field | Value |
|-------|-------|
| GCS path | `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` |
| Preserved at | `gs://pdt_training/juanwu/vamf_preserved/baseline_dit_imagenet_256_latent/` |
| wandb.txt | `f5fffn99` |
| Config beta | default (0.0) |
| exp_name | `dit_imagenet_256_latent` |
| Size | 37.74 GiB |
| Steps | 10k-210k (21 checkpoints) |
| FID worker | `77ncnhsm` (crashed, FID 59.84 at 215k) |
| Status | CRASHED / DIVERGED — not usable |

### beta=0.25

| Field | Value |
|-------|-------|
| GCS path | `gs://pdt_training/juanwu/meanflow/meanflow/vamf_beta025_dit_b4_imagenet_256_20260503_170426/` |
| Preserved at | `gs://pdt_training/juanwu/vamf_preserved/vamf_beta025_dit_b4_imagenet_256/` |
| wandb.txt | `08i3wa20` |
| Config beta | tangent_beta=0.25 |
| exp_name | `vamf_beta025_dit_b4_imagenet_256_20260503_170426` |
| Size | 54.14 GiB |
| Steps | 10k-300k (30 checkpoints) |
| All workers | `884l3avm` (FID), `08i3wa20`, `iksm3wyz`, `p9lhdisy` |
| FID worker | `884l3avm` |
| Final FID | 11.76 |

### beta=0.5

| Field | Value |
|-------|-------|
| GCS path | `gs://pdt_training/juanwu/meanflow/vamf_beta05_dit_b4_imagenet_256_20260502_211910/` |
| Preserved at | `gs://pdt_training/juanwu/vamf_preserved/vamf_beta05_dit_b4_imagenet_256/` |
| wandb.txt | `q8sq19s6` |
| Config beta | tangent_beta=0.5 |
| exp_name | `vamf_beta05_dit_b4_imagenet_256_20260502_211910` |
| Size | 54.14 GiB |
| Steps | 10k-300k (30 checkpoints) |
| All workers | `milo2x6t` (FID), `q8sq19s6`, `1h0rxlfz`, `3ouv2oxg` |
| FID worker | `milo2x6t` |
| Final FID | 12.51 |

### beta=1.0

| Field | Value |
|-------|-------|
| GCS path | `gs://pdt_training/juanwu/meanflow/vamf_l2_dit_b4_imagenet_256_20260501_140313/` |
| Preserved at | `gs://pdt_training/juanwu/vamf_preserved/vamf_l2_dit_b4_imagenet_256/` |
| wandb.txt | `7ku9ef6z` |
| Config beta | ema_tangent=True (-> tangent_beta=1.0) |
| exp_name | `vamf_l2_dit_b4_imagenet_256_20260501_140313` |
| Size | 54.26 GiB |
| Steps | 10k-300k (30 checkpoints) |
| All workers | `7ku9ef6z`, `9uvcq5fw`, `f0ozsd25`, `hrnr0uvz` (FID) |
| FID worker | `hrnr0uvz` |
| Final FID | 23.36 |


## 4. Open Items for the Human

**(a) Reword the ordering sentence in experiment.tex.**

The supplementary text (line 366) claims:
> "the empirical four-point ordering ... holds at every 5k-aligned
> matched-step checkpoint we logged (54 consecutive datapoints from
> step 30k to step 295k)"

With the wandb-logged values, the strict ordering holds at **14/54
steps** in that window, not 54/54. Beta=0.25 beats beta=0 from step
80k to step 275k (40 consecutive steps, 0.1-0.5 FID gap). The 3-point
sub-ordering (0.25 < 0.5 < 1.0) holds at 52/54 steps.

The evidence supports a reword along the lines of: "the three-point
sub-ordering FID(0.25) < FID(0.5) < FID(1) holds at 52/54 steps from
30k to 295k; the full four-point ordering re-establishes at convergence
(step 280k+), with a mid-training window where beta=0.25 slightly
outperforms the unbiased baseline, consistent with the bias-variance
account."

**Caveat:** The paper's published FID table uses different beta=0 values
from the wandb log (see Section 2). If those values came from a
legitimate re-evaluation with a different sample seed, the "every
matched-step" claim may hold for THAT evaluation. You need to identify
the source.

**(b) Pick the canonical beta=0.**

Only one viable option: `dit_b4_latent_20260427_045409` (`fidpdet7`).
The first baseline diverged. But note: the second baseline's wandb FID
values are NOT what the paper table reports for beta=0. Identify the
paper's FID JSON source.

**(c) Preserve the second baseline.**

The checkpoint audit preserved only the first baseline (37.74 GiB) to
`vamf_preserved/`. The second baseline (54.07 GiB, the actual
canonical one) has NOT been preserved. This should be copied:
```
gsutil -m rsync -r \
  gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/ \
  gs://pdt_training/juanwu/vamf_preserved/dit_b4_latent_baseline/
```
