# Beta=0 Provenance: Verified Proof + Seed Robustness

Audit date: 2026-06-26. Zero TPU, zero FID re-evaluation, no paper edits.


## Task A — Provenance Proof

### A1. Eight-step match table

Paper-reported beta=0 FID vs wandb run `hde9iaqj` (the behavior run),
at all 8 steps tabulated in `dit-fid.tex`:

| Step | Paper beta=0 | `hde9iaqj` wandb | Difference |
|------|-------------|-------------------|------------|
| 30k | 128.9 | 128.942 | +0.042 |
| 50k | 54.5 | 54.460 | -0.040 |
| 100k | 22.0 | 21.994 | -0.006 |
| 150k | 15.5 | 15.524 | +0.024 |
| 200k | 13.0 | 12.955 | -0.045 |
| 250k | 11.9 | 11.898 | -0.002 |
| 275k | 11.6 | 11.634 | +0.034 |
| 295k | 11.37 | 11.370 | +0.000 |

Max |diff| = 0.045 FID. All diffs consistent with rounding the wandb
value to 1 decimal place. **Match confirmed.**


### A2. Pairing evidence (matched-pair, not post-hoc pick)

#### Complete enumeration of beta=0 MeanFlowDiT training runs

182 wandb runs across 18 distinct `exp_name` groups. Only 2 reached a
healthy completion (FID < 15 at 295k):

| # | exp_name | FID worker | Final FID | Status | Notes |
|---|----------|-----------|-----------|--------|-------|
| 1 | `mf_dit_imagenet_256_20260418` | (none) | N/A | 0fin/4crash | Early prototype, all crashed |
| 2 | `dit_imagenet_256_20260418_014439` | (none) | N/A | 0fin/28crash | 7 retry rounds, all crashed |
| 3 | `dit_imagenet_256_20260418_021638` | (none) | N/A | 0fin/4crash | Crashed |
| 4 | `dit_imagenet_256_20260418_024833` | (none) | N/A | 0fin/4crash | Crashed |
| 5 | `dit_imagenet_256_20260418_034744` | (none) | N/A | 0fin/2crash | Crashed |
| 6 | `dit_imagenet_256_20260418_134459` | (none) | N/A | 0fin/4crash | Crashed |
| 7 | `dit_imagenet_256_20260418_134905` | (none) | N/A | 0fin/20crash | 5 retry rounds, all crashed |
| 8 | `dit_imagenet_256_20260418_150554` | (none) | N/A | 0fin/4crash | Crashed |
| 9 | `dit_imagenet_256_latent` | `77ncnhsm` | 59.84 | 0fin/52crash | First baseline, diverged at ~215k |
| 10 | `dit_b2_cfg_20260422_181053` | `2gc724qw` | 106.60 | 0fin/4crash | DiT-B/2 variant, diverged |
| 11 | `dit_b4_cfg_20260424_124400` | `mbxtfeiw` | 354.82 | 0fin/17crash | Crashed early |
| 12 | `dit_b4_cfg_20260424_171210` | (none) | N/A | 0fin/4crash | Crashed |
| 13 | `dit_b4_cfg_20260424_171330` | `aewwmcwq` | 107.60 | 4fin/3crash | Third baseline, training diverged (FID plateau ~107) |
| **14** | **`dit_b4_latent_20260427_045409`** | **`fidpdet7`** | **11.53** | **4fin/0crash** | **Independent replication** |
| 15 | `vamf_tw_dit_b4_20260428_203132` | (none) | N/A | 0fin/4crash | Crashed |
| 16 | `vamf_tw_dit_b4_20260428_203443` | `ymxp1xfg` | 30.96 | 0fin/4crash | Crashed early |
| 17 | `meanflow_dit_b4_...20260501_015438_behavior` | `gzk44puc` | 43.10 | 0fin/12crash | Earlier behavior attempt, crashed |
| **18** | **`meanflow_dit_b4_...20260501_140313_behavior`** | **`hde9iaqj`** | **11.37** | **1fin/3crash** | **Paper's baseline** |

Only runs #14 (`fidpdet7`, FID=11.53) and #18 (`hde9iaqj`, FID=11.37)
reached healthy convergence at 300k steps.

#### Matched-pair evidence for `hde9iaqj`

1. **Shared timestamp:** The behavior run's exp_name
   (`meanflow_dit_b4_imagenet_256_**20260501_140313**_behavior`)
   shares the identical `20260501_140313` timestamp with the beta=1 run
   (`vamf_l2_dit_b4_imagenet_256_**20260501_140313**`). Both were
   started on 2026-05-01 at 14:03:13 UTC.

2. **Naming convention:** The `_behavior` suffix is the project's
   convention for the beta=0 control that accompanies a VaMF
   experiment. The beta=1 run uses `vamf_l2_` prefix; the control uses
   `meanflow_` prefix with `_behavior` suffix.

3. **Same GCS parent:** Both live under
   `gs://pdt_training/juanwu/meanflow/`:
   - `meanflow_dit_b4_imagenet_256_20260501_140313_behavior/`
   - `vamf_l2_dit_b4_imagenet_256_20260501_140313/`

4. **Figure script:** `plot_dit_fid_curves.py` uses
   `--baseline_floor=12.12`, which matches the behavior run's FID
   plateau (12.12 at step 235k), NOT `fidpdet7`'s (12.47 at the same
   step).

5. **NOT a post-hoc pick:** The only other healthy beta=0 run
   (`fidpdet7`) was started on 2026-04-27 — 4 days BEFORE the beta=1
   run (2026-05-01). It was NOT launched as a matched pair with the
   beta sweep; it is an earlier independent run. The behavior run WAS
   launched as a matched pair.


### A3. Field-by-field protocol table

All fields extracted from wandb `run.config` via regex. `experiment.py`
line 462 confirms in-training eval uses `state.ema_params`; line 200
confirms 50k samples.

| Field | beta=0 (`hde9iaqj`) | beta=0.25 (`884l3avm`) | beta=0.5 (`milo2x6t`) | beta=1 (`hrnr0uvz`) |
|-------|--------------------|-----------------------|----------------------|---------------------|
| **tangent_beta** | default (0.0) | 0.25 | 0.5 | default (0.0) via ema_tangent=True -> 1.0 |
| **ema_tangent** | default (False) | default (False) | default (False) | True |
| **adaptive_weight_power** | 1.0 | 1.0 | 1.0 | **0.0** |
| cfg_omega | 1.0 | 1.0 | 1.0 | 1.0 |
| cfg_kappa | 0.5 | 0.5 | 0.5 | 0.5 |
| epsilon | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| timestamp_cond | t_and_t_minus_r | t_and_t_minus_r | t_and_t_minus_r | t_and_t_minus_r |
| timestamp_sampler | logit-normal | logit-normal | logit-normal | logit-normal |
| timestamp_sampler_kwargs | mean=-0.4, std=1.0 | mean=-0.4, std=1.0 | mean=-0.4, std=1.0 | mean=-0.4, std=1.0 |
| timestamp_overlap_rate | 0.75 | 0.75 | 0.75 | 0.75 |
| norm_eps | 0.01 | 0.01 | 0.01 | 0.01 |
| in_channels | 4 | 4 | 4 | 4 |
| image_size | 32 | 32 | 32 | 32 |
| features | 768 | 768 | 768 | 768 |
| patch_size | 4 | 4 | 4 | 4 |
| depth | 12 | 12 | 12 | 12 |
| num_heads | 12 | 12 | 12 | 12 |
| ffn_ratio | 4 | 4 | 4 | 4 |
| dropout_rate | 0.0 | 0.0 | 0.0 | 0.0 |
| num_classes | 1000 | 1000 | 1000 | 1000 |
| class_dropout_prob | 0.1 | 0.1 | 0.1 | 0.1 |
| vae_path | pcuenq/sd-vae-ft-mse-flax | pcuenq/sd-vae-ft-mse-flax | pcuenq/sd-vae-ft-mse-flax | pcuenq/sd-vae-ft-mse-flax |
| vae_scaling_factor | 0.18215 | 0.18215 | 0.18215 | 0.18215 |
| **Params for eval** | **EMA** | **EMA** | **EMA** | **EMA** |
| **EMA decay** | **0.9999** | **0.9999** | **0.9999** | **0.9999** |
| **#samples (FID)** | **50,000** | **50,000** | **50,000** | **50,000** |
| **Reference stats** | ILSVRC/imagenet-1k train | ILSVRC/imagenet-1k train | ILSVRC/imagenet-1k train | ILSVRC/imagenet-1k train |
| **Reference revision** | 49e2ee26f381... | 49e2ee26f381... | 49e2ee26f381... | 49e2ee26f381... |
| eval_every_n_steps | 5000 | 5000 | 5000 | 5000 |
| num_train_steps | 300000 | 300000 | 300000 | 300000 |

**Sampler / NFE:** MeanFlow uses a single forward pass (NFE=1) for
generation — there is no multi-step ODE solver. CFG is applied via
`cfg_omega=1.0` (no guidance amplification) and `cfg_kappa=0.5`
(interpolation parameter). All 4 runs use identical settings.

**EMA decay:** Not logged explicitly in wandb, but all runs derive from
`meanflow_dit_imagenet_256_latent()` which specifies `ema_rate=0.9999`
(config.py:395). The beta=1 config (`vamf_l2_dit_...`) inherits from
the base config and does not override ema_rate (config.py:441-449).

**Protocol differences (bold rows):**
- `tangent_beta`: the treatment variable (0.0 / 0.25 / 0.5 / 1.0)
- `ema_tangent`: only beta=1 uses True (→ tangent_beta=1.0)
- `adaptive_weight_power`: beta=1 uses 0.0 (uniform weighting);
  others use 1.0 (Karras adaptive). This is an intentional design
  choice documented in config.py:438: "avoid double weighting" since
  `ema_tangent=True` removes the variance term that adaptive weighting
  reweights. **This difference affects beta=1 only, not the beta=0 vs
  beta=0.25 comparison.**

**Verdict:** The beta=0 / beta=0.25 / beta=0.5 comparison is
fully protocol-matched. The beta=1 run has an intentional config
difference (`adaptive_weight_power=0.0`) that does NOT affect the
beta=0/0.25 seed-robustness analysis.


## Task B — Seed Robustness

### B1. Confirmed: 54/54 with `hde9iaqj` as beta=0

`fid_ordering.py` with `hde9iaqj` as beta=0 (full output, 30k-295k window):

```
=== Per-step 4-point FID ordering (fid; 59 matched steps) ===
predicted (paper) ordering: beta 0 < 0.25 < 0.5 < 1

    step     b=0.0    b=0.25     b=0.5     b=1.0   ordering             strict?
    5000   349.312   348.442   349.192   347.655   1.0<0.25<0.5<0.0     NO
   10000   526.200   502.252   525.643   528.849   0.25<0.5<0.0<1.0     NO
   15000   348.241   353.005   366.584   358.361   0.0<0.25<1.0<0.5     NO
   20000   224.888   236.889   241.451   239.350   0.0<0.25<1.0<0.5     NO
   25000   165.389   174.539   181.384   176.346   0.0<0.25<1.0<0.5     NO
   30000   128.942   138.466   145.170   148.809   0.0<0.25<0.5<1.0     yes
   35000    99.831   109.635   116.219   123.901   0.0<0.25<0.5<1.0     yes
   ... (all yes from 30k to 295k) ...
   290000    11.348    11.730    12.505    23.553   0.0<0.25<0.5<1.0     yes
   295000    11.370    11.764    12.515    23.356   0.0<0.25<0.5<1.0     yes

strict ordering holds at 54/59 matched steps (92%).
```

**54/54 in [30k, 295k]. Confirmed.**

The 5 non-strict steps (5k-25k) are early-training transient — the
paper's claim explicitly excludes those.


### B2. Seed-variance analysis (`seed_variance.py`)

Full output, 54 matched steps from 30k to 295k. `b0_paper` =
`hde9iaqj`, `b0_alt` = `fidpdet7`, `b025` = `884l3avm`.

```
=== beta=0 seed robustness (54 matched steps) ===
convention: 'effect' = b025 - b0  (>0 => beta=0 better)

    step  b0_paper   b0_alt     b025  seedspread  eff_paper  eff_alt  both>0.25?
   30000   128.942  126.529  138.466       2.413      9.524   11.937  yes
   35000    99.831   98.833  109.635       0.999      9.803   10.802  yes
   40000    79.165   78.432   88.390       0.733      9.225    9.958  yes
   45000    64.354   64.294   72.691       0.060      8.337    8.397  yes
   50000    54.460   55.077   61.563       0.618      7.104    6.486  yes
   55000    47.350   48.398   53.203       1.048      5.853    4.805  yes
   60000    41.749   43.103   46.683       1.355      4.934    3.580  yes
   65000    37.280   39.064   41.355       1.784      4.075    2.291  yes
   70000    33.730   35.879   37.024       2.149      3.294    1.145  yes
   75000    30.671   32.925   33.245       2.254      2.575    0.321  yes
   80000    28.285   30.587   30.396       2.301      2.110   -0.191  no
   85000    26.197   28.408   28.092       2.212      1.895   -0.316  no
   90000    24.627   26.908   26.373       2.281      1.746   -0.535  no
   95000    23.305   25.271   24.733       1.966      1.428   -0.539  no
  100000    21.994   23.868   23.334       1.874      1.340   -0.534  no
  105000    20.912   22.736   22.274       1.824      1.362   -0.461  no
  110000    20.036   21.705   21.218       1.669      1.182   -0.487  no
  115000    19.176   20.776   20.322       1.600      1.146   -0.454  no
  120000    18.508   20.040   19.568       1.532      1.060   -0.473  no
  125000    17.765   19.289   18.821       1.524      1.055   -0.468  no
  130000    17.296   18.696   18.256       1.399      0.959   -0.440  no
  135000    16.865   18.250   17.700       1.385      0.835   -0.550  no
  140000    16.391   17.714   17.183       1.323      0.792   -0.531  no
  145000    15.977   17.201   16.668       1.224      0.691   -0.533  no
  150000    15.524   16.731   16.209       1.207      0.685   -0.522  no
  155000    15.054   16.221   15.803       1.167      0.748   -0.419  no
  160000    14.728   15.743   15.461       1.015      0.732   -0.282  no
  165000    14.460   15.355   15.090       0.895      0.629   -0.265  no
  170000    14.200   15.042   14.787       0.842      0.587   -0.255  no
  175000    13.888   14.707   14.500       0.819      0.611   -0.207  no
  180000    13.670   14.482   14.208       0.812      0.538   -0.274  no
  185000    13.453   14.220   13.960       0.766      0.507   -0.260  no
  190000    13.288   13.956   13.755       0.667      0.466   -0.201  no
  195000    13.097   13.687   13.589       0.590      0.492   -0.098  no
  200000    12.955   13.525   13.344       0.571      0.390   -0.181  no
  205000    12.854   13.310   13.213       0.455      0.359   -0.096  no
  210000    12.687   13.125   13.056       0.438      0.369   -0.069  no
  215000    12.619   13.047   12.858       0.428      0.239   -0.189  no
  220000    12.516   12.802   12.708       0.286      0.192   -0.094  no
  225000    12.389   12.671   12.545       0.282      0.156   -0.126  no
  230000    12.257   12.536   12.409       0.279      0.152   -0.127  no
  235000    12.124   12.466   12.327       0.342      0.203   -0.140  no
  240000    12.008   12.361   12.312       0.353      0.304   -0.048  no
  245000    11.916   12.315   12.226       0.399      0.310   -0.089  no
  250000    11.898   12.275   12.114       0.376      0.216   -0.161  no
  255000    11.883   12.061   11.940       0.178      0.057   -0.120  no
  260000    11.815   11.982   11.860       0.167      0.044   -0.122  no
  265000    11.706   11.919   11.834       0.213      0.128   -0.085  no
  270000    11.638   11.807   11.797       0.169      0.159   -0.010  no
  275000    11.634   11.774   11.743       0.139      0.109   -0.031  no
  280000    11.541   11.671   11.765       0.129      0.224    0.094  yes
  285000    11.457   11.582   11.763       0.125      0.305    0.180  yes
  290000    11.348   11.571   11.730       0.223      0.382    0.159  yes
  295000    11.370   11.533   11.764       0.162      0.394    0.232  yes
```

**Summary statistics:**

| Quantity | Value |
|----------|-------|
| Median beta=0 seed spread | **0.815 FID** |
| Median |beta=0 -> 0.25 effect| | **0.657 FID** |
| Seed spread / effect (median) | **1.24x** |
| Steps where |effect| < seed spread | **40/54** |
| **Seed-robust ordering (BOTH seeds beat 0.25)** | **14/54** |
| Paper-seed-only wins (fragile) | **40/54** |

**Interpretation:** The beta=0 vs beta=0.25 FID gap (~0.2-1.3 FID
mid-training, ~0.1-0.4 at convergence) is **within the run-to-run
variance** of two beta=0 seeds (~0.2-2.3 FID). The strict ordering
that the paper achieves 54/54 with `hde9iaqj` drops to **14/54** with
`fidpdet7` — and the seed-robust count (both seeds beat 0.25) is also
only **14/54**. All 14 seed-robust steps are in early training
(30k-75k) where the effect size is large, plus final convergence
(280k-295k) where beta=0 retakes the lead.

The mid-training window (80k-275k) has seed spread > effect at
**every step**. At these steps, the paper's ordering holds only because
the paper seed (`hde9iaqj`) is the faster-converging of the two
beta=0 seeds.

**This does NOT mean the theoretical ordering is wrong** — it means the
beta=0/0.25 FID difference is small enough (~0.5 FID) that 2 seeds
cannot resolve it statistically. The convergence-point ordering
(step 280k+) IS seed-robust. The "every matched-step" claim is not.


## Task C — Open Items

### C1. Behavior run preservation

| Field | Value |
|-------|-------|
| Source | `gs://pdt_training/juanwu/meanflow/meanflow_dit_b4_imagenet_256_20260501_140313_behavior/` |
| Destination | `gs://pdt_training/juanwu/vamf_preserved/behavior_baseline/` |
| Size | **43.3 GiB** (1.1k objects) |
| Method | `gsutil -m rsync -r` |
| Status | **COMPLETE** |

Same-bucket caveat: both source and copy are in `gs://pdt_training`.


### C2. Updated authoritative run map

| Role | exp_name | wandb (FID worker) | GCS path | Final FID | Preserved |
|------|----------|-------------------|----------|-----------|-----------|
| **Paper beta=0** | `meanflow_dit_b4_..._20260501_140313_behavior` | `hde9iaqj` | `.../meanflow_dit_b4_imagenet_256_20260501_140313_behavior/` | 11.37 | `vamf_preserved/behavior_baseline/` (43.3 GiB) |
| **Indep. beta=0 replication** | `dit_b4_latent_20260427_045409` | `fidpdet7` | `.../dit_b4_latent_20260427_045409/` | 11.53 | `vamf_preserved/dit_b4_latent_baseline/` (54.1 GiB) |
| Diverged beta=0 (#1) | `dit_imagenet_256_latent` | `77ncnhsm` | `.../dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` | 59.84 | `vamf_preserved/baseline_dit_imagenet_256_latent/` (37.7 GiB) |
| Diverged beta=0 (#3) | `dit_b4_cfg_20260424_171330` | `aewwmcwq` | `.../meanflow/dit_b4_cfg_20260424_171330/` | 107.60 | not preserved |
| **beta=0.25** | `vamf_beta025_dit_b4_imagenet_256_20260503_170426` | `884l3avm` | `.../meanflow/vamf_beta025_dit_b4_.../` | 11.76 | `vamf_preserved/vamf_beta025_dit_b4_imagenet_256/` (54.1 GiB) |
| **beta=0.5** | `vamf_beta05_dit_b4_imagenet_256_20260502_211910` | `milo2x6t` | `.../vamf_beta05_dit_b4_.../` | 12.51 | `vamf_preserved/vamf_beta05_dit_b4_imagenet_256/` (54.1 GiB) |
| **beta=1** | `vamf_l2_dit_b4_imagenet_256_20260501_140313` | `hrnr0uvz` | `.../vamf_l2_dit_b4_.../` | 23.36 | `vamf_preserved/vamf_l2_dit_b4_imagenet_256/` (54.3 GiB) |


### C3. `four_method_fid.json` status

**Not found.** Searched:
- Local repo: `find . -name "four_method_fid.json"` — not present
- `logs/` directory: exists but contains only empty `meanflow/` and
  `vamf/` subdirs
- GCS behavior run directory: no JSON files
- GCS `gs://pdt_training/juanwu/`: no `dit_probe` prefix

The JSON was local-only (`logs/vamf/dit_probe/four_method_fid.json`)
and was never committed (gitignored). It may still exist on the TPU VM.
The figure script's docstring ("produced by the four-method wandb pull
script") suggests it was generated by a one-off script that also does
not exist in the repo. **It can be reconstructed from the 4 wandb FID
histories** already exported to `docs/generative/vamf/wandb_export/`.


## Open Items for the Human

**(a) Reword `experiment.tex` toward a seed-aware claim.** The current
"holds at every 5k-aligned matched-step checkpoint (54 consecutive
datapoints from step 30k to step 295k)" is true for the paper's seed
but not seed-robust. Suggested reword:

> "The four-point FID ordering FID(beta=0) < FID(beta=0.25) <
> FID(beta=0.5) < FID(beta=1) holds at all 54 matched-step checkpoints
> from step 30k to step 295k for the reported baseline. The
> convergence-point ordering (step 280k+) replicates with an independent
> beta=0 seed; at intermediate steps the beta=0/beta=0.25 margin
> (~0.2-0.5 FID) is comparable to run-to-run variance (~0.8 FID),
> while the 3-point sub-ordering FID(0.25) < FID(0.5) < FID(1) holds
> at 52/54 steps with either seed."

This is honest without underselling: the theory's direction is correct,
the convergence ordering holds, and the 0.25/0.5/1.0 sub-ordering is
robust.

**(b) Consider adding error bars or a seed-variance note.** The
`fidpdet7` replication gives a natural "error bar" for the beta=0
data point. At convergence (295k): beta=0 = 11.37 +/- 0.16 (range of
2 seeds), beta=0.25 = 11.76 (1 seed). The effect (0.39 FID) is 2.4x
the seed range at 295k — defensible at convergence.

**(c) Reconstruct `four_method_fid.json`.** Can be built from
the wandb exports using `hde9iaqj` for the `baseline` key. Needed if
the figure script is rerun.

**(d) Do NOT start the v_ref / ||b||^2 (A1) work.** Separate,
compute-bearing, awaiting explicit human go.
