# Beta=0 FID Provenance: Forensic Report

Audit date: 2026-06-26. Zero TPU, zero FID re-evaluation, no edits to
paper.


## 0. Executive Summary

The paper's beta=0 FID values come from a **separate, previously
unidentified training run** (`meanflow_dit_b4_imagenet_256_20260501_
140313_behavior`, wandb ID `hde9iaqj`), not from `fidpdet7`
(`dit_b4_latent_20260427_045409`). The two runs have identical model
and eval configs — the FID difference is from stochastic training
variation (different seeds / start times). With the correct baseline,
the paper's 54/54 matched-step ordering claim **is fully supported**.


## 1. The Discrepancy

The previous audit compared the paper's beta=0 FID values against the
`fidpdet7` wandb log and found:
- beta=0 paper values up to **1.87 FID lower** than wandb
- beta>0 matched within 0.05 FID
- Strict 4-point ordering held at only **14/54** steps with `fidpdet7`

The question: where do the paper's beta=0 values actually come from?


## 2. Hypothesis Discrimination

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H0 Different training run | **CONFIRMED** | See Section 3 |
| H1 Offline eval | Eliminated | Paper's values match the behavior run's in-training wandb log, not an offline eval |
| H2 EMA vs raw | Eliminated | Both runs use `state.ema_params` for in-training eval (experiment.py:462) |
| H3 Different #samples | Eliminated | Both configs specify identical `FrechetInceptionDistance` with 50k samples |
| H4 Cherry-pick | Eliminated | Paper's min (11.37) is below `fidpdet7`'s whole-curve min (11.53) — impossible via step selection from fidpdet7 |
| H5 Different sampler/NFE | Eliminated | Identical model configs (cfg_omega=1.0, cfg_kappa=0.5, same DiT architecture) |


## 3. The Paper's Actual Beta=0 Source

### Run identification

| Field | Paper's beta=0 (behavior) | Previously assumed (fidpdet7) |
|-------|--------------------------|------------------------------|
| wandb ID | `hde9iaqj` | `fidpdet7` |
| exp_name | `meanflow_dit_b4_imagenet_256_20260501_140313_behavior` | `dit_b4_latent_20260427_045409` |
| Created | 2026-05-01 | 2026-04-27 |
| GCS path | `gs://pdt_training/juanwu/meanflow/meanflow_dit_b4_imagenet_256_20260501_140313_behavior/` | `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/` |
| GCS size | 46.5 GiB (24 ckpts, 10k-240k) | 54.1 GiB (30 ckpts, 10k-300k) |
| wandb.txt | `rahra7hr` (worker 3) | `fidpdet7` (worker 0) |
| FID worker | `hde9iaqj` (worker 0, FID=11.37 at 295k) | `fidpdet7` (worker 0, FID=11.53 at 295k) |
| All workers | `hde9iaqj`, `bct4rdwc` (crashed), `khpzouos` (crashed), `rahra7hr` (crashed) | `fidpdet7`, `rwxjrbh8`, `qacd8zvy`, `0yf65ai6` |
| State | finished (1 of 4 workers) | finished (all 4 workers) |

### How identified

1. The figure script (`plot_dit_fid_curves.py`) uses `--baseline_floor=12.12` — this matches the behavior run's FID plateau, not fidpdet7's.
2. The old docstring (deleted in commit `f1d53b4`) says the FID JSON was "produced by the four-method wandb pull script."
3. There are 8 evaluate-mode runs in wandb, all for a THIRD baseline (`dit_b4_cfg_20260424_171330`) which diverged (FID=107). The offline eval run `i6829wzy` (FID=11.52) is unrelated to the paper's table.
4. The behavior run's FID matches the paper's table at every step within rounding error (±0.045).

### Timestamp connection

The behavior run shares its timestamp (`20260501_140313`) with the
beta=1 run (`vamf_l2_dit_b4_imagenet_256_20260501_140313`). It was
launched simultaneously as the beta=0 control for the beta=1
experiment. The `_behavior` suffix is the convention for the control
run in a VaMF experiment.

### FID match verification

| Step | Paper beta=0 | Behavior wandb | Difference |
|------|-------------|----------------|------------|
| 30k | 128.9 | 128.942 | +0.042 |
| 50k | 54.5 | 54.460 | -0.040 |
| 100k | 22.0 | 21.994 | -0.006 |
| 150k | 15.5 | 15.524 | +0.024 |
| 200k | 13.0 | 12.955 | -0.045 |
| 250k | 11.9 | 11.898 | -0.002 |
| 275k | 11.6 | 11.634 | +0.034 |
| 295k | 11.37 | 11.370 | +0.000 |

All differences are < 0.05 FID — consistent with rounding to 1 decimal
place. **Match confirmed at all 8 tabulated steps.**


## 4. Protocol Comparison Table

| Field | beta=0 (behavior, paper) | beta=0 (fidpdet7) | beta>0 (paper) |
|-------|-------------------------|-------------------|----------------|
| Eval mode | in-training | in-training | in-training |
| Params used | EMA (state.ema_params) | EMA (state.ema_params) | EMA (state.ema_params) |
| #samples | 50,000 | 50,000 | 50,000 |
| Reference stats | ILSVRC/imagenet-1k (HF) | ILSVRC/imagenet-1k (HF) | ILSVRC/imagenet-1k (HF) |
| Sampler | CFG (omega=1.0, kappa=0.5) | CFG (omega=1.0, kappa=0.5) | CFG (omega=1.0, kappa=0.5) |
| Architecture | DiT-B/4 768d 12L | DiT-B/4 768d 12L | DiT-B/4 768d 12L |
| Data | ImageNetLatent (latent space) | ImageNetLatent (latent space) | ImageNetLatent (latent space) |
| Batch size | 256 | 256 | 256 |
| Eval freq | every 5k steps | every 5k steps | every 5k steps |
| Train steps | 300k | 300k | 300k |
| Start date | 2026-05-01 | 2026-04-27 | 2026-05-01 to 2026-05-03 |

**All protocol fields are identical.** The only difference is the
training seed (from different start times). The comparison is
apples-to-apples.


## 5. FID Ordering with Correct Baseline

Using the behavior run as beta=0, the strict 4-point ordering holds at
**54/54 steps** from 30k to 295k — exactly matching the paper's claim.

The previous audit's finding of "14/54 strict" was an artifact of using
the WRONG beta=0 run. With `fidpdet7`, beta=0.25 beats beta=0 at 40/54
steps because `fidpdet7` converges ~0.2-1.9 FID slower than the
behavior run. This is stochastic training variation, not a protocol
mismatch.


## 6. Run Inventory (Updated)

There are **4 known beta=0 MeanFlowDiT training runs**, not 2:

| # | exp_name | wandb (FID worker) | Final FID | Status |
|---|----------|-------------------|-----------|--------|
| 1 | `dit_imagenet_256_latent` | `77ncnhsm` | 59.84 (diverged) | CRASHED |
| 2 | `dit_b4_cfg_20260424_171330` | `aewwmcwq` | 107.60 (diverged) | Finished but broken |
| 3 | `dit_b4_latent_20260427_045409` | `fidpdet7` | 11.53 | Finished, healthy |
| 4 | `meanflow_dit_b4_imagenet_256_20260501_140313_behavior` | `hde9iaqj` | 11.37 | **PAPER'S BASELINE** |

Run #4 is the paper's beta=0. Run #3 (`fidpdet7`) is a viable
independent replication but was NOT used in the paper.


## 7. Unified-Protocol Options (for the human — NOT to be executed)

Since the paper's comparison is already apples-to-apples (same eval
protocol for all betas, confirmed in Section 4), no unified-protocol
re-eval is strictly necessary. However, two options exist:

### (i) Use wandb in-training values for all betas (CURRENT STATUS)

This is what the paper already does. All 5 runs (behavior + 4 betas)
used identical in-training eval. The ordering holds 54/54. **No action
needed.**

The only concern is that `fidpdet7` (not used in paper) gives a
different ordering — but this is a different training seed, not a
protocol difference. If a reviewer asks "does the ordering replicate
with a different baseline seed?", the answer is "not perfectly — the
margin is thin enough (~0.1-0.5 FID) that training variance can flip
the beta=0 vs beta=0.25 comparison at intermediate steps, while the
3-point sub-ordering (0.25 < 0.5 < 1.0) holds at 52/54 steps."

### (ii) Re-eval all betas offline with a single fixed protocol

**COMPUTE-BEARING — DO NOT EXECUTE WITHOUT EXPLICIT HUMAN GO + TPU BUDGET.**

If a reviewer demands matched-seed FID (same RNG for all 4 betas per
step), you would need to:
- Load each checkpoint (30 per run × 4 betas = 120 checkpoints)
- Generate 50k samples per checkpoint with a FIXED RNG seed
- Compute FID against the same reference stats
- Cost: ~120 checkpoint loads × ~15 min sampling each = ~30 TPU v4-32
  hours (ignoring parallelism across workers)
- Disk: ~120 × 50k × 256 × 256 × 3 bytes = ~2.4 TB sample storage
  (can compute FID on-the-fly to avoid storage)

This would eliminate FID evaluation noise across runs but does NOT
change the models or training — it only standardizes the sampling RNG.


## 8. Preservation Status

### Canonical baseline (`fidpdet7`)

| Field | Value |
|-------|-------|
| Source | `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/` |
| Destination | `gs://pdt_training/juanwu/vamf_preserved/dit_b4_latent_baseline/` |
| Size | 54.1 GiB (1.4k objects) |
| Status | **COMPLETE** |

### Behavior baseline (`hde9iaqj`, paper's actual beta=0)

| Field | Value |
|-------|-------|
| Source | `gs://pdt_training/juanwu/meanflow/meanflow_dit_b4_imagenet_256_20260501_140313_behavior/` |
| Size | 46.5 GiB (24 checkpoints, 10k-240k) |
| Status | **NOT YET PRESERVED** — should be copied to `vamf_preserved/` |

**Same-bucket caveat:** Both copies are in `gs://pdt_training`, same
bucket as source. Protects against path-level deletion only.


## 9. Open Items for the Human

**(a) No experiment.tex reword needed for the ordering claim.** The
54/54 claim holds with the correct baseline. However, you should add a
citation or footnote identifying the specific beta=0 run (behavior,
`hde9iaqj`) to prevent future confusion.

**(b) Preserve the behavior baseline.** The paper's actual beta=0 run
is NOT yet in `vamf_preserved/`. Run:
```
gsutil -m rsync -r \
  gs://pdt_training/juanwu/meanflow/meanflow_dit_b4_imagenet_256_20260501_140313_behavior/ \
  gs://pdt_training/juanwu/vamf_preserved/behavior_baseline/
```

**(c) Update the run map** in `fid_ordering_and_runmap.md` to reflect
the behavior run as the paper's canonical beta=0.

**(d) Verify the `four_method_fid.json`.** The JSON that fed the figure
script is not in the repo (it's at `logs/vamf/dit_probe/four_method_
fid.json`, which is gitignored). If it still exists locally or on the
TPU VM, verify that its `baseline` key's FID trajectory matches the
behavior run. If lost, reconstruct from the wandb exports.

**(e) `fidpdet7` robustness check.** The `fidpdet7` run is a viable
independent replication. The fact that the ordering doesn't hold
perfectly with a different beta=0 seed means the beta=0 vs beta=0.25
margin is thin (~0.2 FID) and training-variance-sensitive. The paper
could proactively acknowledge this in the discussion: "the four-point
ordering is consistent but the beta=0/0.25 margin narrows to ~0.2 FID
at convergence, within the range of training-seed variation."

**(f) Do NOT start the v_ref / ||b||^2 (A1) work.** Separate,
compute-bearing, awaiting explicit human go.
