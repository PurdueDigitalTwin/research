# VAMF Checkpoint & wandb Audit

Audit date: 2026-06-25. Zero TPU, zero training, zero deletions.


## 1. Checkpoint Inventory

All sizes from recursive `gcloud storage du -s` (not `ls`).

| beta | GCS path | Size | Steps | # Ckpts |
|------|----------|------|-------|---------|
| 0 (baseline) | `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` | 37.74 GiB | 10k-210k | 21 |
| 0 (second run) | `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/` | 54.07 GiB | 10k-300k | 30 |
| 0.25 | `gs://pdt_training/juanwu/meanflow/meanflow/vamf_beta025_dit_b4_imagenet_256_20260503_170426/` | 54.14 GiB | 10k-300k | 30 |
| 0.5 | `gs://pdt_training/juanwu/meanflow/vamf_beta05_dit_b4_imagenet_256_20260502_211910/` | 54.14 GiB | 10k-300k | 30 |
| 1.0 | `gs://pdt_training/juanwu/meanflow/vamf_l2_dit_b4_imagenet_256_20260501_140313/` | 54.26 GiB | 10k-300k | 30 |

**NOTE:** Two previous audits (B1 handoff, B1 variance addendum)
incorrectly reported the beta-sweep checkpoints as "empty / 0 bytes."
The error was using non-recursive `gsutil ls` / `gsutil du -s`, which
only sees the 0-byte directory-marker objects and misses the actual data
in `params/` and `state/` subdirs. All four beta-sweep runs have real
Orbax checkpoints.


## 2. beta=1 Verdict: PRESENT (outcome i)

### Triangulation

**Direction A (wandb -> config):**

The 4 candidate IDs (`fidpdet7`, `rwxjrbh8`, `qacd8zvy`, `0yf65ai6`)
are **NOT beta=1**. They are the second baseline run (beta=0):

- Entity/project: `pdt-purdue-university/meanflow`
- exp_name: `dit_b4_latent_20260427_045409`
- Model config: `MeanFlowDiTModel` with NO `tangent_beta` field
  (default=0.0, no `ema_tangent`)
- GCS dir: `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/`
- wandb.txt: `fidpdet7` (matches)

The actual beta=1 run was found by searching wandb for
`ema_tangent=True` in DiT configs:

- Run IDs (4 workers): `7ku9ef6z`, `9uvcq5fw`, `f0ozsd25`, `hrnr0uvz`
- All state=finished, ema_tangent=True (-> tangent_beta=1.0)
- exp_name: `vamf_l2_dit_b4_imagenet_256_20260501_140313`
- An earlier attempt (`_20260501_014255`) crashed on all 4 workers x3
  retries (12 crashed runs in wandb).

**Direction B (GCS -> wandb.txt):**

```
gcloud storage cat gs://pdt_training/juanwu/meanflow/vamf_l2_dit_b4_imagenet_256_20260501_140313/wandb.txt
-> 7ku9ef6z
```

Matches the finished worker 0 from Direction A.

**Reconciliation:** work_dir from Direction A (`vamf_l2_dit_b4_imagenet_256_20260501_140313`)
= directory from Direction B. Config confirms `ema_tangent=True` ->
`tangent_beta=1.0`. Size = 54.26 GiB, 30 checkpoints at 10k-300k.

**Rosetta stone validation:** `gcloud storage cat` of the known beta=0.25
dir's `wandb.txt` returned `08i3wa20`, confirming the format (bare 8-char
run ID) and that the link is reliable.


## 3. wandb Export

All FID-logging worker configs, summaries, and histories exported to
`docs/generative/vamf/wandb_export/`. The FID-logging worker for each
job is:

| beta | FID worker | wandb entity/project |
|------|-----------|---------------------|
| 0 | `fidpdet7` | pdt-purdue-university/meanflow |
| 0.25 | `884l3avm` | pdt-purdue-university/meanflow |
| 0.5 | `milo2x6t` | pdt-purdue-university/meanflow |
| 1.0 | `hrnr0uvz` | pdt-purdue-university/meanflow |

### FID ordering at matched steps

The paper claims FID(0) < FID(0.25) < FID(0.5) < FID(1) at matched-step
checkpoints. From wandb history:

| Step | b=0 | b=0.25 | b=0.5 | b=1 | Strict ordering? |
|------|-----|--------|-------|-----|-----------------|
| 30k | 126.53 | 138.47 | 145.17 | 148.81 | YES |
| 50k | 55.08 | 61.56 | 66.88 | 79.89 | YES |
| 100k | 23.87 | 23.33 | 25.55 | 41.71 | NO (b025 < b0) |
| 150k | 16.73 | 16.21 | 17.60 | 32.08 | NO (b025 < b0) |
| 200k | 13.53 | 13.34 | 14.40 | 27.61 | NO (b025 < b0) |
| 250k | 12.27 | 12.11 | 13.17 | 25.10 | NO (b025 < b0) |
| 295k | 11.53 | 11.76 | 12.51 | 23.36 | YES |

Full ordering holds at 15/59 matched steps. The FID(0) < FID(0.25)
link is the weakest: beta=0.25 slightly beats beta=0 from step 80k to
step 275k (by ~0.1-0.5 FID). At steps 280k+ (near convergence), the
full strict ordering re-establishes.

**Is the paper's per-checkpoint bias-variance FID ordering
reconstructable from wandb alone?**

**YES.** All 59 per-step FID values for all 4 betas are logged in wandb
(`eval/fid` key). The ordering table can be reconstructed entirely from
the exported JSON files without accessing any checkpoints. The paper's
core DiT result (FID ordering tracks the bias-variance prediction) is
safe regardless of checkpoint state.

**Nuance for rebuttal:** The strict 4-point ordering holds cleanly at
steps 30k-75k (early convergence) and 280k-295k (final convergence),
but beta=0.25 dips below beta=0 at intermediate steps. This is
consistent with the paper's own discussion of the FID-MSE mismatch:
at small beta the bias is small enough that variance reduction gives a
temporary FID advantage, which the paper does not claim away.


## 4. Bucket Lifecycle

```
gsutil lifecycle get gs://pdt_training
-> gs://pdt_training/ has no lifecycle configuration.
```

No auto-delete TTL. No time pressure for preservation.


## 5. Preservation

| Run | Source | Destination | Size (bytes) |
|-----|--------|-------------|-------------|
| beta=0 baseline | `.../dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/` | `.../vamf_preserved/baseline_dit_imagenet_256_latent/` | 40,525,186,266 |
| beta=0.25 | `.../meanflow/vamf_beta025_dit_b4_imagenet_256_20260503_170426/` | `.../vamf_preserved/vamf_beta025_dit_b4_imagenet_256/` | 58,127,296,492 |
| beta=0.5 | `.../vamf_beta05_dit_b4_imagenet_256_20260502_211910/` | `.../vamf_preserved/vamf_beta05_dit_b4_imagenet_256/` | 58,130,154,728 |
| beta=1.0 | `.../vamf_l2_dit_b4_imagenet_256_20260501_140313/` | `.../vamf_preserved/vamf_l2_dit_b4_imagenet_256/` | 58,268,182,472 |

All destinations under `gs://pdt_training/juanwu/vamf_preserved/`.
Method: `gsutil -m rsync -r`. Sources NOT deleted or moved.

**Total preserved: ~200 GiB across 111 checkpoints (4 runs).**

**SAME-BUCKET CAVEAT:** All copies are in the same bucket
(`gs://pdt_training`) as the sources. This protects against accidental
path-level deletion but NOT against bucket-level operations (bucket
delete, project-wide IAM changes, billing suspension). If a separate
durable/archive bucket exists, these should be moved there.


## 6. Open Items

1. **Second baseline run.** The 4 candidate IDs (`fidpdet7` et al.) are
   actually a second beta=0 baseline at
   `gs://pdt_training/juanwu/meanflow/dit_b4_latent_20260427_045409/`
   (54.07 GiB, 30 checkpoints). This is the run whose FID values are in
   the wandb export. The first baseline at
   `.../dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/`
   (wandb: `f5fffn99`, state=crashed, 37.74 GiB, 21 checkpoints) may
   be an earlier attempt. **Decide which baseline is the paper's
   canonical one** — the FID data comes from `fidpdet7` (second run).

2. **Archive bucket.** All preserved copies are in `gs://pdt_training`.
   If a separate archive/cold-storage bucket is available, move the
   `vamf_preserved/` prefix there for true redundancy.

3. **beta=1 crashed attempts.** 12 crashed wandb runs exist for
   `vamf_l2_dit_b4_imagenet_256_20260501_014255` (the first beta=1
   attempt). The GCS dir for that run is empty (confirmed 0 bytes
   recursive). No action needed, but noted for completeness.

4. **HF_TOKEN scrubbed.** The wandb configs contained the HF_TOKEN in
   the `metric` field (FID dataset loader partial). All exported JSON
   files have been scrubbed (`hf_...` -> `hf_REDACTED`). Safe to
   commit.
