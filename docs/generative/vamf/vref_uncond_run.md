# Unconditional v_ref Run Report (Step 2' of 3)

Date: 2026-06-26. VM: tpu-v4-32-us-central2-b. Branch: feat/vamf-rebuttal-beta-anneal.

## Summary

Retrained the v_ref flow-matching model with **unconditional** (null-class)
labels to match the paper's unconditional MeanFlow theory and the probe's
null-class evaluation. The conditional run (`qejr4spi`) was stopped and its
checkpoints preserved for potential supplementary analysis.

## Decision: World 2 (unconditional)

The paper's theory is unconditional MeanFlow, so v_ref must estimate the
unconditional marginal velocity E[v_cond | x_t, t] with null-class
conditioning. This makes the validity gate clean:

- FM loss plateau should land near sigma^2 * d ~ 8.2e3
- ||b_ref||^2 = loss - 8.2e3 is a valid metric
- Matches the probe's null-class evaluation exactly

## Task 1 — Conditional run stopped

| Field | Value |
|-------|-------|
| Run ID | `qejr4spi` (coordinator) |
| State | Stopped (crashed from kill -9) |
| Last step | 4950 (wandb) |
| Last loss | 3286.4 |
| Latest checkpoint | step 5000 |
| Checkpoint path | `gs://pdt_training/juanwu/meanflow/fm_dit_b4_vref/checkpoints/5000/` |

Checkpoints **NOT deleted** — may be resumed later for class-conditional
supplementary analysis (CFG-regime reviewer question).

## Task 2 — Probe's null-class representation (verbatim)

### How the DiT handles class labels

1. `LabelEmbedder` (`dit.py:921`) uses `nn.Embed` with
   `num_embeddings = num_classes + int(use_cfg_embedding)`.
2. Line 836 in `MeanFlowDiTModule.__call__` forces
   `dropout_rate=max(self.dropout_rate, 1.0)`, so `use_cfg_embedding`
   is always `True` and the table always has **1001 entries**
   (indices 0-999 = classes, index 1000 = null/unconditional).
3. Line 840: `deterministic=True` — the LabelEmbedder's internal
   dropout never fires. Class dropout is handled **externally** in
   the training step.

### Probe's null-class token

```python
# probe_dit_checkpoint.py, line 148
null_label = jnp.int32(model.num_classes)  # = 1000

# line 153
labels = jnp.full((n,), null_label, dtype=jnp.int32)
```

The probe feeds **index 1000** (a learned null embedding, NOT zero)
for all samples. This is the standard DiT CFG null token.

### v_ref unconditional config

Feeds the same null-class token for all samples via:
```python
config.model.fm_only_cfg = True       # activates dropout path
config.model.class_dropout_prob = 1.0  # 100% dropout -> all null
```

In `_fm_only_training_step` (line 1576-1581):
```python
if self.fm_only_cfg:
    drop_mask = jnp.less(
        jax.random.uniform(cfg_rng, shape=batch_dims),
        self.class_dropout_prob,  # = 1.0
    )
    y_inp = jnp.where(drop_mask, self.num_classes, labels)
    # drop_mask is always True -> y_inp = 1000 for all samples
```

Verified: `jnp.all(y_inp == 1000)` is `True` for all batch sizes.

### Conditioning consistency (World 2)

| Component | Label | Representation |
|-----------|-------|----------------|
| Probe u_MF(x_t, t, t) | null (unconditional) | `jnp.int32(1000)` |
| v_ref training | null (unconditional) | `jnp.int32(1000)` via 100% dropout |
| Step 3 evaluation | null (unconditional) | Both must use `jnp.int32(1000)` |

**All three match.** The Step 3 ||b||^2 estimator must evaluate both the
MeanFlow proxy and v_ref at null-class conditioning.

## Task 3 — Unconditional config

New config function `fm_dit_imagenet_256_latent_uncond` added to
`config.py` (commit `32251c2`). Pure config change — no modification
to `meanflow.py`. Inherits everything from `fm_dit_imagenet_256_latent`
except:

| Field | Conditional (old) | Unconditional (new) |
|-------|-------------------|---------------------|
| `exp_name` | `fm_dit_b4_imagenet_256_latent` | `fm_dit_b4_vref_uncond` |
| `fm_only_cfg` | `False` | `True` |
| `class_dropout_prob` | 0.1 (inherited) | `1.0` |

Everything else identical: DiT-B/4, 131M params, SD-VAE latents,
logit-normal t-sampler, AdamW lr=1e-4, EMA 0.9999, checkpoint every
5k, 200k steps, keep 5.

## Task 4 — Sanity test + full run

### Sanity (first ~300 steps)

| Check | Result |
|-------|--------|
| Loss at step 0 | 6895 (cf. conditional: 6891) |
| Loss at step 50 | 5653 |
| Loss at step 100 | 4435 |
| Loss at step 150 | 4199 |
| Loss at step 200 | 4076 |
| Loss at step 250 | 3929 |
| Loss at step 300 | 3943 |
| Throughput | 4.60-4.68 steps/sec |
| Root disk delta (W25) | +2,652 KB (flat) |
| Root disk delta (W13) | +1,460 KB (flat) |
| Root disk delta (W227) | +9,652 KB (flat) |
| Root disk delta (W84) | +9,668 KB (flat) |

Loss is monotonically decreasing (except step 300 noise). Initial loss
6895 is nearly identical to the conditional run's 6891 — expected since
the null-class embedding gives slightly less information.

### Full run status

The sanity test transitioned directly into the full run (the fiddle
`set:num_train_steps=100` override doesn't reach the trainer's nested
field, so the config's 200k steps applied). This is the production run.

| Field | Value |
|-------|-------|
| Config | `fm_dit_imagenet_256_latent_uncond` |
| exp_name | `fm_dit_b4_vref_uncond` |
| wandb coordinator | `pdt-purdue-university/meanflow/runs/h7zl3wfw` |
| wandb siblings | `1pt6lmij` (w1), `qxnh4wf1` (w2), `pmwbpcsy` (w3) |
| GCS log_dir | `gs://pdt_training/juanwu/meanflow/fm_dit_b4_vref_uncond` |
| Launched | 2026-06-26 ~10:28 UTC |
| Projected wall-clock | ~11.9h at 4.65 steps/sec |
| Checkpoint cadence | every 5k steps, keep 5 |
| Commit | `32251c2` |

### Launch command (for auto-resume)

```bash
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker all \
    --command 'eval $(grep "export HF_TOKEN" ~/.bashrc) && \
        cd pdt-research && ulimit -c 0 && \
        export HF_HOME=/dev/shm/data/huggingface && \
        export WANDB_DIR=/dev/shm/wandb && \
        export TMPDIR=/dev/shm/tmp && \
        export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache && \
        mkdir -p /dev/shm/bazel-sandbox /dev/shm/wandb \
                 /dev/shm/tmp /dev/shm/jax_cache && \
        bazelisk run --config=tpu //src/projects/generative:main -- \
            --distributed=true \
            --experiment config:src.projects.generative.config.fm_dit_imagenet_256_latent_uncond \
            --experiment '"'"'set:exp_name="fm_dit_b4_vref_uncond"'"'"' \
            --work_dir=gs://pdt_training/juanwu'
```

Same command works for fresh start and auto-resume from latest checkpoint.

### Monitoring

- **Validity gate (two-sided):** FM loss plateau should land near 8.2e3
  (unconditional sigma^2 * d).
  - plateau >> 8.2e3: undertrained, extend
  - plateau << 8.2e3: investigate (reported 8.2e3 may be off)
- **||b_ref||^2 = loss - 8.2e3** is now a valid metric (unconditional).
- **Stop when loss plateaus** (likely < 200k steps).

### Preservation target

`gs://pdt_training/juanwu/vamf_preserved/fm_dit_b4_vref_uncond/`

## Step 3 eval interface

For the ||b||^2 estimator (Step 3), both models must use **null-class
conditioning** (index 1000):

```python
# MeanFlow proxy: u_MF(x_t, t, t) with null labels
null_label = jnp.int32(model.num_classes)  # 1000
labels = jnp.full((batch_size,), null_label, dtype=jnp.int32)
ts = model._make_timestamps(t_in=t, r_in=t)
u_mf = model._network.apply(
    variables={"params": params},
    inputs=z, timestamps=ts, labels=labels,
    edm_cond=None, deterministic=True,
)

# v_ref: same interface, same null labels
u_vref = vref_network.apply(
    variables={"params": vref_params},
    inputs=z, timestamps=ts, labels=labels,
    edm_cond=None, deterministic=True,
)

# Per-sample ||b||^2 estimate
b_sq = jnp.sum(jnp.square(u_mf - u_vref), axis=(-1, -2, -3))
```

## Guardrails

- Root-disk safety: verified flat across ~300 steps. All scratch routed
  to /dev/shm.
- Conditional run `qejr4spi` preserved (NOT deleted). Checkpoint at
  step 5000 in `gs://pdt_training/juanwu/meanflow/fm_dit_b4_vref/`.
- Distinct exp_name `fm_dit_b4_vref_uncond` does NOT clobber conditional
  checkpoints.
- Step 3 (||b||^2 estimator + beta* computation): awaits explicit go.
- Paper not edited.

## Open item

Optional later: resume conditional run `qejr4spi` from step 5000 for
class-conditional v_ref analysis. This provides a supplementary
CFG-regime result for anticipated reviewer question on conditioning.
