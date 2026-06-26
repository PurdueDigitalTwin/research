# v_ref Full Run Report (Step 2 of 3)

Date: 2026-06-26. VM: tpu-v4-32-us-central2-b. Branch: feat/vamf-rebuttal-beta-anneal.

## Summary

Trained an independent flow-matching velocity reference model (DiT-B/4,
131M params) on ImageNet-256 latents. This is Step 2 of the v_ref
pipeline: scope (Step 1, done) -> train (this step) -> estimate ||b||^2
and beta* (Step 3, awaits go).

## Section 2a — Pipeline + Preemption Gate

### Pipeline test (100 steps)

| Check            | Result                                    |
| ---------------- | ----------------------------------------- |
| Data pipeline    | ImageNetLatentDataModule loaded end-to-end |
| Model init       | DiT-B/4, 131,156,032 params (524.6 MB)    |
| Checkpoint save  | Steps 50, 100 to GCS                      |
| Steady-state     | **4.63-4.67 steps/sec**                   |
| JIT compilation  | ~97s first step                           |
| Root disk delta  | +2.8 KB (49,672,788 -> 49,675,560 KB)     |

### Preemption test

Killed process after step 100 checkpoint. Relaunched same command
targeting 150 steps.

| Check            | Result                                              |
| ---------------- | --------------------------------------------------- |
| Auto-detect ckpt | "Found 2 checkpoint steps"                          |
| Auto-resume      | "Resumed train state at step 100 (target 150)."     |
| wandb resume     | Same run ID (zqm81dzv), "Resuming run"              |
| Step continuity  | Progress bar started at 100/150                     |
| Loss continuity  | step 90: 4644 -> step 110: 4427 (no discontinuity)  |
| Root disk final  | +4.8 KB total (49,672,788 -> 49,677,556 KB)         |

**Gate verdict: ALL GREEN.** Pipeline, checkpointing, auto-resume, and
root-disk safety all verified.

### Smoke test loss curve

```
step    loss
   0   6891
  10   6604
  20   6299
  30   6117
  40   5934
  50   5784
  60   5540
  70   5149
  80   4833
  90   4644
 110   4427  <- resumed here
 120   4406
 130   4336
 140   4225
```

Loss monotonically decreasing. No discontinuity across the resume at
step 100. wandb run: `pdt-purdue-university/meanflow/runs/zqm81dzv`.

## Section 2b — Full v_ref Run

### Config

```
experiment:     fm_dit_imagenet_256_latent
exp_name:       fm_dit_b4_vref
fm_only:        True
fm_only_cfg:    False (always true class label, no null-dropout)
num_train_steps: 200000
checkpoint:     every 5000 steps, keep 5
ema_rate:       0.9999
batch_size:     256 (64 per worker x 4 workers)
optimizer:      AdamW, lr=1e-4, wd=0
t-sampler:      logit-normal (mean=-0.4, std=1.0)
work_dir:       gs://pdt_training/juanwu
```

### Launch command

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
            --experiment config:src.projects.generative.config.fm_dit_imagenet_256_latent \
            --experiment '"'"'set:exp_name="fm_dit_b4_vref"'"'"' \
            --work_dir=gs://pdt_training/juanwu'
```

Same command works for fresh start and resume (auto-detect).

### Projected wall-clock

At 4.65 steps/sec (measured):

| Horizon | Steps   | Wall-clock |
| ------- | ------- | ---------- |
| 50k     | 50,000  | 3.0 hours  |
| 100k    | 100,000 | 6.0 hours  |
| 200k    | 200,000 | 11.9 hours |

### Monitoring

Loss should decrease toward sigma^2 * d (the irreducible noise floor).
At convergence: FM_loss ~ sigma^2 * d, and ||b_ref||^2 ~ 0.

Validity gate: if the loss plateaus well above the expected noise floor,
v_ref is undertrained and needs more steps or a learning rate change.

wandb project: pdt-purdue-university/meanflow.

### Status

Full run launched 2026-06-26 09:49 UTC. Run name: fm_dit_b4_vref.
wandb coordinator (worker 0): pdt-purdue-university/meanflow/runs/qejr4spi.
GCS log_dir: gs://pdt_training/juanwu/meanflow/fm_dit_b4_vref.

## Section 4 — Conditioning Consistency

### Probe (probe_dit_checkpoint.py)

The existing probe uses **unconditional** (null-class) labels:
```python
null_label = jnp.int32(model.num_classes)  # line 138
u_fn(z, t, t)  # evaluates with null_label
```

### v_ref training (fm_only_cfg=False)

v_ref trains with **true class labels** (no null-dropout):
```python
if self.fm_only_cfg:
    # CFG dropout path (NOT used)
    ...
else:
    y_inp = labels  # always true class
```

### Mismatch and Step 3 implications

There is a conditioning mismatch between probe and v_ref:
- Probe: unconditional (null_label)
- v_ref: class-conditional (true labels)

For Step 3 (||b||^2 estimation), both the MeanFlow proxy u_MF(x_t, t, t)
and v_ref must be evaluated with **matched conditioning**. Since v_ref was
trained class-conditional, the Step 3 evaluation must also use true class
labels for the MeanFlow boundary query.

### MeanFlow CFG formula (for reference)

```
v_g = omega*v + (1 - omega - kappa)*v_uncond + kappa*v_cond
```
where omega=1.0, kappa=0.5. At the boundary (r=t), the MeanFlow model
outputs u(x_t, t, t) which serves as the proxy for v(x_t, t).

## Guardrails

- Root-disk safety: verified flat across 150 steps and 2 process launches.
  All scratch routed to /dev/shm.
- Auto-resume: verified working. Same command resumes from latest checkpoint.
- Step 3 (||b||^2 estimator + beta* computation): awaits explicit go.
- Paper not edited.
- Preservation target: gs://pdt_training/juanwu/vamf_preserved/fm_dit_b4_vref/
