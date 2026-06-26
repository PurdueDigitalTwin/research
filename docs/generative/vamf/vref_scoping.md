# v_ref Scoping: Estimand, Implementation, Smoke Test, Cost

Step 1 of 3. No full training run. Date: 2026-06-26.

## Task 1 — Paper Definitions (Verbatim)

All definitions from `method.tex` and `supplementary.tex`.

### b (proxy bias)

From notation table and Eq. (5) / Section 3.2:

> b := v_hat - v(x_t, t)

where:

- `v_hat` is any **deterministic proxy** for the marginal velocity
  (e.g., `u_{theta_bar}(x_t, t, t)`, the EMA copy at the boundary)
- `v(x_t, t)` is the **instantaneous marginal velocity field**,
  defined by Lemma 1 as `v(x, t) = E_{x_0 ~ p(x_0 | x_t = x)}[v_cond]`

The bias b is **deterministic given x_t** (both v_hat and v are
conditioned on x_t). The paper estimates ||b||^2 via the
EMA-tracking proxy:

> ||b||^2_hat := E\_{x_t}[||u_theta(x_t, t, t) - u\_{theta_bar}(x_t, t, t)||^2]

which is the gap between current params and EMA params at the
boundary. This is NOT the true bias — it's a practical upper bound
that the probe_dit_checkpoint.py measurement already uses.

### sigma^2 / noise term

From notation table and Section 3.1:

> Sigma\_{v'} := Cov\_{x_0 | x_t}[v']

where `v' = v_cond - v(x, t)` is the conditional velocity fluctuation
(zero-mean by Lemma 1). Under the scalar-isotropic approximation:

> Sigma\_{v'} ~ sigma^2 * I_d

so `sigma^2 * d = Tr(Sigma_{v'})` is the total conditional variance.

The paper reports `sigma^2 * d ~ 8.2e3` from `Tr(Sigma_{v'})` on
a DiT checkpoint batch (dit-bias-shrinkage table, Appendix C.3).

### beta\* formula

From Theorem 3 (Eq. 10):

> beta\* = [kappa / (kappa + 1)] * [sigma^2 * d / (sigma^2 * d + ||b||^2)]

Two factors:

1. **Noise-cancellation** `kappa/(kappa+1)`: determined by the Jacobi
   factor `J = (t-r) * d_{x_t} u_theta - I_d ~ kappa * I_d`
2. **James-Stein shrinkage** `sigma^2 d / (sigma^2 d + ||b||^2)`:
   pulls beta\* toward the unbiased corner when proxy bias is large

The matrix-form upper bound (no-bias variant) from the paper:

> beta\*_no_bias := Tr(J Sigma_{v'} (J+I)^T) / Tr((J+I) Sigma\_{v'} (J+I)^T) ~ 0.94

This is measured using `u_theta(x_t, t, t)` as the marginal-velocity
proxy (probe_dit_checkpoint.py, line 274). The true beta\*\_matrix
lies in (0, 0.94\].

### What v_ref must be

**b is defined against v(x_t, t) = E[v_cond | x_t, t], the
instantaneous marginal velocity.** Therefore v_ref must converge to
v(x_t, t). This is exactly what a **standard conditional
flow-matching model** (rectified flow) learns:

> L_FM = E\_{t, x_0, x_1} ||v_theta(x_t, t) - v_cond||^2

At convergence: `v_theta -> E[v_cond | x_t, t] = v(x_t, t)`.

**v_ref must NOT be a MeanFlow model** (which predicts the *average*
velocity u(x, r, t) over a time interval, not the instantaneous
velocity). Using a MeanFlow model's boundary value u(x_t, t, t) as
v_ref is the shortcut the paper already uses for the probe — but it's
circular for estimating ||b||^2 because the bias IS the gap between
u_theta(x_t, t, t) and v(x_t, t).

## Task 2 — v_ref Training Plan

### Existing FM support in the codebase

The repo has **no standalone FM/rectified-flow training mode**. The
closest is the **FM anchor** in `MeanFlowDiTModel` (meanflow.py
line 1417-1446): an auxiliary regression `u_theta(x_t, t-delta, t) -> v_cond` at small delta. But this is a secondary loss, not a
primary training objective.

### Minimal implementation (DONE)

Added behind a flag — **3 files changed, ~90 lines total**:

#### 1. `meanflow.py` — `fm_only` flag on `MeanFlowDiTModel`

- `fm_only: bool = False` in constructor (line 1002)
- `fm_only_cfg: bool = False` — whether to keep CFG class dropout
- `_fm_only_training_step()` method (inserted before `forward()`)

When `fm_only=True`, `training_step` dispatches to
`_fm_only_training_step` which:

- Evaluates `u = network(z, timestamps_at_boundary)` (r = t)
- Computes `loss = mean(||u - v_cond||^2)`
- No JVP, no adaptive weighting, no EMA tangent, no FM anchor
- Single forward pass per step (vs 2-3 for MeanFlow)
- Logs the same scalar keys for compatibility with experiment.py

**Reuses:** same DiT-B/4 backbone, same VAE latent pipeline, same
t-sampler (logit-normal, mean=-0.4, std=1.0), same data pipeline,
same optimizer (AdamW, lr=1e-4, wd=0), same EMA (0.9999), same
checkpoint infrastructure (Orbax).

#### 2. `config.py` — `fm_dit_imagenet_256_latent()`

- Inherits from `meanflow_dit_imagenet_256_latent()`
- Sets `fm_only=True`, `fm_only_cfg=False`
- `exp_name = "fm_dit_b4_imagenet_256_latent"`
- `num_train_steps = 200_000` (v_ref needs fewer steps)
- `checkpoint_every_n_steps = 5_000` (preemption tolerance)
- `max_checkpoints_to_keep = 5`

#### 3. `vamf/experiments/run_toy.py` — `"fm"` method

- Added to the method enum
- In `_loss_fn_and_aux`: when `method == "fm"`, evaluates
  `u_fn(z, t, t)` and computes `||u - v_cond||^2` directly
  (no JVP, no EMA tangent)

### ||b||^2 logging during training

The FM loss IS the ||b||^2 probe:

> E||v_cond - v_ref(x_t, t)||^2 = E||v'||^2 + ||b_ref||^2
> = sigma^2 * d + ||b_ref||^2

At convergence, `||b_ref||^2 -> 0` and the loss plateaus at
`sigma^2 * d` (the irreducible noise floor). The loss curve thus
serves double duty:

1. **Convergence monitor:** plateau = v_ref is trained
2. **sigma^2 * d estimate:** the plateau value

The actual ||b||^2 of a *MeanFlow* model is a separate post-hoc
computation: load both models, compute
`E||u_MF(x_t, t, t) - v_ref(x_t, t)||^2` on a batch. This is
Step 3 work.

The training loss is already logged to wandb by experiment.py at
every `log_every_n_steps=50`. The eval-step FID is also logged,
but FID is not meaningful for v_ref (it's not a generative model
in the MeanFlow sense — it predicts instantaneous velocity, not
average velocity for one-step generation).

## Task 3 — Smoke Test Results

### Toy/DGMM (CPU, local)

**Checkerboard (2D), 500 steps:**

```
step    loss     SW1
   0   13.336   1.166
 100   11.773   1.129
 200    9.213   1.145
 300   10.747   1.132
 400    9.274   1.085
 499    9.293   1.084
```

Loss decreases ~30% (13.3 -> 9.3). SW1 improves (1.17 -> 1.08).
FM objective working correctly — no JVP, no MeanFlow identity.

**DGMM-16 (16D), 500 steps:**

```
step    loss     SW1
   0   22.005   0.298
 100   14.092   0.296
 200   13.738   0.293
 300   13.321   0.280
 400   13.383   0.270
 499   14.110   0.255
```

Loss decreases ~36% (22.0 -> 14.1). Higher-dimensional loss is
larger as expected (more conditional variance in 16D). SW1 improves
(0.30 -> 0.26). FM objective verified at d=16.

### DiT/ImageNet (config verification only)

The config `fm_dit_imagenet_256_latent()` builds correctly and
inherits all fields from the baseline:

```
exp_name:           fm_dit_b4_imagenet_256_latent
fm_only:            True
fm_only_cfg:        False
num_train_steps:    200000
checkpoint_every:   5000
max_ckpts:          5
ema_rate:           0.9999
timestamp_sampler:  logit-normal
```

**Cannot run the ImageNet pipeline spin-up locally** — the DiT-B/4 +
SD-VAE requires more memory than the local M1 GPU. The pipeline
spin-up (a handful of steps) should be done on the TPU VM as the
first step of Step 2, before the full run.

### Existing tests

All 12 existing VaMF tests pass (test_vamf.py, 272s on CPU).
The `fm_only` flag is backward-compatible — defaults to False,
so existing MeanFlow/VaMF configs are unaffected.

### Checkpoint + resume

The FM-only mode uses the same `training_step` -> `experiment.py`
-> Orbax checkpoint pipeline as MeanFlow. No code path changes for
checkpoint save/restore. The checkpoint format is identical (same
`params/` + `state/` structure). Resume is handled by
`experiment.py` lines 350-400: restore state -> continue from
`state.step`.

**Preemption tolerance design:**

- `checkpoint_every_n_steps = 5_000` (vs 10k for MeanFlow) means
  max ~45 min of lost work on preemption (at ~1.8 steps/sec)
- `max_checkpoints_to_keep = 5` keeps 25k steps of history
- Resume command (identical to MeanFlow, just different config):
  ```bash
  gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
      --project research-481912 --zone us-central2-b --worker all \
      --command "cd pdt-research && \
          HF_HOME='/dev/shm/data/huggingface' HF_TOKEN=\$HF_TOKEN \
          bazelisk run --config=tpu //src/projects/generative:main -- \
              --distributed=true \
              --experiment config:fm_dit_imagenet_256_latent \
              --experiment set:exp_name=\"'fm_dit_b4_vref'\" \
              --work_dir=gs://pdt_gen_ai/juanwu"
  ```
- To resume from a checkpoint after preemption, add:
  ```
  --experiment set:trainer.checkpoint_dir=\"'gs://pdt_gen_ai/juanwu/fm_dit_b4_vref/checkpoints/<step>'\""
  ```

**Cannot run the checkpoint + kill + resume cycle locally** since
the DiT pipeline doesn't fit on local hardware. This should be
verified as part of the TPU pipeline spin-up in Step 2 (run ~100
steps, checkpoint, kill, resume, confirm step counter continuity).

## Task 4 — Full-Run Cost Estimate

### VM topology

- **VM:** `tpu-v4-32-us-central2-b`
- **Chips:** 32 TPU v4 chips (4 workers x 8 chips/worker)
- **Project:** `research-481912`, zone `us-central2-b`

### Steps/sec estimate

From the paper's experiment section (Appendix C.3):

- Baseline MeanFlow: **2.65 steps/sec** (DiT-B/4, batch=256, TPU v4-32)
- VaMF beta=1: **2.20 steps/sec** (22% overhead from extra EMA forward + FM anchor)
- VaMF beta=0.25/0.5: ~2.4 steps/sec (10% overhead from EMA forward only)

The FM-only mode is **simpler than baseline MeanFlow**:

- **No JVP computation** (saves the most expensive part — the
  Jacobian-vector product through the DiT backbone)
- **No CFG evaluation** (no need to compute v_uncond and v_cond
  separately — the target is v = e - x directly)
- **Single forward pass per step** (vs 2 forward passes for MeanFlow:
  one for the JVP tangent CFG velocity, one for the main MeanFlow
  prediction)

Estimated speed: **~3.5-4.0 steps/sec** on TPU v4-32.
Conservative estimate: **3.0 steps/sec** (assuming overhead from
data loading, checkpoint I/O, wandb logging).

**Cannot measure exact steps/sec without the VM.** The above is
extrapolated from the fact that FM removes the JVP (which accounts
for ~50% of MeanFlow's per-step cost) and one CFG forward pass.
The first ~100 steps of the pipeline spin-up in Step 2 will give
the actual number.

### Projected wall-clock

| Horizon | Steps   | Est. wall-clock (@ 3.0 s/s) | Est. wall-clock (@ 4.0 s/s) |
| ------- | ------- | --------------------------- | --------------------------- |
| 50k     | 50,000  | 4.6 hours                   | 3.5 hours                   |
| 100k    | 100,000 | 9.3 hours                   | 6.9 hours                   |
| 200k    | 200,000 | 18.5 hours                  | 13.9 hours                  |

**v_ref likely converges faster than MeanFlow's 300k.** The FM loss
has no MeanFlow identity / JVP complexity — it's a simple regression.
The ||b||^2 curve (= FM loss minus sigma^2\*d) will show when v_ref
stabilizes. The config targets 200k steps as an upper bound; if the
||b||^2 curve plateaus earlier (plausible at ~50-100k), the run can
be stopped.

### Checkpoint cadence

- Every 5k steps (~28 min at 3.0 s/s, ~21 min at 4.0 s/s)
- Max 5 checkpoints kept (25k steps of history)
- Each checkpoint: ~1.8 GiB (DiT-B/4 has ~130M params, float32)

### Resume command

```bash
# Fresh start
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker all \
    --command "cd pdt-research && \
        HF_HOME='/dev/shm/data/huggingface' HF_TOKEN=\$HF_TOKEN \
        bazelisk run --config=tpu //src/projects/generative:main -- \
            --distributed=true \
            --experiment config:fm_dit_imagenet_256_latent \
            --experiment set:exp_name=\"'fm_dit_b4_vref'\" \
            --work_dir=gs://pdt_gen_ai/juanwu"

# Resume from checkpoint (after preemption)
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker all \
    --command "cd pdt-research && \
        HF_HOME='/dev/shm/data/huggingface' HF_TOKEN=\$HF_TOKEN \
        bazelisk run --config=tpu //src/projects/generative:main -- \
            --distributed=true \
            --experiment config:fm_dit_imagenet_256_latent \
            --experiment set:exp_name=\"'fm_dit_b4_vref'\" \
            --experiment set:trainer.checkpoint_dir=\"'gs://pdt_gen_ai/juanwu/fm_dit_b4_vref/checkpoints/<STEP>'\" \
            --work_dir=gs://pdt_gen_ai/juanwu"
```

Sync code to all workers before launching (same as any experiment):

```bash
# 1. Push to GitHub locally
# 2. Pull on primary worker
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker 0 \
    --command "cd pdt-research && git fetch -a && git pull --rebase"
# 3. Fan out to other workers
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker 0 \
    --ssh-flag="-A" \
    --command "./sync_folder.sh /home/juanwu/pdt-research"
```

## Hard Gate

**The full v_ref training run is NOT launched.** Step 2 (full run)
and Step 3 (||b||^2 estimator + beta\* computation) await explicit
human go. The implementation is complete and smoke-tested; the cost
estimate and resume commands are ready for the budget decision.
