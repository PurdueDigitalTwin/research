# B1 Beta-Anneal Handoff

Experiment: annealed tangent-mixing coefficient beta(step) for VaMF.
Branch: `feat/vamf-rebuttal-beta-anneal`

## Task A: `beta_schedule.py`

**Status: DONE**

Created `src/projects/generative/vamf/model/beta_schedule.py`:

- `beta_at_step()` — host-side (pure Python), returns float. Used by the
  toy trainer where beta is computed on the host each step and passed as a
  traced `jnp.float32` scalar into the jitted train step.
- `jax_beta_at_step()` — JAX-traceable (`jnp` arithmetic). Used by the
  DiT trainer where beta must be computed *inside* `training_step` from
  `state.step` because `experiment.py` wraps the step function with
  `functools.partial + jax.pmap` and the signature is fixed to
  `(state, batch)`.

Supported shapes: `constant`, `linear`, `cosine`, `step`.
All shapes windowed by `[s0, s1]` fraction of `total_steps`.
Includes inline smoke test (boundary values, [0,1] range, monotone
non-increasing, constant backward-compat).

Registered in `src/projects/generative/vamf/model/BUILD`.

## Task B: `run_toy.py` changes

**Status: DONE**

Six changes to `src/projects/generative/vamf/experiments/run_toy.py`:

1. Import `beta_schedule`
2. Five new flags: `beta_anneal_shape`, `beta_start`, `beta_end`,
   `beta_anneal_s0`, `beta_anneal_s1`
3. `_loss_fn_and_aux(..., tangent_beta=None)` — accepts override;
   falls back to `self._tangent_beta` when `None`
4. `training_step` / `compute_gradient` — forward `tangent_beta` kwarg
5. `_train_step(state, key, beta_val)` / `_grad_probe(state, key, beta_val)` —
   accept beta; train loop computes it host-side each step
6. Output filenames encode schedule params; args dict includes all 5 fields

**Bug found and fixed:** The original call site passed
`beta_start=FLAGS.beta_start` (default 1.0) to `beta_at_step()` for ALL
shapes including `constant`. This made `--tangent_beta=0.4` silently run
at `beta=1.0`. Fix: short-circuit `shape=="constant"` to use
`FLAGS.tangent_beta` directly, matching the DiT trainer.

Backward-compat verified: `--method=meanflow` and
`--method=vamf_tmix --beta_anneal_shape=constant --tangent_beta=0.0`
produce identical step-0 outputs (loss, SW1).

## Task C: Phase 1 Sweep (DGMM-64)

**Status: IN PROGRESS**

10 configs x 3 seeds = 30 runs, 200k steps each.
Running on local CPU via `ProcessPoolExecutor(max_workers=3)`.

Configs:
| Label | Method | Shape | beta | s1 |
|---|---|---|---|---|
| baseline | meanflow | constant | 1.0 | 0.6 |
| static_b0.4 | vamf_tmix | constant | 0.4 | 0.6 |
| static_b1.0 | vamf_tmix | constant | 1.0 | 0.6 |
| linear_s0.3 | vamf_tmix | linear | 1.0 | 0.3 |
| linear_s0.6 | vamf_tmix | linear | 1.0 | 0.6 |
| linear_s0.8 | vamf_tmix | linear | 1.0 | 0.8 |
| cosine_s0.3 | vamf_tmix | cosine | 1.0 | 0.3 |
| cosine_s0.6 | vamf_tmix | cosine | 1.0 | 0.6 |
| cosine_s0.8 | vamf_tmix | cosine | 1.0 | 0.8 |
| step_s0.6 | vamf_tmix | step | 1.0 | 0.6 |

## Task D: Phase 1 Analysis

**Status: DONE — VERDICT: KILL**

All 30 runs (10 configs x 3 seeds) completed. Every non-baseline config
missed baseline SW1.

```
Config          SW1 (mean±SEM)    1st-half nr    Tier
─────────────────────────────────────────────────────
baseline        0.0269 ± 0.0013   0.60           —
static_b0.4     0.0288 ± 0.0023   0.63           MISS
static_b1.0     0.0324 ± 0.0007   0.68           MISS
linear_s0.3     0.0316 ± 0.0008   0.61           MISS
linear_s0.6     0.0323 ± 0.0007   0.64           MISS
linear_s0.8     0.0330 ± 0.0007   0.65           MISS
cosine_s0.3     0.0318 ± 0.0007   0.61           MISS
cosine_s0.6     0.0322 ± 0.0007   0.64           MISS
cosine_s0.8     0.0328 ± 0.0007   0.65           MISS
step_s0.6       0.0340 ± 0.0007   0.68           MISS
```

Key findings:

1. **Monotonic quality degradation**: More time at beta > 0 → worse SW1.
   Recovery tail length is the only variable that matters.
2. **Shape is irrelevant**: linear and cosine at the same s1 produce
   nearly identical results (e.g., linear_s0.6=0.0323 vs cosine_s0.6=0.0322).
3. **No noise ratio improvement**: Every config has nr >= baseline (0.60).
   Annealing does NOT reduce gradient variance on this problem.
4. **static_b0.4 is closest** to baseline (0.0288 vs 0.0269) but still
   outside the 1-SEM band (0.0282).
5. **step shape is worst** (0.0340) — abrupt transitions hurt more than
   smooth ones.

Conclusion: beta-annealing provides no early-training stability benefit
on DGMM-64 that offsets the quality cost of running at beta > 0.
**Do NOT spend TPU hours on Phase 2.**

## Task E: Preservation Pass

**Status: BLOCKED**

All DiT beta-sweep checkpoint directories in GCS are empty (0 bytes):
- `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/vamf_*/` — empty
- Only the baseline run at
  `gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent/meanflow/dit_imagenet_256_latent/`
  has actual data (37.74 GiB, checkpoints at 10k-210k steps).

Cannot generate samples or dump probes from nonexistent checkpoints.

## Task F: DiT Trainer (Phase 2 Staging)

**Status: DONE**

Changes in `src/projects/generative/meanflow.py`:

1. `MeanFlowDiTModel.__init__`: 6 new anneal params
   (`beta_anneal_shape`, `beta_anneal_start`, `beta_anneal_end`,
   `beta_anneal_s0`, `beta_anneal_s1`, `beta_anneal_total_steps`)
2. `training_step` (~line 1246): When `shape != "constant"`, computes
   `beta = jax_beta_at_step(state.step, ...)` inside the jitted step.
   When `shape == "constant"`, uses `self.tangent_beta` directly
   (no schedule function, correct by construction).
3. Beta-mixing section (~line 1303): Uses `isinstance(beta, float)` to
   branch. Python float skips unused evals; traced JAX scalar evaluates
   both branches and mixes with arithmetic (no `jax.lax.cond` needed).

New config: `vamf_b1_anneal_dit_imagenet_256_latent()` in `config.py`.
Defaults: cosine shape, s1=0.6, checkpoint_every_n_steps=25000.

Registered `beta_schedule` dep in `src/projects/generative/BUILD`.

### Open questions answered

**Q1: How does beta enter the jitted step?**
Via `state.step` — a traced integer already used by `jax.random.fold_in`
at line 1151. `jax_beta_at_step` uses `jnp` arithmetic so the
computation traces without recompilation.

**Q2: Does the pmap wrapper allow extra args?**
No. `experiment.py:403` wraps `model.training_step` with
`functools.partial + jax.pmap` with fixed `(state, batch)` signature.
Solution: compute beta inside `training_step` from `state.step`.

**Q3: FM anchor interaction?**
`self.fm_anchor_weight` is a static Python float set at init. It stays
CONSTANT — only the tangent beta is annealed. No interaction.

## Phase 2 Launch Commands

**NOT RECOMMENDED.** Phase 1 returned KILL — no schedule beat baseline.

The infrastructure is staged if you want to override (commit `ddeb06b`),
but the data says every schedule degrades quality monotonically.

```bash
# Example command (DO NOT RUN without strong justification):
gcloud compute tpus tpu-vm ssh juanwu@tpu-v4-32-us-central2-b \
    --project research-481912 --zone us-central2-b --worker all \
    --command "cd pdt-research && \
        HF_HOME='/dev/shm/data/huggingface' HF_TOKEN=\$HF_TOKEN \
        bazelisk run --config=tpu //src/projects/generative:main -- \
            --distributed=true \
            --experiment config:vamf_b1_anneal_dit_imagenet_256_latent \
            --experiment set:exp_name=\"'vamf_b1_<SHAPE>_s<S1>'\" \
            --experiment set:model.beta_anneal_shape=\"'<SHAPE>'\" \
            --experiment set:model.beta_anneal_s1=<S1> \
            --work_dir=gs://pdt_training/juanwu/meanflow/dit_imagenet_256_latent"
```

## Verdict

**KILL.** Phase 1 screening on DGMM-64 shows beta-annealing
monotonically degrades sample quality (SW1) with no compensating noise
ratio reduction. The Phase 2 diff is staged but should NOT be launched.
TPU hours are better spent on other experiments.

## Commits

```
893529d feat: Add beta-anneal schedule for VAMF B1 experiment
ddeb06b feat: Stage Phase 2 beta-anneal for DiT MeanFlow trainer
e73b108 fix: Use tangent_beta directly for constant anneal shape in toy trainer
```
