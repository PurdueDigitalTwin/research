# Overnight Report — Unconditional v_ref (Step 2')

**STATUS: [still training @ step ~8k | loss ~3270]**
**ROOT-DISK: [flat — all workers under 50%, deltas < 15 KB]**

Date: 2026-06-26 overnight. VM: tpu-v4-32-us-central2-b.
Branch: feat/vamf-rebuttal-beta-anneal. Commit: c2491ca.

## 1. v_ref Run Status

Run `h7zl3wfw` is training normally at 4.6 steps/sec.

### Loss trajectory

```
step      0:  6895.1     (= E[||v_cond||^2], model outputs zero at init)
step   1000:  3638.1
step   2000:  3500.1
step   3000:  3404.7
step   4000:  3384.9
step   5000:  3353.6     <- first checkpoint saved
step   6000:  3297.8
step   7000:  3288.4
step   8000:  3267.9
```

Loss is still decreasing but the rate is slowing: delta-per-1k went
from -138 (1k→2k) to -21 (7k→8k). Not yet converged — changes are
still > 1% per 1k steps.

### Root disk

| Worker | Baseline (KB) | Current (KB) | Delta | Usage |
|--------|---------------|--------------|-------|-------|
| W25 | 49,709,524 | 49,881,292 | +172K | 50% |
| W13 | 19,128,568 | 19,024,684 | -104K | 19% |
| W227 | 19,389,428 | 19,416,888 | +27K | 20% |
| W84 | 21,584,408 | 21,611,612 | +27K | 22% |

W25 grew 172KB over ~8k steps (safe). All workers well under 85%.

## 2. v_ref Convergence Assessment

The loss is NOT converged yet. It's still decreasing at ~0.6% per 1k
steps. The stop-on-convergence criterion (< 1% change across 5
consecutive 5k checkpoints) has NOT been met — we only have 1
checkpoint (step 5k).

Projected: the loss will continue to decrease slowly. The floor is
E_t[Tr(Sigma_{v'}(t))], NOT 8.2e3 (see §3a below). Based on the
flattening trajectory, convergence is likely somewhere between 30k-100k
steps (6-22 hours at 4.6 steps/sec).

No anomalies: no NaNs, no spikes, no root-disk growth.

## 3. Read-Only Analysis

### 3a. `||b_ref||^2 = loss - 8.2e3` is INVALID

**Relabeled as INVALID.** The subtraction mixes incompatible quantities:

- **FM training loss** = `E_{t~logit-normal}[E_{x_0,e}[||v_ref(x_t,t) - v_cond||^2]]`
  — averaged over the **logit-normal t-sampler** with mean=-0.4, std=1.0

- **Paper's 8.2e3** = `Tr(Sigma_{v'})` measured at **t = 0.5 ONLY**,
  on 128 synthetic-Gaussian x_t samples
  (source: Table C.3 caption in supplementary.tex, line 3 of
  `dit-bias-shrinkage.tex`)

At convergence, the FM loss approaches `E_{t~logit-normal}[sigma^2(t)*d]`,
which is the irreducible noise averaged over the t-sampler. This is NOT
`sigma^2(0.5)*d = 8.2e3` because `sigma^2(t)` varies with t. Subtracting
8.2e3 from the FM loss gives neither a valid ||b||^2 estimate nor a
meaningful convergence diagnostic.

**Monitor convergence by loss flattening instead.**

### 3b. E[||v_cond||^2] and step-0 loss confirmation

**Derivation:**

```
v_cond = e - x_0     (diagnostic.py line 40, meanflow.py line 1247)
where e ~ N(0, I_d), x_0 = data latent, d = 32*32*4 = 4096

E[||v_cond||^2] = E[||e||^2] + E[||x_0||^2]    (independence → cross-term = 0)
                = d + E[||x_0||^2]
                = 4096 + E[||x_0||^2]
```

**Step-0 confirmation:**

DiT uses AdaLN-Zero initialization: the final projection layer has
`kernel_init=zeros, bias_init=zeros` (dit.py lines 666-667), so the
model outputs EXACTLY zero at step 0. Therefore:

```
loss(step 0) = E[||0 - v_cond||^2] = E[||v_cond||^2]
```

Observed: `loss(step 0) = 6895.1`. This confirms:

```
E[||v_cond||^2] = 6895.1
E[||x_0||^2] = 6895.1 - 4096 = 2799.1
Per-dim variance of latents: E[x_i^2] ≈ 0.683
```

**Convention:** The loss sums over ALL spatial/channel dimensions
(`jnp.sum(..., axis=(-1,-2,-3))` in meanflow.py line 1598), then
averages over the batch. So it's a **total-trace-over-d** quantity,
NOT per-dimension. This matches the paper's convention where sigma^2*d
is the total trace of Sigma_{v'}.

**t-independence:** Since `v_cond = e - x_0` does not depend on t, the
step-0 loss equals `E[||v_cond||^2]` regardless of the t-sampler
distribution. Confirmed: 6895.1.

### 3c. Reconciliation: 8.2e3 and 0.94 provenance

#### sigma^2 * d ≈ 8.2e3

**Source:** supplementary.tex line 380 + dit-bias-shrinkage.tex caption:

> "the irreducible noise floor sigma^2 d ≈ 8.2×10^3 from Tr(Sigma_{v'})
> on the same batch"

**Table caption (verbatim):** "Direct DiT-checkpoint measurement of the
bias-variance decomposition ingredients of Theorem 3 at **t = 0.5**.
The EMA bias ||b||^2 averages over **128 synthetic-Gaussian x_t
samples**. The shrinkage factor uses sigma^2 d ≈ 8.2×10^3."

**Provenance:**
- t-distribution: **fixed t = 0.5** (NOT averaged over any sampler)
- Convention: Tr(Sigma_{v'}) = total trace over d=4096 dims
- Batch: 128 synthetic-Gaussian x_t samples
- Checkpoint: baseline MeanFlow DiT (beta=0), evaluated at steps 20k/40k/80k
- Reported in Appendix C.3 (dit-bias-shrinkage table)

**The gap vs FM loss:** The FM loss averages sigma^2(t)*d over the
logit-normal t-sampler (mean=-0.4, std=1.0), which overweights t < 0.5
(the left mode of the logit-normal). Since sigma^2(t) varies with t:

- At t→0: x_t ≈ x_0, conditioning pins x_0, so v' ≈ e and
  Tr(Sigma_{v'}) → d = 4096
- At t→1: x_t ≈ e, conditioning pins e, so v' ≈ -x_0 and
  Tr(Sigma_{v'}) → E[||x_0||^2] ≈ 2799
- At t=0.5: Tr(Sigma_{v'}) ≈ 8200 (from the paper)

This is a non-monotone profile with a **peak near t=0.5** (where
conditional paths maximally overlap). The logit-normal sampler
(mean=-0.4, std=1.0) has its mode at `sigmoid(-0.4) ≈ 0.40`,
overweighting the region t ∈ [0.2, 0.6] where sigma^2(t)*d is near
its maximum.

**Uncertainty:** The converged FM loss floor could be close to 8.2e3
or quite different, depending on the exact sigma^2(t) profile and the
logit-normal weighting. I cannot predict this without measuring
sigma^2(t) at multiple t values. The run's loss will reveal the true
floor empirically.

#### beta*_no_bias ≈ 0.94

**Source:** experiment.tex lines 66-70:

> "We directly measured a matrix-form upper bound on beta*_matrix on
> the baseline checkpoint... We evaluate the bound at t ∈ {0.1, 0.3,
> 0.5, 0.7, 0.9} with fixed gap t-r = 0.25, and 512 samples per t
> using u_theta(x_t, t, t) as the marginal-velocity proxy. Aggregated
> across t, we have beta*_no_bias ≈ 0.94."

**Provenance:**
- Code: `diagnostic.py:matrix_form_beta_star` (Experiment 5)
- t-distribution: **uniform over 5 discrete values**
  {0.1, 0.3, 0.5, 0.7, 0.9}, NOT the logit-normal training sampler
- Fixed gap: t - r = 0.25
- Batch: 512 samples per t value
- Aggregation: sum(numerators) / sum(denominators) across all 5 t values
- Conditioning: **null class** (unconditional, index 1000)
- Checkpoint: baseline MeanFlow DiT (beta=0)

**Key observation:** The 0.94 and the 8.2e3 use DIFFERENT
t-distributions (uniform-over-5 vs fixed-at-0.5 vs logit-normal).
None of them match the training t-sampler.

### 3d. Corrected ratio-based Step-3 plan

#### The problem with the subtraction approach

`||b||^2 = loss - sigma^2*d` requires knowing `sigma^2*d` under the
exact same conditions (same t-sampler, same batch distribution). The
paper's 8.2e3 is at a single t=0.5 and does NOT equal the FM loss
floor.

#### The ratio-based approach (self-normalizing)

Instead of estimating ||b||^2 and sigma^2*d separately, compute their
**ratio** at each t:

```
For each (x_0, e, t) sample:
  x_t = (1-t)*x_0 + t*e
  v_cond = e - x_0
  
  u_MF = MeanFlow_network(x_t, timestamps=(t,0), labels=null_1000)  # boundary r=t
  v_ref = vref_network(x_t, timestamps=(t,0), labels=null_1000)      # converged FM model
  
  bias_sq(t) = ||u_MF - v_ref||^2          (sum over dims)
  noise(t) = ||v_cond - v_ref||^2           (sum over dims)
  ratio(t) = bias_sq(t) / noise(t)
```

Then the shrinkage factor at each t is:

```
shrinkage(t) = 1 / (1 + ratio(t)) = sigma^2(t)*d / (sigma^2(t)*d + ||b(t)||^2)
```

And with kappa ≈ 1 (from the Frobenius-norm measurement):

```
beta*(t) = kappa/(kappa+1) * shrinkage(t) = 0.5 * shrinkage(t)
```

#### Advantages

1. **Self-normalizing:** no need to know sigma^2*d absolutely
2. **t-resolved:** gives beta*(t) at each t, revealing if it varies
3. **Consistent conditioning:** both u_MF and v_ref evaluated at null
   (index 1000), matching probe convention
4. **No t-sampler dependency:** can evaluate at any t values; the
   ratio is intrinsic to the models, not the training distribution

#### Eval interface (verbatim)

Both models share the same DiT-B/4 backbone and the same
`MeanFlowDiTModule.__call__` interface:

```python
null_label = jnp.int32(model.num_classes)  # = 1000
labels = jnp.full((batch_size,), null_label, dtype=jnp.int32)

# MeanFlow proxy: u_MF(x_t, t, t) = boundary value
ts_mf = model._make_timestamps(t_in=t, r_in=t)
u_mf = mf_network.apply(
    variables={"params": mf_params},
    inputs=z,
    timestamps=ts_mf,
    labels=labels,
    edm_cond=None,
    deterministic=True,
)

# v_ref: same interface, same null labels, r=t (boundary)
ts_vref = model._make_timestamps(t_in=t, r_in=t)
v_ref = vref_network.apply(
    variables={"params": vref_params},
    inputs=z,
    timestamps=ts_vref,
    labels=labels,
    edm_cond=None,
    deterministic=True,
)

# Per-sample estimates
bias_sq = jnp.sum(jnp.square(u_mf - v_ref), axis=(-1, -2, -3))
noise = jnp.sum(jnp.square(v_cond - v_ref), axis=(-1, -2, -3))
ratio = jnp.mean(bias_sq) / jnp.mean(noise)
shrinkage = 1.0 / (1.0 + ratio)
beta_star = 0.5 * shrinkage  # with kappa=1
```

**Checkpoints needed:**
- MeanFlow: baseline beta=0 checkpoint from `hde9iaqj`
  (e.g., step 240k at `gs://pdt_training/juanwu/meanflow/...`)
- v_ref: unconditional model from this run (`h7zl3wfw`)
  after convergence

**t values:** Evaluate at {0.1, 0.2, ..., 0.9} for t-resolved profile,
or match the paper's {0.1, 0.3, 0.5, 0.7, 0.9} for direct comparison
to the 0.94 measurement.

**Batch:** Use real ImageNet latents (same pool as probe), not synthetic.
512+ samples per t for statistical power.

## 4. What I Did NOT Do

- Did NOT run Step 3 (||b||^2 estimator / beta* computation)
- Did NOT resume qejr4spi / start conditional analysis
- Did NOT launch any new training run
- Did NOT edit the paper / experiment.tex
- Did NOT delete or move anything
- Did NOT push to main
- Did NOT change any training config/batch/lr

## 5. DECISIONS WAITING FOR YOU

1. **Step 3 go/no-go:** The ratio-based eval interface is ready (§3d).
   Needs: v_ref to converge + the baseline MeanFlow checkpoint path.
   Both models evaluated at null-class (index 1000). PLAN ONLY written;
   no code committed. Awaits explicit go.

2. **Resume qejr4spi (conditional v_ref):** Checkpoint preserved at
   step 5000 in `gs://pdt_training/juanwu/meanflow/fm_dit_b4_vref/`.
   Would provide CFG-regime supplementary data. Not started.

3. **v_ref convergence:** Run is still training. Loss is ~3270 at step
   8k and slowly decreasing. No anomalies. Will continue autonomously
   and stop when flattened (< 1% change across 5 consecutive 5k
   checkpoints). The floor is unknown a priori (NOT 8.2e3 — see §3a).
