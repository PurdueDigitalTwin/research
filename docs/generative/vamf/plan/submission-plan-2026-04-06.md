# VaMF NeurIPS 2026 Submission Plan

**Date:** 2026-04-06
**Author:** Claude (with Juanwu)
**Target venue:** NeurIPS 2026 (OpenReview)
**Deadlines:**
- Abstract registration: **2026-05-04 (Mon)**
- Full paper PDF: **2026-05-06 (Wed)**

**Days remaining at authoring:**
- 28 days to abstract deadline
- 30 days to full paper deadline

---

## 1. Context at authoring time

### 1.1 Paper state

The LaTeX draft lives at `docs/generative/vamf/` with `main.tex` + eight
`contents/*.tex` files. The `pdt_report.sty` style file and
`reference.bib` (57 entries) are in place. A compiled PDF exists at
`release/target/main.pdf` (mtime 2026-04-04).

| File | Lines | Status |
|---|---:|---|
| `main.tex` | 56 | Done (structure + includes) |
| `contents/abstract.tex` | 6 | Prose done, contains `\todo[inline]{Add experiment results later...}` |
| `contents/introduction.tex` | 15 | **Done** (three-paragraph + contribution bullets) |
| `contents/related.tex` | 10 | **Done** (four paragraphs: FM, one-step, MF variants, variance reduction) |
| `contents/method.tex` | 270 | **Done.** Structure matches 2026-04-02 structural review: D1/D2/D3 design principles, NLL variant primary, MSE variant in appendix, Propositions 1–3 + 6 in main, Algorithm 2 (NLL) in main text |
| `contents/experiment.tex` | 21 | **Placeholder** (`\lipsum[9]\lipsum[10]` + sample table) |
| `contents/conclusion.tex` | 4 | **Placeholder** (`\lipsum[11]\lipsum[12]`) |
| `contents/postscripts.tex` | 9 | Done (acknowledgement + bibliography) |
| `contents/X_appendix.tex` | 289 | **Done.** Lipschitz corollary, ANR proposition (Prop 6), Compound Prediction Error (Prop 4, d/dt vs d/ds), Algorithm 1 MSE variant |

**Title:** Currently `On Variances Reduction in Training Mean Flows`.
Must be `On Variance Reduction in Training Mean Flows` (grammar fix
flagged by both the 2026-04-02 structural review and the 2026-04-04 mock
review). Alternative considered: `Reducing Gradient Variance in Mean
Flow Training` — pick one before the abstract deadline.

### 1.2 Reviews on record

- `reviews/structural-review-2026-04-02.md` — structural review flagging
  three issues (GVD disjoint, theory-method disconnect, two-method
  framing). **Most recommendations already reflected in the current
  `method.tex`** (D1/D2/D3 structure, NLL-as-primary, MSE-in-appendix,
  Proposition 1 as the punchline after the gradient variance bridge).
- `reviews/2026-04-04.md` — full simulated NeurIPS review scoring 4/10
  and recommending Reject/resubmit for W1 (no experiments). Strengths
  (clean unifying theory, rigorous proofs, thoughtful TVM comparison)
  are real; the disqualifying weakness is the lack of experiments. This
  review is the primary action list for §5.

### 1.3 Code state

The 2026-04-03 implementation plan's Phase 1 (core VaMF model), Phase 2
(variance head), and Phase 3.1 (configs) are done and merged on
`juanwu/meanflow`:

- `VAMeanFlowUNetModel` in `src/projects/generative/meanflow.py` with
  EMA tangent, FM anchor, variance head behind `predict_variance` flag,
  `nll_warmup_steps`, `variance_floor`.
- `vamf_unet_cifar_10` and `vamf_nll_unet_cifar_10` configs in
  `src/projects/generative/config.py` with `adaptive_weight_power=0.0`
  and `num_train_steps=800_000` (**must be cut**).
- **Phase 3.2 ablation flags are NOT in code yet** — `no_fm_anchor`,
  `no_ema_tangent`, `no_sg_jvp`, `boundary_tangent`, `stochastic_tangent`
  are listed in the 2026-04-03 plan but not implemented.

Recent relevant commits on `juanwu/meanflow`:
- `ac24875` — default config with SNR weighting for VaMF
- `e23936a` — SNR weighting implementation (fixes MSE scaling bug)
- `81dff20` — scalar logging fixes
- `8593eaf` — local RNG fix + gradient norms logging

### 1.4 Live run

Run ID `9k3bt7aa` (`vamf_with_snr_unet_cifar_10_20260406_131122`) on
TPU v4-32, launched after the 2026-04-06 MSE scaling bug fix.

At time of authoring:
- Step: 46,948 / 800k scheduled
- FID: **187.6** at step 45,000
- Descent rate: −22.9 FID per 2,500-step eval, averaged over last 5 evals
- Runtime: 8.68 h (5,407 steps/h)
- Loss balance: mf/fm ≈ 80/20 (matches Algorithm 2 intent)
- No anomalies; grad norm stable around 1,500–2,000

This is the VaMF-MSE entry for Table 1. Plan is to let it run to
~150k steps (≈ 14h more) and then stop.

### 1.5 Compute

**Two v4-32 VMs** available, both in `us-central2-b`, both checkpointing
to the shared bucket `gs://pdt_training/juanwu`:

| VM | Role (this plan) | Currently |
|---|---|---|
| `tpu-v4-32-us-central2-b` | VM-A — main-results slot | Running `9k3bt7aa` (R2, VaMF-MSE) |
| `behavior` | VM-B — NLL slot, then ablations | Idle; repo at old commit, needs `git pull` + `sync_folder.sh` before use (added 2026-04-06) |

Measured: 5,407 steps/h on current CIFAR-10 UNet backbone.

Per-VM derived budgets:
- 100k-step run ≈ 18.5 h
- 150k-step run ≈ 27.8 h
- 30 days × 24 h = 720 h per VM; conservative usable ≈ 500 h each after
  checkpoint restarts, debugging, and queue gaps

**Total compute across both VMs over 30 days ≈ 1,000 h** — enough for
~36 × 150k runs or ~54 × 100k runs with margin. Each run occupies an
entire v4-32 (all 4 workers), so effective concurrency is **2 runs at
once**, one per VM.

**VM assignment strategy:** keep VM-A on the main-results critical path
(R2 → R3 → R4 → R6) and VM-B on the NLL headline + ablations
(R1 → R5a → R5b → R5c → R5d → R5e). This keeps the R1 trajectory
(which gates the Apr 19 go/no-go decision) on a dedicated VM with no
queueing behind baselines.

---

## 2. Critical path

Experimental results for §5 are the single blocker. Writing is narrow:
abstract touch-up + experiment.tex + conclusion.tex + minor fixes.

Everything else — theory, related work, introduction, method, appendix —
is already in the draft at publishable quality. Do not rewrite them
unless an experiment forces a claim change.

---

## 3. Experimental program

Mapped one-to-one to the 2026-04-04 reviewer's weakness list (W1) so
that every demand has a concrete run assigned.

| ID | Reviewer demand | Run(s) | Purpose |
|---|---|---|---|
| E1 | FID/IS CIFAR-10: MF, iMF, AlphaFlow, Re-MeanFlow, VaMF-MSE, VaMF-NLL under identical backbone | R1 (VaMF-NLL), R2 (9k3bt7aa, VaMF-MSE), R3 (iMF), R4 (vanilla MF) | Headline Table 1 |
| E2 | Training loss curves showing resolution of non-decreasing loss | Falls out of E1 | Figure 1 |
| E3 | Component ablation: EMA alone, FM anchor alone, variance head alone, full | R5a (λ=0), R5b (boundary tangent), R5c (stochastic tangent), R5d (MSE λ=0), R5e (no variance head) | Table 2 |
| E4 | Curvature-variance empirical verification: `‖Δ‖` vs `(t−r)·‖Dv/Dt‖`; `σ²_θ` vs `(t−r)²` | Gaussian mixture closed-form toy + R1 diagnostics | Figure 2 |
| E5 | Gradient variance decomposition: MF vs VaMF, per-t, Jacobian-amplified vs irreducible | R6 instrumented (gradient cosine + per-t variance estimator + curvature proxy) | Figure 3 |
| E6 | Scale to ImageNet-64 or LSUN | R7 (VaMF-NLL ImageNet-64), **stretch** | Table 3 (if feasible) |

AlphaFlow and Re-MeanFlow baselines in E1: if re-implementation under
our UNet is not feasible in the window, report **published numbers
alongside** our iMF and vanilla MF reproductions, clearly labeled. The
reviewer's demand is "identical architecture and compute budgets" — we
should meet this for iMF and vanilla MF at minimum, which are the
closest baselines.

Reviewer questions Q1–Q5 should be answered as one-paragraph entries in
§5 or the appendix, grounded in E1–E6 numbers:

- **Q1** (EMA vs separate-head formal comparison): one paragraph in §5
  discussion, grounded in Table 1 comparing VaMF-NLL vs iMF under the
  same backbone.
- **Q2** (FM anchor δ recommendations + bias): appendix section with a
  small δ-sweep if compute permits, otherwise qualitative with
  `δ_max=0.01` justification.
- **Q3** (empirical `‖Dv/Dt‖` magnitude across t): covered by E4 figure.
- **Q4** (when variance head > α-schedule): §5 ablation discussion
  grounded in Table 2.
- **Q5** (multi-step sampling): §5 or §6 short paragraph + 1-step vs
  2-step column in Table 1.

---

## 4. Week-by-week schedule

### Week 1 — 2026-04-06 (Mon) → 04-12 (Sun)

Lock main-result runs and ablation scaffolding. With two VMs,
**R1 and R2 run in parallel from Apr 7 onward**, which removes
the R1-must-wait-for-R2 serialization in the single-VM plan and buys
roughly 28 h on the critical path.

| Date | VM-A (`tpu-v4-32-us-central2-b`) | VM-B (`behavior`) | Non-TPU work |
|---|---|---|---|
| Apr 6 (today) | `9k3bt7aa` (R2) continues to ~150k. | Sync: `git pull` + `sync_folder.sh` + smoke-test `bazelisk run` with a 100-step dry run to confirm the VM is healthy. | Audit the NLL code path (`VAMeanFlowUNetModel` with `predict_variance=True`) against Algorithm 2 in `method.tex`: verify warmup MSE→NLL transition, softplus+floor, JVP not flowing through `log_var`, variance LR multiplier if needed. |
| Apr 7 | R2 completes by ~mid-day (step 150k), then launch **R3** (`improved_meanflow_unet_cifar_10`, iMF reproduction). | Launch **R1** (`vamf_nll_unet_cifar_10`). | Three small PRs on `juanwu/meanflow`: (a) NLL audit fixes, (b) step-budget cut `800_000 → 150_000` in both `vamf_*_unet_cifar_10` configs, (c) title fix `Variances→Variance` + `\todo` removal in `abstract.tex`. Land before either launch. |
| Apr 8 | R3 continuing (~20k). | R1 continuing (~20k); once healthy, it is the go/no-go anchor for Apr 19. | Implement ablation flags `no_fm_anchor` and `boundary_tangent` in `VAMeanFlowUNetModel` (two highest-value, rest can wait). |
| Apr 9 | R3 continuing (~50k). | R1 continuing (~50k). | Land instrumentation for E5: per-step `cosine(∇L_MF, ∇L_FM)`, per-t gradient variance estimator, EMA-JVP curvature proxy. Behind a flag; smoke-test on CPU. |
| Apr 10 | R3 finishing (~100k). | R1 continuing (~80k). | Start writing `experiment.tex` section headers and prose skeleton (no numbers yet). |
| Apr 11 (Sat) | R3 done by mid-day; launch **R4** (vanilla MF baseline — E2 loss curve + E5 MF gradient-variance side). | R1 finishing (~110k). | Implement and run the **Gaussian-mixture closed-form toy** for E4. CPU-only, few hours of dev work. Commit plotting script under `assets/`. |
| Apr 12 (Sun) | R4 continuing (~30k). | R1 completes (~150k). Collect numbers, snapshot checkpoint. | Collect week 1 status. |

**Week 1 exit criteria:**
- R1 (VaMF-NLL) **complete at 150k** with real FID/IS numbers
- R2 (VaMF-MSE) **complete at 150k** with real FID/IS numbers
- R3 (iMF) **complete at 150k** with real FID/IS numbers
- R4 (vanilla MF) launched and on track
- E4 Gaussian-mixture plot generated
- Two ablation flags merged
- `experiment.tex` skeleton has real headers (no lipsum)

This is strictly stronger than the single-VM Week 1 plan, which only
required R1 to be *converging at ≥80k*. With two VMs, Week 1 should
produce three complete main-results numbers (R1, R2, R3) by Sun
Apr 12, not Wed Apr 15.

### Week 2 — 2026-04-13 (Mon) → 04-19 (Sun)

Ablations, instrumentation results, first full draft of `experiment.tex`.
With two VMs, ablations run in pairs. Each 100k ablation run is ~18.5 h,
so a pair fits per day-and-a-half per VM; two VMs × 6.5 days ≈ 8 ablation
slots, which comfortably covers R5a–R5e + R6.

| Date | VM-A (`tpu-v4-32-us-central2-b`) | VM-B (`behavior`) | Non-TPU work |
|---|---|---|---|
| Apr 13 | R4 completing. Extract 1-/2-step FID/IS for R1–R4. Launch **R5a** (VaMF-NLL, `fm_anchor_weight=0` — isolates FM anchor). | Launch **R5b** (VaMF-NLL with boundary-condition tangent — isolates EMA anchor). | Start writing "Ablation Study" subsection with placeholder numbers; fill in R1–R4 numbers in Table 1. |
| Apr 14 | R5a continuing. | R5b continuing. | |
| Apr 15 | R5a finishes mid-day; launch **R5c** (VaMF-NLL with stochastic `(e−x₀)` tangent — the ablation that should look worst, = vanilla MF + variance head). | R5b finishes mid-day; launch **R5d** (VaMF-MSE with `fm_anchor_weight=0`, symmetric with R5a). | Draft Q1 (EMA vs separate-head) and Q4 (when variance head > α-schedule) paragraphs in §5, grounded in R1+R3. |
| Apr 16 | R5c continuing. | R5d continuing. | Draft Q2 (FM anchor δ) and Q3 (`‖Dv/Dt‖` magnitude) paragraphs. |
| Apr 17 | R5c finishes; launch **R5e** (VaMF-MSE with fixed uniform `w(t)` — isolates the variance head). Skip if equivalent to an existing run. | R5d finishes; launch **R6** (instrumented VaMF-NLL with E5 logging). If E5 instrumentation was already live in R1, re-use R1 logs and leave VM-B free for a re-run of any R5 that misbehaved. | Draft Q5 (multi-step sampling) paragraph. |
| Apr 18 (Sat) | R5e finishes. | R6 finishes. | Collect all E3 ablation numbers. Draft Table 1 (main results) and Table 2 (ablations) with real numbers. |
| Apr 19 (Sun) | Idle or buffer for re-runs. | Idle or buffer for re-runs. | **Week 2 exit + Apr 19 go/no-go.** Draft E4 + E5 figures. Finalize Q1–Q5 paragraphs. |

**Week 2 exit criteria (= Apr 19 go/no-go):**
- E1 main results table with real numbers for at least VaMF-MSE, VaMF-NLL, iMF, vanilla MF
- E3 ablation table with real numbers for the 5 R5 runs
- E4 and E5 figures with real data
- VaMF-NLL is competitive with iMF under matched compute

**Go/no-go decision rule:** If VaMF-NLL at 150k steps is worse than iMF
at 150k steps on the same backbone, pivot to **MSE-variant-as-main** on
Mon Apr 20. The structural review supports this fallback and
`9k3bt7aa` (R2) is the safety net. The pivot costs ~2–3 days of
rewriting §5 and the intro's third-paragraph headline claim, plus
swapping Algorithm 2 and Algorithm 1 between main text and appendix.

### Week 3 — 2026-04-20 (Mon) → 04-26 (Sun)

Full writing pass, ImageNet-64 stretch, structural review verification.

| Date | Action |
|---|---|
| Apr 20 | Commit headline numbers. Write `experiment.tex` body: Setup, Main Results, Ablations, Theory Validation. Write `conclusion.tex`. |
| Apr 21 | Hit every Q1–Q5 from the 2026-04-04 review in §5 or §6 prose. Apply typo and minor fixes from the 2026-04-04 review (see §6 of this plan). |
| Apr 22 | If schedule permits: launch **R7** (VaMF-NLL on ImageNet-64) for E6. At ~4× compute per step, 100k steps ≈ 75h (~3.1 days). Only launch if it can finish by Apr 29 (cf. Week 4 freeze). |
| Apr 23 | Polish all figures (colorblind-safe palette, self-contained captions, regeneration scripts under `assets/`). Polish all tables (stddev, bold best, footnotes for compute match). |
| Apr 24 | Second-pass read of `method.tex` in light of real numbers — tighten claims the experiments do not fully support. |
| Apr 25 | Full re-compile. Check page count against 8-page NeurIPS limit. Trim if needed (move Q1/Q4 paragraphs to appendix first, then condense proofs). |
| Apr 26 | v1 draft complete. Structural-review-2026-04-02 three-item verification: (a) GVD at the opening of §4.3, not a standalone subsection; (b) each theorem has a downstream method consumer; (c) NLL variant primary in main text, MSE in appendix. |

### Week 4 — 2026-04-27 (Mon) → 05-06 (Wed)

Buffer, polish, abstract freeze, submit.

| Date | Action |
|---|---|
| Apr 27 (Mon) | Self-review pass against 2026-04-04 review W1–W5 and Q1–Q5. Fill gaps. |
| Apr 28 (Tue) | Re-derive inequalities: Prop 2 bias decomposition, Prop 6 signs, Prop 4 compound error bounds. |
| Apr 29 (Wed) | **Compute buffer day.** Any run that died. Last day anything new can *start* and still converge for the abstract. |
| Apr 30 (Thu) | Appendix completion: full proofs, Algorithm 1 MSE variant, reproducibility statement (reviewer W5), "code will be released" note. |
| May 1 (Fri) | Figure audit + regeneration scripts committed. Reference audit (all 57+ entries, consistent format, arXiv IDs). |
| May 2 (Sat) | **Experiments freeze.** Final numbers locked into tables. Anything still running past this point is PDF-only, cannot affect the abstract. |
| May 3 (Sun) | **Abstract lock-down.** Write final abstract with real headline FID replacing `\todo`. Lock title. Compile and read abstract out loud. Self-review author field. |
| **May 4 (Mon)** | **Abstract deadline.** Register on OpenReview: title, authors, abstract. **Do this in the morning.** Rest of day: late-breaking PDF polish. |
| May 5 (Tue) | Final read-through. Diff against v1 draft for regressions. **Upload PDF in the morning.** |
| **May 6 (Wed)** | **Full paper deadline.** Reserve for emergency re-upload only. |

---

## 5. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | NLL variant has a scaling bug similar to the MSE one just fixed | **high** | **high** | Audit before Apr 7 launch. Compare first 20k steps against `9k3bt7aa` at matched step. Check warmup transition is not a discontinuity. |
| 2 | VaMF-NLL FID plateaus above iMF | medium | high | Apr 19 go/no-go → pivot to MSE variant as main text; `9k3bt7aa` is the safety net. |
| 3 | E5 gradient-variance instrumentation breaks the compiled graph or slows training | medium | medium | Implement behind a flag. Smoke-test on a 2k-step CPU run before TPU launch. |
| 4 | Ablation flags expose a latent bug in `_loss_fn` when a branch is removed | medium | medium | Ship with a unit test per flag verifying loss stays finite on a fixed batch. |
| 5 | Page budget blows past 8 pages when experiments are filled in | medium | low | Structural review budgets §5 for 2 pages. Move Q1/Q4 answers to appendix first, condense proofs second. |
| 6 | ImageNet-64 (E6) monopolizes the last week | low | medium | Only launch on Apr 22 and only if it can finish by Apr 29. Otherwise skip or frame as "preliminary." |
| 7 | CIFAR-10 FID numbers are not competitive (>30) | medium | **high** | Pivot framing: variance-reduction-benefits paper with ablation + theory-validation figures carrying the narrative, not raw FID. Use E4/E5 figures as primary evidence. |
| 8 | iMF reproduction misses published numbers under our UNet | medium | medium | Report both: "iMF (published)" and "iMF (ours, matched compute)". Compare VaMF against the matched reproduction for the claim, reference published number for context. |
| 9 | AlphaFlow / Re-MeanFlow re-implementation under our UNet infeasible in time | medium | low | Report published numbers, clearly labeled. Only iMF and vanilla MF are mandatory for identical-backbone comparison. |
| 10 | OpenReview title / abstract locked at May 4 registration (no post-edit window) | unknown | medium | Verify venue policy on 04-07. Treat as locked by default. Means title fix must land by May 3 end of day. |
| 11 | Author-block anonymization not set up in `main.tex` / `pdt_report.sty` | unknown | medium | Audit on 04-07 alongside the NLL code path. Cheap check, blocks nothing now. |

---

## 6. Fixes from the 2026-04-04 reviewer to land during writing

**Critical (must fix before submission):**
- W1: Experiments section — entire reason for this plan
- Title: `On Variances Reduction` → `On Variance Reduction in Training Mean Flows`
- Abstract: remove `\todo[inline]{Add experiment results later...}`, replace with 1-sentence numerical summary

**Important:**
- W2: discuss C² smoothness assumption for non-smooth activations (ReLU/LeakyReLU) — short paragraph in appendix
- W2: discuss EMA decay rate μ tradeoff — short paragraph in §5 or appendix, ideally supported by a small μ sweep if compute permits
- W3: variance head spatial pooling rationale — explicit one-paragraph discussion in method section or appendix
- W3: `σ²_min` guidance — appendix sentence
- W5: reproducibility statement + "code will be released" — standard paragraph at end of appendix

**Minor typography / notation:**
- "continouously" → "continuously" (Theorem 3)
- "dervative" → "derivative" (Theorem 3)
- "herein" overuse → replace with "here" / "in this expression"
- `V^EMA_θ` defined in Proposition 1 before formal definition — forward reference note or reorder
- "Expectation Divergence" → "Expectation Gap" (per structural review, clearer naming)
- Consider placing Related Work before Method (structural review item)

---

## 7. OpenReview mechanics to verify before May 4

1. **Title / abstract editability after registration.** Some venues
   lock, some allow edits until the paper deadline. Treat as locked by
   default.
2. **Submission form fields.** TL;DR? keywords? conflict domains?
   compute statement? Stage answers before May 4, not on the day of.
3. **Reproducibility statement location.** Dedicated form field or in
   paper body?
4. **Anonymization.** NeurIPS is typically double-blind for initial
   submission. Verify `main.tex` + `pdt_report.sty` have a toggle and
   that the default compile is anonymized. Current `main.tex` hard-codes
   author as "Juanwu Lu, Purdue" — needs a conditional.

---

## 8. Immediate action items (unblocked by user greenlight)

1. **Provision the `behavior` VM for VM-B duty.** Worker 0 currently
   has the repo at commit `1110a3f` (pre-MSE-scaling-fix). Before
   launching R1 on it, run on local machine:
   ```bash
   gcloud compute tpus tpu-vm ssh juanwu@behavior --project research-481912 \
       --zone us-central2-b --worker 0 --command \
       "cd pdt-research && git fetch -a && git pull --rebase && git log -1 --oneline"
   gcloud compute tpus tpu-vm ssh juanwu@behavior --project research-481912 \
       --zone us-central2-b --worker 0 --ssh-flag="-A" --command \
       "./sync_folder.sh /home/juanwu/pdt-research"
   gcloud compute tpus tpu-vm ssh juanwu@behavior --project research-481912 \
       --zone us-central2-b --worker all --command \
       "cd pdt-research && git log -1 --oneline"
   ```
   Then smoke-test with a short bazel run to confirm TPU visibility.
2. **NLL code path audit** + three small PRs on `juanwu/meanflow`:
   (a) audit fixes, (b) step-budget cut 800k→150k, (c) title +
   abstract `\todo` removal. Target: Apr 7, before R1 launches on VM-B.
3. **Let `9k3bt7aa` run to 150k** (≈ 14 h more) on VM-A. It is the
   VaMF-MSE data point. Do not preempt — with VM-B now available, R1
   no longer needs VM-A as a launch slot.
4. **Implement `no_fm_anchor` and `boundary_tangent` ablation flags**
   this week. The other three R5 ablations can wait for Week 2.
5. **Audit anonymization toggle** in `main.tex` / `pdt_report.sty`.
   Cheap check during Apr 7 PR work.
6. **Monitoring:** manual polling via
   `.claude/scripts/check_wandb.py <run_id>` unless a scheduled remote
   trigger is explicitly requested. With two VMs running concurrently
   from Apr 7 onward, check both runs at each poll.

---

## 9. Appendix: reference data

### 9.1 Run `9k3bt7aa` health at authoring time

```
step      0  fid=453.27
step   2500  fid=448.59
step   5000  fid=437.87
step   7500  fid=416.54
step  10000  fid=378.40
step  12500  fid=346.20
step  15000  fid=352.80  <- early bounce
step  17500  fid=371.74
step  20000  fid=379.38
step  22500  fid=375.71
step  25000  fid=365.43
step  27500  fid=348.24
step  30000  fid=326.14
step  32500  fid=302.19  <- sustained descent begins
step  35000  fid=276.79
step  37500  fid=252.04
step  40000  fid=228.78
step  42500  fid=206.96
step  45000  fid=187.61
```

Descent rate over last 5 evals: −22.92 per 2,500 steps. Extrapolation
(rates will slow): step 55k ≈ FID 140, step 65k ≈ FID 95,
step 75k ≈ FID 50. Plateau likely somewhere in 20–60 range; worth
running to 150k to see where it lands.

### 9.2 Config of `9k3bt7aa`

```
model: VAMeanFlowUNetModel
  features=128, dropout_rate=0.2, image_size=32
  timestamp_sampler=logit-normal (mean=-0.4, stddev=1.0)
  timestamp_overlap_rate=0.5
  adaptive_weight_power=0.0  (SNR-only weighting per fix)
  snr_epsilon=0.01
  fm_anchor_weight=0.5
  fm_anchor_delta_min=1e-4, fm_anchor_delta_max=0.01
  predict_variance=False  (MSE variant)

optimizer:
  adam b1=0.9 b2=0.999
  lr: warmup_constant 1e-8 → 1e-4 over 10_000 steps
  ema_rate=0.99995
  grad_clip_value=1, grad_clip_method=None  (effectively disabled)

trainer:
  num_train_steps=800_000   <- must be cut to 150_000
  log_every_n_steps=50
  eval_every_n_steps=2_500
  checkpoint_every_n_steps=10_000
```

### 9.3 Existing documents

| Path | Purpose |
|---|---|
| `docs/generative/vamf/main.tex` | Root document |
| `docs/generative/vamf/contents/*.tex` | Eight section files (see §1.1 above) |
| `docs/generative/vamf/reference.bib` | 57 BibTeX entries |
| `docs/generative/vamf/commands.tex` | Math macros |
| `docs/generative/vamf/pdt_report.sty` | Style file |
| `docs/generative/vamf/reviews/structural-review-2026-04-02.md` | Structural review (most items already addressed) |
| `docs/generative/vamf/reviews/2026-04-04.md` | Mock NeurIPS review; action list for §5 |
| `docs/generative/vamf/plan/implementation-plan-2026-04-03.md` | Code implementation plan (Phases 1–3 done) |
| `docs/generative/vamf/plan/submission-plan-2026-04-06.md` | **This file** |
| `docs/generative/vamf/release/target/main.pdf` | Last compiled PDF (mtime 2026-04-04) |
| `docs/generative/vamf/assets/illustration.pdf` | Figure asset |
| `docs/generative/vamf/assets/generate_illustration.py` | Figure generator |
| `.claude/scripts/check_wandb.py` | Compact wandb run health check script |
