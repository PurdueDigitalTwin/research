#!/usr/bin/env python3
"""Re-analyze VAMF B1 Phase-1 JSONs to settle the gradient-VARIANCE question.

The B1 handoff concluded "annealing does NOT reduce gradient variance" but it
read only ``nr = tr_cov / mean_norm_sq`` (a noise-to-SIGNAL ratio). VAMF theory
predicts beta>0 reduces the gradient VARIANCE itself, i.e. the NUMERATOR
``tr_cov``. Since beta>0 also changes the loss surface, it can shrink the
denominator ``mean_norm_sq`` too, so ``nr`` can rise even while ``tr_cov`` falls.

This script decomposes the three quantities, focuses on the CONSTANT-beta runs
(the only clean test of "does a fixed beta>0 reduce tr_cov at matched steps"),
and emits an explicit, conditional verdict — no one-line over-claim.

Usage:
    python analyze_grad_var.py --dir /path/to/work_dir [--dataset dgmm_64]

Reads files saved by run_toy.py:
    {dataset}_{method}_{shape}_s{s1}_b{tangent_beta}_{seed}.json
each containing args / history / grad_var_history / final.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
def classify(args):
    """Map a run's args to (label, regime, beta_const) where beta_const is the
    fixed beta for constant runs or None for annealed runs."""
    method = args.get("method", "")
    shape = args.get("beta_anneal_shape", "constant")
    tb = float(args.get("tangent_beta", 1.0))
    if method == "meanflow":
        return ("baseline_b0", "constant", 0.0)
    if method == "vamf_tmix" and shape == "constant":
        return (f"static_b{tb:g}", "constant", tb)
    if method == "vamf_tmix":
        s1 = args.get("beta_anneal_s1")
        return (f"{shape}_s{s1:g}", "annealed", None)
    return (f"{method}_{shape}", "other", None)


def mean_sem(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def bin_of(step, total):
    f = step / float(total)
    if f < 0.2:
        return "early"
    if f < 0.6:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="work_dir with the *.json runs")
    ap.add_argument("--dataset", default="dgmm_64")
    ap.add_argument("--plot", action="store_true", help="also write PNGs (needs matplotlib)")
    args_cli = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args_cli.dir, f"{args_cli.dataset}_*.json")))
    if not paths:
        raise SystemExit(f"no {args_cli.dataset}_*.json found under {args_cli.dir}")

    # runs[label] = {"regime":..., "beta":..., "seeds": {seed: {"total":T, "gv":[...]}}}
    runs = defaultdict(lambda: {"regime": None, "beta": None, "seeds": {}})
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        a = d.get("args", {})
        gv = d.get("grad_var_history", []) or []
        if not gv:
            print(f"WARN: empty grad_var_history in {os.path.basename(p)} — skipped")
            continue
        label, regime, beta = classify(a)
        seed = a.get("seed")
        total = int(a.get("steps", 200_000))
        runs[label]["regime"] = regime
        runs[label]["beta"] = beta
        runs[label]["seeds"][seed] = {"total": total, "gv": gv}

    # ---- per-config, per-bin aggregation over seeds ----
    METRICS = ("tr_cov", "mean_norm_sq", "nr")
    BINS = ("early", "mid", "late")
    agg = {}  # label -> bin -> metric -> (mean, sem) across seeds
    for label, info in runs.items():
        per_seed_binmeans = defaultdict(lambda: defaultdict(list))  # bin->metric->[per-seed mean]
        for seed, sd in info["seeds"].items():
            tmp = defaultdict(lambda: defaultdict(list))  # bin->metric->[vals in seed]
            for e in sd["gv"]:
                b = bin_of(e["step"], sd["total"])
                for m in METRICS:
                    if e.get(m) is not None:
                        tmp[b][m].append(float(e[m]))
            for b in BINS:
                for m in METRICS:
                    if tmp[b][m]:
                        per_seed_binmeans[b][m].append(sum(tmp[b][m]) / len(tmp[b][m]))
        agg[label] = {b: {m: mean_sem(per_seed_binmeans[b][m]) for m in METRICS} for b in BINS}

    # ---- print the decomposed table ----
    order = sorted(agg, key=lambda L: (runs[L]["regime"] != "constant",
                                       runs[L]["beta"] if runs[L]["beta"] is not None else 9,
                                       L))
    print(f"\n=== Gradient-variance decomposition ({args_cli.dataset}, "
          f"{len(runs)} configs, mean over seeds) ===\n")
    hdr = f"{'config':<16}{'bin':<7}" + "".join(f"{m:>18}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for label in order:
        for b in BINS:
            row = f"{label:<16}{b:<7}"
            for m in METRICS:
                mu, se = agg[label][b][m]
                row += f"{mu:>11.4g}±{se:<6.2g}" if not math.isnan(se) else f"{mu:>11.4g}{'':<7}"
            print(row)
        print()

    # ---- the decisive comparison: constant betas at matched (early) steps ----
    def tr_early(label):
        return agg[label]["early"]["tr_cov"][0] if label in agg else float("nan")

    def mns_early(label):
        return agg[label]["early"]["mean_norm_sq"][0] if label in agg else float("nan")

    b0 = next((L for L in agg if runs[L]["regime"] == "constant" and runs[L]["beta"] == 0.0), None)
    bpos = sorted([L for L in agg if runs[L]["regime"] == "constant" and (runs[L]["beta"] or 0) > 0],
                  key=lambda L: runs[L]["beta"])

    print("=== VARIANCE VERDICT (constant-beta runs, EARLY bin) ===\n")
    if b0 is None or not bpos:
        print("Cannot run the clean test: need meanflow(beta=0) + >=1 static beta>0 run.")
        return

    tr0 = tr_early(b0)
    print(f"reference tr_cov[beta=0, early] = {tr0:.4g}\n")
    ratios = []
    for L in bpos:
        r = tr_early(L) / tr0 if tr0 else float("nan")
        ratios.append((runs[L]["beta"], r))
        sig = mns_early(L) / mns_early(b0) if mns_early(b0) else float("nan")
        print(f"  beta={runs[L]['beta']:<4g}  tr_cov ratio = {r:.3f}   "
              f"(signal mean_norm_sq ratio = {sig:.3f})")

    eps = 0.02
    reduced = all(r < 1 - eps for _, r in ratios)
    monotone = all(ratios[i][1] >= ratios[i + 1][1] - 1e-6 for i in range(len(ratios) - 1))
    none_reduced = all(r >= 1 - eps for _, r in ratios)

    print()
    if reduced and (monotone or len(ratios) == 1):
        print("--> tr_cov IS REDUCED by beta>0 in early training"
              + (" (monotone in beta)." if len(ratios) > 1 else "."))
        print("    The variance-reduction MECHANISM is reproduced on DGMM-64.")
        print("    => The B1 KILL is a BIAS/QUALITY story, not a variance failure:")
        print("       beta>0 lowers gradient variance but injects bias that hurts SW1 ")
        print("       monotonically with cumulative beta-exposure. This is the paper's")
        print("       FID-MSE mismatch reproduced on a controlled toy -> A3 ammunition.")
        print("    The earlier 'no nr improvement' was an ARTIFACT: check the signal")
        print("    ratios above — if mean_norm_sq also shrank, nr can rise while tr_cov falls.")
    elif none_reduced:
        print("--> tr_cov is NOT reduced by beta>0 (even the raw numerator).")
        print("    Do NOT use this toy's variance numbers in the rebuttal. Two candidates:")
        print("    (a) the EMA proxy (3-layer MLP on hard data) is a POOR control variate")
        print("        on this toy -> inflated variance, a toy artifact that need not")
        print("        transfer to DiT scale; or")
        print("    (b) a genuine inconsistency with the paper's DGMM '1.2-4.3x variance")
        print("        reduction' claim -> reconcile against the exact quantity/config")
        print("        that figure was generated from (see the cross-check task).")
    else:
        print("--> MIXED: some beta>0 reduce tr_cov, some do not, or non-monotone.")
        print("    Report the per-beta ratios; lean cautious, do not over-claim either way.")

    print("\nCAVEAT (carry into any writeup): a toy KILL justifies NOT running Phase-2 DiT")
    print("annealing, but is NOT evidence the variance mechanism fails at scale. Keep the")
    print("two claims separate.")

    # ---- optional plots ----
    if args_cli.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            for m in ("tr_cov", "nr"):
                plt.figure(figsize=(7, 4))
                for L in [b0] + bpos:
                    # pool seeds: plot per-step mean
                    bystep = defaultdict(list)
                    for sd in runs[L]["seeds"].values():
                        for e in sd["gv"]:
                            if e.get(m) is not None:
                                bystep[e["step"]].append(float(e[m]))
                    steps = sorted(bystep)
                    vals = [np.mean(bystep[s]) for s in steps]
                    plt.plot(steps, vals, label=L)
                plt.xlabel("step"); plt.ylabel(m); plt.legend(); plt.title(f"{m} vs step")
                out = os.path.join(args_cli.dir, f"b1_{m}_vs_step.png")
                plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
                print(f"wrote {out}")
        except Exception as ex:  # noqa: BLE001
            print(f"(plotting skipped: {ex})")


if __name__ == "__main__":
    main()
