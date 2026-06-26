#!/usr/bin/env python3
"""Per-step FID ordering analysis for VAMF (rebuttal evidence for experiment.tex).

The paper claims the bias-variance FID ordering holds at "every matched-step
checkpoint" (FID monotonically increasing in beta: beta=0 best). The wandb export
shows it holds strictly at only ~15/59 steps, with beta=0.25 BEATING beta=0 across
a sustained mid-training window. The human previously saw this and suspected
training instability. This script adjudicates signal-vs-noise quantitatively, and
produces the per-step table needed to reword the claim.

INPUT: a normalized CSV the caller builds from the wandb export, columns:
    beta,step,fid          (optionally extra metric columns are ignored)
one row per (beta, step). If multiple worker rows exist per (beta, step) they are
averaged. Use the CANONICAL run per beta (see the audit's authoritative mapping).

Usage:
    python fid_ordering.py --csv fid_long.csv [--metric fid] [--plot out.png]
"""

import argparse
import csv
import math
import statistics as st
from collections import defaultdict

TARGET_BETAS = [0.0, 0.25, 0.5, 1.0]   # the paper's 4-point ladder


def rolling_median(xs, w=5):
    n = len(xs)
    out = []
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(st.median(xs[lo:hi]))
    return out


def local_noise(fids_by_step):
    """Jitter estimate independent of the smooth downward trend: std of the
    residual after subtracting a rolling median. (Raw first differences would
    conflate the genuine training trend with noise.)"""
    steps = sorted(fids_by_step)
    vals = [fids_by_step[s] for s in steps]
    if len(vals) < 3:
        return float("nan")
    resid = [v - m for v, m in zip(vals, rolling_median(vals))]
    return st.pstdev(resid) if len(resid) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="normalized CSV: beta,step,fid")
    ap.add_argument("--metric", default="fid", help="metric column name (default fid)")
    ap.add_argument("--plot", default=None, help="optional PNG path for the curves")
    a = ap.parse_args()

    # ---- load: per beta -> {step: [values]} then average dups ----
    raw = defaultdict(lambda: defaultdict(list))
    with open(a.csv, newline="") as f:
        r = csv.DictReader(f)
        if "beta" not in r.fieldnames or "step" not in r.fieldnames or a.metric not in r.fieldnames:
            raise SystemExit(f"CSV must have columns: beta, step, {a.metric}. Got {r.fieldnames}")
        for row in r:
            try:
                b = float(row["beta"]); s = int(float(row["step"])); v = float(row[a.metric])
            except (ValueError, TypeError):
                continue
            if math.isnan(v):
                continue
            raw[b][s].append(v)
    fid = {b: {s: sum(vs) / len(vs) for s, vs in d.items()} for b, d in raw.items()}

    missing = [b for b in TARGET_BETAS if b not in fid]
    if missing:
        raise SystemExit(f"CSV missing betas {missing}; present: {sorted(fid)}")

    # ---- steps where ALL four target betas have a value ----
    common = sorted(set.intersection(*[set(fid[b]) for b in TARGET_BETAS]))
    if not common:
        raise SystemExit("no steps where all four betas are present")

    # ---- per-step ordering ----
    print(f"\n=== Per-step 4-point FID ordering ({a.metric}; {len(common)} matched steps) ===")
    print(f"predicted (paper) ordering: beta 0 < 0.25 < 0.5 < 1 (FID increasing in beta)\n")
    hdr = f"{'step':>8}" + "".join(f"{('b='+str(b)):>10}" for b in TARGET_BETAS) + f"   {'ordering':<20} strict?"
    print(hdr); print("-" * len(hdr))
    n_strict = 0
    rows = []
    for s in common:
        vals = [fid[b][s] for b in TARGET_BETAS]
        strict = all(vals[i] < vals[i + 1] for i in range(3))
        n_strict += strict
        order = "<".join(str(b) for b, _ in sorted(zip(TARGET_BETAS, vals), key=lambda t: t[1]))
        rows.append((s, vals, strict, order))
        print(f"{s:>8}" + "".join(f"{v:>10.3f}" for v in vals) + f"   {order:<20} {'yes' if strict else 'NO'}")
    print(f"\nstrict ordering holds at {n_strict}/{len(common)} matched steps "
          f"({100*n_strict/len(common):.0f}%).")

    # ---- the beta=0.25 vs beta=0 crossover (the specific violation) ----
    gap = {s: fid[0.0][s] - fid[0.25][s] for s in common}  # >0 => 0.25 BETTER than 0
    # longest contiguous run of gap>0
    best = (0, None, None); cur = 0; start = None
    for i, s in enumerate(common):
        if gap[s] > 0:
            if cur == 0:
                start = s
            cur += 1
            if cur > best[0]:
                best = (cur, start, s)
        else:
            cur = 0
    run_len, w0, w1 = best
    sigma = st.median([local_noise(fid[b]) for b in TARGET_BETAS])

    print("\n=== beta=0.25 vs beta=0 crossover (signal vs instability) ===")
    if run_len == 0:
        print("no step where beta=0.25 beats beta=0; nothing to adjudicate.")
    else:
        win_steps = [s for s in common if w0 <= s <= w1 and gap[s] > 0]
        win_gaps = [gap[s] for s in win_steps]
        mean_gap = sum(win_gaps) / len(win_gaps)
        max_gap = max(win_gaps)
        ratio = mean_gap / sigma if sigma and not math.isnan(sigma) else float("nan")
        print(f"longest window where 0.25<0 : steps {w0}..{w1}  ({run_len} consecutive matched steps)")
        print(f"mean advantage in window     : {mean_gap:.3f} FID   (max {max_gap:.3f})")
        print(f"local noise scale (sigma)    : {sigma:.3f} FID   -> advantage/noise = {ratio:.2f}x")
        sustained = run_len >= max(5, int(0.1 * len(common)))
        big = (not math.isnan(ratio)) and ratio > 1.0
        print()
        if sustained and big:
            print("--> SYSTEMATIC (signal, not instability): the advantage is sustained over")
            print(f"    {run_len} consecutive checkpoints and exceeds the local noise scale.")
            print("    Consistent with the bias-variance account: a moderate beta wins in early/mid")
            print("    training via variance reduction, then bias dominates and beta=0 retakes the")
            print("    lead at convergence. The 'training instability' read is NOT supported.")
            print("    => reword experiment.tex: not 'every matched-step', but 'stable at convergence")
            print("       with a mid-training beta-ordered crossover consistent with bias-variance'.")
        elif not sustained:
            print("--> LIKELY NOISE: the 0.25<0 region is too short to distinguish from jitter.")
            print("    The 'instability' read is plausible; do not build a claim on the crossover.")
        else:
            print("--> AMBIGUOUS: sustained but within the noise scale. Report magnitudes; do not")
            print("    over-claim either way.")

    # ---- optional plot ----
    if a.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            for b in TARGET_BETAS:
                ss = sorted(fid[b]); plt.plot(ss, [fid[b][s] for s in ss], marker=".", label=f"beta={b}")
            if run_len:
                plt.axvspan(w0, w1, alpha=0.12, color="red", label="0.25<0 window")
            plt.xlabel("step"); plt.ylabel(a.metric.upper()); plt.legend()
            plt.title("FID vs step by beta"); plt.gca().invert_yaxis()
            plt.tight_layout(); plt.savefig(a.plot, dpi=120); print(f"\nwrote {a.plot}")
        except Exception as ex:  # noqa: BLE001
            print(f"(plot skipped: {ex})")


if __name__ == "__main__":
    main()
