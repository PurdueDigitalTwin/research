#!/usr/bin/env python3
"""beta=0 SEED-robustness check for VAMF.

Two beta=0 DiT runs exist with (per the forensic pass) identical protocol,
differing only by training seed:
  - hde9iaqj  : the paper's baseline (matched-pair with the beta=1 run)  -> "b0_paper"
  - fidpdet7  : an independent beta=0 replication                        -> "b0_alt"

With b0_paper the 4-point ordering holds 54/54; with b0_alt it holds 14/54 and
beta=0.25 wins mid-training. So the ordering may be a property of the chosen
beta=0 seed, not of beta. This script quantifies that: at each matched step it
compares the beta=0 SEED SPREAD |b0_paper - b0_alt| against the beta=0 ->
beta=0.25 EFFECT, and reports the SEED-ROBUST ordering count (steps where BOTH
beta=0 seeds beat beta=0.25 -- the honest version of "54/54").

This adjudicates how to word the claim. It does NOT touch the paper.

INPUT CSV columns (one row per step):  step,b0_paper,b0_alt,b025
(optionally b05,b1 -- ignored here; use fid_ordering.py for the full ladder.)

Usage:  python seed_variance.py --csv seed_compare.csv
"""

import argparse
import csv
import math
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV: step,b0_paper,b0_alt,b025")
    a = ap.parse_args()

    need = ["step", "b0_paper", "b0_alt", "b025"]
    rows = []
    with open(a.csv, newline="") as f:
        r = csv.DictReader(f)
        miss = [c for c in need if c not in r.fieldnames]
        if miss:
            raise SystemExit(f"CSV missing columns {miss}; has {r.fieldnames}")
        for d in r:
            try:
                rows.append((int(float(d["step"])),
                             float(d["b0_paper"]), float(d["b0_alt"]), float(d["b025"])))
            except (ValueError, TypeError):
                continue
    rows.sort()
    if not rows:
        raise SystemExit("no usable rows")

    print(f"\n=== beta=0 seed robustness ({len(rows)} matched steps) ===")
    print("convention: 'effect' = b025 - b0  (>0 => beta=0 better, the paper's direction)\n")
    hdr = (f"{'step':>8}{'b0_paper':>10}{'b0_alt':>9}{'b025':>9}"
           f"{'seedspread':>12}{'eff_paper':>11}{'eff_alt':>9}  both>0.25?")
    print(hdr); print("-" * len(hdr))

    seed_spreads, eff_paper_abs = [], []
    n_robust = 0           # both beta=0 seeds beat beta=0.25
    n_paper_only = 0       # paper seed beats 0.25 but alt does not (seed-fragile win)
    n_eff_within_band = 0  # |paper effect| < seed spread
    for s, bp, ba, b25 in rows:
        spread = abs(bp - ba)
        eff_p = b25 - bp
        eff_a = b25 - ba
        both = (eff_p > 0) and (eff_a > 0)
        n_robust += both
        if eff_p > 0 and eff_a <= 0:
            n_paper_only += 1
        if abs(eff_p) < spread:
            n_eff_within_band += 1
        seed_spreads.append(spread); eff_paper_abs.append(abs(eff_p))
        print(f"{s:>8}{bp:>10.3f}{ba:>9.3f}{b25:>9.3f}"
              f"{spread:>12.3f}{eff_p:>11.3f}{eff_a:>9.3f}  {'yes' if both else 'no'}")

    n = len(rows)
    med_spread = st.median(seed_spreads)
    med_eff = st.median(eff_paper_abs)
    print(f"\nmedian beta=0 seed spread     : {med_spread:.3f} FID")
    print(f"median |beta=0 -> 0.25 effect|: {med_eff:.3f} FID")
    ratio = med_spread / med_eff if med_eff > 0 else float("inf")
    print(f"seed spread / effect (median) : {ratio:.2f}x")
    print(f"steps where |effect| < seed spread : {n_eff_within_band}/{n}")
    print(f"SEED-ROBUST ordering (both seeds beat 0.25): {n_robust}/{n}")
    print(f"paper-seed-only wins (fragile)             : {n_paper_only}/{n}")

    print()
    if med_spread >= med_eff and n_robust < n:
        print("--> The beta=0/0.25 effect is WITHIN beta=0 seed variance, and the strict")
        print(f"    ordering is seed-robust at only {n_robust}/{n} steps (vs 54/54 for the paper")
        print("    seed alone). Word the claim around seeds, not 'every matched-step':")
        print("    e.g. 'the convergence ordering replicates across seeds; the beta=0/0.25 margin")
        print("    is within run-to-run variance (~{:.1f} FID)'. Consider error bars / >=2 seeds.".format(med_eff))
    else:
        print("--> The effect exceeds beta=0 seed variance and the ordering is largely seed-robust;")
        print("    the stronger ordering claim is defensible. Still report the seed spread for honesty.")


if __name__ == "__main__":
    main()
