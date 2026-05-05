"""Plot \\beta-sweep figure: empirical metric across datasets.

Two metric modes are supported:

  - ``sw1`` (default): final-step sliced Wasserstein-1, the bias-variance
    landscape proxy that Theorem 4 minimizes (in MSE).
  - ``nr``: per-step gradient noise ratio Tr(Cov[g])/||E[g]||², the
    direct gradient-variance metric that Theorem 1 governs.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_beta_sweep -- \\
        --sweep_dir=logs/vamf/beta_sweep_200k \\
        --metric=sw1 \\
        --output=docs/generative/vamf/results/sw1_vs_beta.pdf
"""

import collections
import json
import os
import typing

from absl import app
from absl import flags
import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np

from src.projects.generative.vamf.figures import _style
from src.utilities import logging as _logging

flags.DEFINE_enum(
    name="style",
    default=_style.DEFAULT_STYLE,
    enum_values=list(_style.STYLES),
    help="Render target ('paper' | 'slides').",
)
flags.DEFINE_string(
    name="sweep_dir",
    default=None,
    required=True,
    help=(
        "Directory containing beta_<β>/<dataset>_vamf_tmix_<seed>.json "
        "outputs from run_beta_sweep.sh."
    ),
)
flags.DEFINE_enum(
    name="metric",
    default="sw1",
    enum_values=("sw1", "sw2", "nr"),
    help="Which empirical metric to plot vs β.",
)
flags.DEFINE_list(
    name="datasets",
    default=[
        "checkerboard",
        "eight_gaussians",
        "two_moons",
        "swiss_roll",
        "two_spirals",
        "pinwheel",
    ],
    help="Datasets to include (one panel each).",
)
flags.DEFINE_bool(
    name="overlay_theory",
    default=True,
    help="Whether to overlay the closed-form M(β) prediction.",
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)


_METRIC_TO_KEY = {"sw1": "swd1", "sw2": "swd2", "nr": "nr"}


def _scan(sweep_dir: str) -> typing.Dict[str, typing.Dict[float, list]]:
    r"""Walk beta_<β>/* and return {dataset: {β: [normalized_records]}}.

    Each record is a flat dict with keys (when present in the JSON):
      - swd1, swd2, loss   (from `final` block, fallback to last `history` entry)
      - nr, tr_cov         (from last `grad_var_history` entry; empty unless
        NR-probing was enabled in run_toy.py)
      - kappa, b_sq, sigma2d  (Theorem 4 ingredients; require logging
        support in run_toy.py — currently absent, overlay will be
        skipped if missing)
    """
    out: typing.Dict[str, typing.Dict[float, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for entry in sorted(os.listdir(sweep_dir)):
        if not entry.startswith("beta_"):
            continue
        beta = float(entry.removeprefix("beta_"))
        beta_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(beta_dir):
            continue
        for fname in sorted(os.listdir(beta_dir)):
            if not fname.endswith(".json"):
                continue
            stem = fname.removesuffix(".json")
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            ds_method, _ = parts
            ds = ds_method.removesuffix("_vamf_tmix")
            with open(os.path.join(beta_dir, fname)) as f:
                d = json.load(f)
            rec: typing.Dict[str, typing.Any] = {}
            # Prefer `final` (top-level convenience copy), fall back to
            # last `history` entry if absent.
            final = d.get("final") or {}
            history = d.get("history") or []
            for k in ("swd1", "swd2", "loss"):
                if k in final:
                    rec[k] = final[k]
                elif history and k in history[-1]:
                    rec[k] = history[-1][k]
            gvh = d.get("grad_var_history") or []
            if gvh:
                last_g = gvh[-1]
                for k in ("nr", "tr_cov", "mean_norm_sq"):
                    if k in last_g:
                        rec[k] = last_g[k]
            for k in ("kappa", "b_sq", "sigma2d"):
                if k in d:
                    rec[k] = d[k]
            out[ds][beta].append(rec)
    return out


def _aggregate(
    seed_results: typing.List[dict],
    key: str,
) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
    r"""Returns the mean and stderr of `key` across seed results."""
    vals = [r[key] for r in seed_results if key in r and r[key] is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def _predict_M(
    beta: npt.NDArray,
    kappa: float,
    b_sq: float,
    sigma2d: float,
) -> npt.NDArray:
    r"""Closed-form M(β) from Theorem 4."""
    bias = beta**2 * (kappa + 1.0) ** 2 * b_sq
    var = sigma2d * ((1.0 - beta) * kappa - beta) ** 2
    return bias + var


def main(argv: typing.List[str]) -> int:
    del argv
    F = flags.FLAGS
    _style.apply_style(F.style)

    data = _scan(F.sweep_dir)
    n_panels = len(F.datasets)
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.8))
    axes = np.atleast_2d(axes).flatten()

    for i, ds in enumerate(F.datasets):
        ax = axes[i]
        if ds not in data:
            ax.set_visible(False)
            _logging.rank_zero_info("No data for dataset %s, skipping", ds)
            continue
        betas = sorted(data[ds].keys())
        means, errs = [], []
        kappa_avg, b_sq_avg, sigma2d_avg = [], [], []
        metric_key = _METRIC_TO_KEY[F.metric]
        for b in betas:
            seed_results = data[ds][b]
            m, e = _aggregate(seed_results, metric_key)
            means.append(m)
            errs.append(e)
            # Theory ingredients (averaged across seeds at this β)
            for key, store in (
                ("kappa", kappa_avg),
                ("b_sq", b_sq_avg),
                ("sigma2d", sigma2d_avg),
            ):
                vals = [r[key] for r in seed_results if key in r]
                if vals:
                    store.append(float(np.mean(vals)))

        beta_arr = np.asarray(betas)
        mean_arr = np.asarray(means, dtype=np.float64)
        err_arr = np.asarray(errs, dtype=np.float64)

        ax.errorbar(
            beta_arr,
            mean_arr,
            yerr=err_arr,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
            label=f"empirical {F.metric.upper()}",
        )

        # Theory overlay: average (κ, ‖b‖², σ²d) across the sweep,
        # then plot the closed-form M(β) on a fine grid (rescaled
        # to match the empirical y-range — M is in MSE units, the
        # empirical metric is in SW₁ units; we share only the shape).
        if F.overlay_theory and kappa_avg and b_sq_avg and sigma2d_avg:
            kappa = float(np.mean(kappa_avg))
            b_sq = float(np.mean(b_sq_avg))
            sigma2d = float(np.mean(sigma2d_avg))
            beta_grid = np.linspace(0.0, 1.0, 101)
            M_pred = _predict_M(beta_grid, kappa, b_sq, sigma2d)
            # Rescale so min(M_pred) = min(empirical) for shape comparison
            M_norm = (M_pred - M_pred.min()) / (
                M_pred.max() - M_pred.min() + 1e-30
            )
            emp_min, emp_max = mean_arr.min(), mean_arr.max()
            M_scaled = emp_min + M_norm * (emp_max - emp_min)
            ax.plot(
                beta_grid,
                M_scaled,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                color="gray",
                label=r"$M(\beta)$ (Theorem 4, rescaled)",
            )
            # Mark predicted β★
            beta_star = (kappa / (kappa + 1.0)) * (
                sigma2d / (sigma2d + b_sq + 1e-30)
            )
            ax.axvline(
                beta_star,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                color="red",
                label=rf"$\beta^\star\!=\!{beta_star:.2f}$",
            )

        ds_label = ds.replace("_", r"\_")
        ax.set_title(rf"\mathtt{{{ds_label}}}", fontsize=10)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(F.metric.upper())
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linestyle=":", alpha=0.3)

    for j in range(len(F.datasets), rows * cols):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = F.output
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _logging.rank_zero_info("Figure saved to %s", out)
    return 0


if __name__ == "__main__":
    app.run(main)
