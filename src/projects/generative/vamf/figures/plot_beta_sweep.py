"""Plot beta-sweep figure: empirical metric across datasets.

We currently support plotting the following two metrics:

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
import matplotlib.figure as mpl_figure
import matplotlib.lines as mpl_lines
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
        "Directory containing beta_<\beta>/<dataset>_vamf_tmix_<seed>.json "
        "outputs from run_beta_sweep.sh."
    ),
)
flags.DEFINE_enum(
    name="metric",
    default="sw1",
    enum_values=("sw1", "sw2", "nr", "tr_cov"),
    help=(
        "Which empirical metric to plot vs beta. 'tr_cov' averages "
        "Tr(Cov[g]) over the second half of training (controlled by "
        "--tr_cov_tail_start)."
    ),
)
flags.DEFINE_integer(
    name="tr_cov_tail_start",
    default=100000,
    help=(
        "When metric=tr_cov, average grad_var_history entries with "
        "step >= this value (defaults to 100k = second half of a 200k run)."
    ),
)
flags.DEFINE_bool(
    name="annotate_reduction",
    default=False,
    help=(
        "When true (line layout), annotate per-panel reduction ratio "
        "metric(beta=0)/metric(beta=1) in the corner."
    ),
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
    help="Whether to overlay the closed-form M(beta) prediction.",
)
flags.DEFINE_enum(
    name="plot_kind",
    default="line",
    enum_values=("line", "bar"),
    help=(
        "Layout: 'line' = one panel per dataset (beta on x); 'bar' = single "
        "panel with datasets on x and beta as hue (grouped bars)."
    ),
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)


_METRIC_TO_KEY = {"sw1": "swd1", "sw2": "swd2", "nr": "nr", "tr_cov": "tr_cov"}

_METRIC_LABEL = {
    "sw1": r"SW$_{1}$",
    "sw2": r"SW$_{2}$",
    "nr": r"$\mathrm{Tr}(\mathrm{Cov}[g])/\|\mathbb{E}[g]\|^{2}$",
    "tr_cov": r"$\mathrm{Tr}(\mathrm{Cov}[g])$",
}


def _scan(
    sweep_dir: str,
    tr_cov_tail_start: int = 0,
) -> typing.Dict[str, typing.Dict[float, list]]:
    r"""Walk beta_<\beta>/* and return {dataset: {\beta: [normalized_records]}}.

    Each record is a flat dict with keys (when present in the JSON):
      - swd1, swd2, loss   (from `final` block, fallback to last `history` entry)
      - nr, tr_cov         (from `grad_var_history`; tail-averaged if
        ``tr_cov_tail_start > 0``, else last entry)
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
                if tr_cov_tail_start > 0:
                    tail = [
                        g for g in gvh if g.get("step", 0) >= tr_cov_tail_start
                    ]
                    src = tail or gvh
                    for k in ("nr", "tr_cov", "mean_norm_sq"):
                        vals = [g[k] for g in src if k in g]
                        if vals:
                            rec[k] = float(np.mean(vals))
                else:
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
    r"""Closed-form least-squares for optimal beta from Theorem 4."""
    bias = beta**2 * (kappa + 1.0) ** 2 * b_sq
    var = sigma2d * ((1.0 - beta) * kappa - beta) ** 2
    return bias + var


def _render_lines(
    F: typing.Any,
    data: typing.Dict[str, typing.Dict[float, typing.List]],
) -> mpl_figure.Figure:
    r"""One panel per dataset, beta on x-axis."""
    n_panels = len(F.datasets)
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.8))
    axes = np.atleast_2d(axes).flatten()

    metric_key = _METRIC_TO_KEY[F.metric]
    for i, ds in enumerate(F.datasets):
        ax = axes[i]
        if ds not in data:
            ax.set_visible(False)
            _logging.rank_zero_info("No data for dataset %s, skipping", ds)
            continue
        betas = sorted(data[ds].keys())
        means, errs = [], []
        kappa_avg, b_sq_avg, sigma2d_avg = [], [], []
        for b in betas:
            seed_results = data[ds][b]
            m, e = _aggregate(seed_results, metric_key)
            means.append(m)
            errs.append(e)
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

        ax.plot(
            beta_arr,
            mean_arr,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color="C0",
            zorder=3,
        )
        if np.any(err_arr > 0):
            ax.fill_between(
                beta_arr,
                mean_arr - err_arr,
                mean_arr + err_arr,
                color="C0",
                alpha=0.2,
                linewidth=0,
                zorder=2,
            )

        if (
            F.annotate_reduction
            and len(mean_arr) >= 2
            and np.isfinite(mean_arr[0])
            and np.isfinite(mean_arr[-1])
            and mean_arr[-1] > 0
        ):
            y0 = float(mean_arr[0])
            y1 = float(mean_arr[-1])
            ratio = y0 / y1
            beta_lo = float(beta_arr[0])
            beta_hi = float(beta_arr[-1])
            # Horizontal reference line at the \beta=0 height across the span.
            ax.plot(
                [beta_lo, beta_hi],
                [y0, y0],
                linestyle=":",
                color="gray",
                alpha=0.5,
                linewidth=1.0,
                zorder=1,
            )
            # Vertical double-arrow at the \beta=1 position spanning the gap.
            ax.annotate(
                "",
                xy=(beta_hi, y1),
                xytext=(beta_hi, y0),
                arrowprops=dict(arrowstyle="<->", color="C3", lw=1.4),
                zorder=4,
            )
            # Reduction-ratio label inside the arrow span.
            y_mid = (
                (y0 * y1) ** 0.5 if (y0 > 0 and y1 > 0) else 0.5 * (y0 + y1)
            )
            ax.text(
                beta_hi - 0.02,
                y_mid,
                rf"$\sim\!{ratio:.1f}\!\times$",
                ha="right",
                va="center",
                fontsize=10,
                color="C3",
                fontweight="bold",
                zorder=5,
            )

        if F.overlay_theory and kappa_avg and b_sq_avg and sigma2d_avg:
            kappa = float(np.mean(kappa_avg))
            b_sq = float(np.mean(b_sq_avg))
            sigma2d = float(np.mean(sigma2d_avg))
            beta_grid = np.linspace(0.0, 1.0, 101)
            M_pred = _predict_M(beta_grid, kappa, b_sq, sigma2d)
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
            beta_star = (kappa / (kappa + 1.0)) * (
                sigma2d / (sigma2d + b_sq + 1e-30)
            )
            ax.axvline(
                beta_star,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                color="red",
                label=rf"$\beta^\ast\!=\!{beta_star:.2f}$",
            )

        ax.set_title(ds, fontsize=10, family="monospace")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(_METRIC_LABEL[F.metric])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        # ax.legend(fontsize=7, loc="best")
        ax.grid(True, linestyle=":", alpha=0.3)

    for j in range(len(F.datasets), rows * cols):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def _render_bars(
    F: typing.Any,
    data: typing.Dict[str, typing.Dict[float, typing.List]],
) -> mpl_figure.Figure:
    r"""Single panel: datasets on x-axis, \beta as hue (grouped bars).

    Each group has one bar per \beta; height = mean over seeds with SEM error
    bars. A red star marks the empirical \beta★ (smallest mean) per dataset.
    """
    metric_key = _METRIC_TO_KEY[F.metric]
    valid_datasets = [ds for ds in F.datasets if ds in data]
    if not valid_datasets:
        raise ValueError("No datasets with sweep data available.")
    # Use the union of betas observed across all valid datasets.
    all_betas: typing.Set[float] = set()
    for ds in valid_datasets:
        all_betas.update(data[ds].keys())
    betas = sorted(all_betas)

    n_ds = len(valid_datasets)
    n_beta = len(betas)
    means = np.full((n_ds, n_beta), np.nan)
    sems = np.zeros((n_ds, n_beta))
    for i, ds in enumerate(valid_datasets):
        for j, b in enumerate(betas):
            seed_results = data[ds].get(b) or []
            m, e = _aggregate(seed_results, metric_key)
            if m is not None:
                means[i, j] = m
                sems[i, j] = e if e is not None else 0.0

    cmap = plt.get_cmap("viridis")  # type: ignore
    colors = [cmap(j / max(n_beta - 1, 1)) for j in range(n_beta)]

    fig, ax = plt.subplots(1, 1, figsize=(13.5, 4.4))
    group_width = 0.85
    bar_width = group_width / n_beta
    x_centers = np.arange(n_ds)

    for j, b in enumerate(betas):
        offsets = (j - (n_beta - 1) / 2) * bar_width
        x = x_centers + offsets
        ax.bar(
            x,
            means[:, j],
            width=bar_width,
            color=colors[j],
            edgecolor="black",
            linewidth=0.3,
            yerr=sems[:, j],
            error_kw=dict(ecolor="#444444", elinewidth=0.7, capsize=0),
            label=rf"$\beta\!=\!{b:.1f}$",
        )

    # optimal beta above the empirical minimum bar in each group.
    for i in range(n_ds):
        valid = ~np.isnan(means[i])
        if not valid.any():
            continue
        j_star = int(np.nanargmin(means[i]))
        offset = (j_star - (n_beta - 1) / 2) * bar_width
        x_star = x_centers[i] + offset
        y_star = means[i, j_star] + sems[i, j_star]
        ax.plot(
            [x_star],
            [y_star * 1.06],
            marker="*",
            markersize=11,
            color="#d62728",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="None",
            zorder=10,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        list(valid_datasets),
        fontsize=9,
        family="monospace",
    )
    ax.set_xlabel("dataset")
    ax.set_ylabel(_METRIC_LABEL[F.metric] + r" (mean $\pm$ SEM)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    star_proxy = mpl_lines.Line2D(
        [],
        [],
        marker="*",
        markersize=10,
        color="#d62728",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linestyle="None",
        label=r"empirical $\beta^{\ast}$",
    )
    ax.legend(
        handles=[star_proxy] + handles,
        labels=[r"empirical $\beta^{\ast}$"] + labels,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=True,
        ncol=1,
        fontsize=10,
        handlelength=1.2,
    )

    plt.tight_layout()
    return fig


def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments

    F = flags.FLAGS
    _style.apply_style(F.style)

    tail_start = F.tr_cov_tail_start if F.metric == "tr_cov" else 0
    data = _scan(F.sweep_dir, tr_cov_tail_start=tail_start)
    if F.plot_kind == "bar":
        fig = _render_bars(F, data)
    else:
        fig = _render_lines(F, data)

    out = F.output
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _logging.rank_zero_info("Figure saved to %s", out)

    return 0


if __name__ == "__main__":
    app.run(main)
