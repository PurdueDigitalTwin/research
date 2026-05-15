"""DiT-B/4 / ImageNet-256 FID-MSE landscape mismatch figure.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_fid_curves --\\
        --fid_json=logs/vamf/dit_probe/four_method_fid.json \\
        --baseline_floor=12.12 \\
        --beta_star_matrix=0.94 \\
        --output=docs/generative/vamf/results/dit_fid_curves.pdf
"""

import json
import os
import typing

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np

from src.projects.generative.vamf.figures import _style
from src.utilities import logging as _logging

# ==============================================================================
# Flags
flags.DEFINE_enum(
    name="style",
    default=_style.DEFAULT_STYLE,
    enum_values=list(_style.STYLES),
    help="Render target ('paper' | 'slides').",
)
flags.DEFINE_string(
    name="fid_json",
    default=None,
    required=True,
    help=(
        "JSON file with {run_label: [[step, fid], ...]} for the 4 runs. "
        "Expected keys: baseline, beta025, beta05, beta1."
    ),
)
flags.DEFINE_float(
    name="baseline_floor",
    default=None,
    help=(
        "Override for the baseline FID floor used to compute ΔFID. "
        "If None, the baseline run's final-logged FID is used."
    ),
)
flags.DEFINE_float(
    name="beta_star_matrix",
    default=0.94,
    help=(
        "Matrix-form gradient-MSE optimum to mark on the x-axis "
        "(directly probed in our DiT measurement)."
    ),
)
flags.DEFINE_enum(
    name="plot_kind",
    default="two_panel",
    enum_values=("fid_only", "two_panel"),
    help=(
        "'fid_only' = single FID-MSE landscape panel; "
        "'two_panel' = side-by-side variance panel (a) + FID panel (b)."
    ),
)
flags.DEFINE_string(
    name="probe_dir",
    default=None,
    help=(
        "Directory containing probe_<run>_<step>.json files (variance "
        "amplification probe). Required when plot_kind=two_panel."
    ),
)
flags.DEFINE_string(
    name="probe_run",
    default="baseline",
    help="Run label whose probe data is shown in panel (a) (default: baseline).",
)
flags.DEFINE_integer(
    name="probe_step",
    default=80_000,
    help="Training step at which the variance probe was measured.",
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)

_RUN_KEYS = ("baseline", "beta025", "beta05", "beta1")
_BETA_OF_KEY = {"baseline": 0.0, "beta025": 0.25, "beta05": 0.5, "beta1": 1.0}


def _final_fid(traj: typing.Sequence[typing.Sequence[float]]) -> float:
    return float(traj[-1][1])


def _is_finished(key: str, total_steps: int = 295_000) -> bool:
    r"""All four runs reached the matched-step horizon at submission.

    Kept as a hook in case a future revision needs to flag in-flight runs.
    """
    del key, total_steps
    return True


def _draw_variance_panel(
    ax: plt.Axes,
    probe_dir: str,
    probe_run: str,
    probe_step: int,
    title_prefix: str = "(a) ",
) -> None:
    r"""Per-step loss variance: stochastic vs deterministic tangent at fixed step.

    Reads ``probe_<run>_<step>.json`` and plots
    ``stochastic_var`` and ``deterministic_var`` vs t on a log y-axis.
    """
    p = os.path.join(probe_dir, f"probe_{probe_run}_{probe_step}.json")
    with open(p) as f:
        d = json.load(f)
    stoch, det = {}, {}
    for k, v in d["results"]["exp1_variance_amplification"].items():
        t = float(k.split("=")[1])
        stoch[t] = v["stochastic_var"]
        det[t] = v["deterministic_var"]
    ts = sorted(stoch.keys())
    ys_stoch = [stoch[t] for t in ts]
    ys_det = [det[t] for t in ts]

    color_stoch = "#f68b3c"
    color_det = "#6a9a5e"
    ax.plot(
        ts,
        ys_stoch,
        marker="o",
        markersize=7,
        linewidth=1.8,
        color=color_stoch,
        label="Loss with stochastic tangent",
    )
    ax.plot(
        ts,
        ys_det,
        marker="s",
        markersize=7,
        linewidth=1.8,
        color=color_det,
        label="Loss with deterministic tangent.",
    )
    # Bracket the gap at the largest t.
    t_gap = ts[-1]
    ratio = ys_stoch[-1] / ys_det[-1]
    ax.annotate(
        "",
        xy=(t_gap + 0.005, ys_stoch[-1]),
        xytext=(t_gap + 0.005, ys_det[-1]),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(
        t_gap - 0.01,
        np.sqrt(ys_stoch[-1] * ys_det[-1]),
        rf"$\sim\!{ratio:.0f}\!\times$",
        ha="right",
        va="center",
        fontsize=10,
        color="black",
        fontweight="bold",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"timepoint $t$")
    ax.set_ylabel(r"Per-step Loss Variance $\mathrm{Var}[\ell]")
    ax.set_title(
        "(a) Per-step loss variance with different tangent.",
        fontsize=10,
        pad=6,
    )
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xlim(0.05, 0.97)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def _draw_fid_panel(
    ax: plt.Axes,
    F,
    palette,
    title_prefix: str = "",
) -> None:
    r"""Plot the FID-vs-beta panel."""

    with open(F.fid_json) as f:
        data = json.load(f)
    for k in _RUN_KEYS:
        if k not in data or not data[k]:
            raise ValueError(
                f"missing or empty trajectory for run '{k}' in {F.fid_json}"
            )

    baseline_floor = (
        F.baseline_floor
        if F.baseline_floor is not None
        else _final_fid(data["baseline"])
    )

    points = []
    for k in _RUN_KEYS:
        beta = _BETA_OF_KEY[k]
        delta = _final_fid(data[k]) - baseline_floor
        points.append((beta, delta, _is_finished(k)))
    points.sort(key=lambda p: p[0])

    beta05_delta = next(d for b, d, _ in points if abs(b - 0.5) < 1e-9)
    A_anchor = beta05_delta / 0.25
    beta_grid = np.linspace(0.0, 1.0, 201)
    mse_pred = A_anchor * beta_grid**2

    ax.plot(
        beta_grid,
        mse_pred,
        linestyle="--",
        linewidth=1.4,
        color="gray",
        alpha=0.85,
        label=r"MSE-prediction $c\cdot\beta^{2}$",
    )

    color_finished = palette.get("meanflow", "#1f77b4")
    color_inflight = palette.get("vamf_tw", "#ff7f0e")
    for b, delta, finished in points:
        marker = "o" if finished else "v"
        color = color_finished if finished else color_inflight
        ax.plot(
            [b],
            [delta],
            marker=marker,
            markersize=10,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linestyle="None",
            zorder=5,
        )
        offset_y = 0.7 if delta > 0.5 else 0.6
        suffix = "" if finished else "\n(still training)"
        ax.annotate(
            rf"$\beta\!=\!{b}$" + suffix,
            xy=(b, delta),
            xytext=(b + 0.02, delta + offset_y),
            fontsize=9,
            color=color,
        )

    pred_at_1 = A_anchor * 1.0**2
    emp_at_1 = next(d for b, d, _ in points if abs(b - 1.0) < 1e-9)
    if emp_at_1 > pred_at_1:
        ax.annotate(
            "",
            xy=(0.97, emp_at_1),
            xytext=(0.97, pred_at_1),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.4),
        )
        ax.text(
            0.94,
            (emp_at_1 + pred_at_1) / 2,
            rf"$\sim\!{emp_at_1 / pred_at_1:.1f}\times$ super-linear",
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            color="#d62728",
        )

    beta_star = float(F.beta_star_matrix)
    ax.axvline(
        beta_star,
        ymin=0,
        ymax=0.05,
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.annotate(
        rf"$\beta^{{\ast}}_{{\mathrm{{matrix}}}}\!\approx\!{beta_star:.2f}$",
        xy=(beta_star, -2.8),
        xytext=(beta_star - 0.1, -1.0),
        fontsize=9,
        color="#2ca02c",
        ha="center",
        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8),
    )
    ax.annotate(
        "FID-optimum",
        xy=(0.00, 0),
        xytext=(0.06, -1.3),
        fontsize=9,
        color=color_finished,
        ha="left",
    )

    ax.set_xlabel(r"tangent-mixing coefficient $\beta$")
    ax.set_ylabel(
        r"$\Delta\mathrm{FID} = \mathrm{FID}(\beta) - \mathrm{FID}(\beta=0)$"
    )
    ax.set_title(
        rf"{title_prefix}FID at converge vs. $\beta$",
        pad=6,
        fontsize=10,
    )
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-2.8, max(emp_at_1, pred_at_1) * 1.15)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)


def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments
    F = flags.FLAGS

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    if F.plot_kind == "two_panel":
        if not F.probe_dir:
            raise ValueError(
                "--probe_dir is required when --plot_kind=two_panel"
            )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.7))
        _draw_variance_panel(
            ax1,
            probe_dir=F.probe_dir,
            probe_run=F.probe_run,
            probe_step=F.probe_step,
            title_prefix="(a) ",
        )
        _draw_fid_panel(ax2, F, palette, title_prefix="(b) ")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.7))
        _draw_fid_panel(ax, F, palette, title_prefix="")

    plt.tight_layout()
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
