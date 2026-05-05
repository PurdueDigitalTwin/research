"""DiT-B/4 / ImageNet-256 FID-vs-MSE landscape mismatch figure.

x-axis: tangent-mixing coefficient ``β ∈ [0, 1]``.
y-axis: ΔFID(β) ≜ FID(β) - FID_baseline at each run's converged
(final-logged) step.
markers: empirical ΔFID at β ∈ {0, 0.25, 0.5, 1} (β = 0.25 marked as
"still training" if the run is not finished).
overlay: MSE-prediction curve ``β² · A`` with A fit to anchor at the
β = 0.5 empirical marker (so the curve passes through the interior
empirical point and visualizes super-linearity at β = 1).
annotations: matrix-form ``β★_matrix ≈ 0.94`` (gradient-MSE optimum)
on the x-axis, FID-optimum label at β = 0, super-linearity ratio at β = 1.

Input JSON: ``{run_label: [[step, fid], ...], ...}`` for run_label in
{"baseline", "beta025", "beta05", "beta1"}, as produced by the
four-method wandb pull script.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_fid_curves -- \\
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
    r"""Treat as finished if final step is within 5k of the configured horizon.

    The β=0 baseline run crashed at ~240k; β=0.5 and β=1 reached 295k;
    β=0.25 is still training at submission. This heuristic flags the
    in-flight run for marker styling.
    """
    return key != "beta025"


def main(argv: typing.List[str]) -> int:
    del argv
    F = flags.FLAGS

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    with open(F.fid_json) as f:
        data = json.load(f)
    for k in _RUN_KEYS:
        if k not in data or not data[k]:
            raise ValueError(f"missing or empty trajectory for run '{k}' in {F.fid_json}")

    baseline_floor = (
        F.baseline_floor
        if F.baseline_floor is not None
        else _final_fid(data["baseline"])
    )

    # Collect (β, ΔFID, finished?) per run.
    points = []
    for k in _RUN_KEYS:
        beta = _BETA_OF_KEY[k]
        delta = _final_fid(data[k]) - baseline_floor
        points.append((beta, delta, _is_finished(k)))
    points.sort(key=lambda p: p[0])

    # Anchor the MSE-prediction curve β² · A at β = 0.5.
    # Find the β=0.5 point's ΔFID and solve for A.
    beta05_delta = next(d for b, d, _ in points if abs(b - 0.5) < 1e-9)
    A_anchor = beta05_delta / 0.25
    beta_grid = np.linspace(0.0, 1.0, 201)
    mse_pred = A_anchor * beta_grid**2

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.7))
    # MSE prediction curve.
    ax.plot(
        beta_grid,
        mse_pred,
        linestyle="--",
        linewidth=1.4,
        color="gray",
        alpha=0.85,
        label=r"MSE-prediction curve $\beta^{2}\cdot A$ "
        r"(anchored at $\beta=0.5$)",
    )

    # Empirical markers.
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

    # Super-linearity gap annotation at β=1.
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

    # Matrix-form β★ marker on x-axis.
    beta_star = float(F.beta_star_matrix)
    ax.axvline(
        beta_star,
        ymin=0,
        ymax=0.05,
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.annotate(
        rf"$\beta^{{\star}}_{{\mathrm{{matrix}}}}\!\approx\!{beta_star:.2f}$ "
        r"(gradient-MSE optimum)",
        xy=(beta_star, 0),
        xytext=(0.50, -2.6),
        fontsize=9,
        color="#2ca02c",
        ha="center",
        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8),
    )
    # FID-optimum at β=0 marker.
    ax.annotate(
        "FID-optimum",
        xy=(0.00, 0),
        xytext=(0.06, -1.6),
        fontsize=9,
        color=color_finished,
        ha="left",
    )

    ax.set_xlabel(r"tangent-mixing coefficient $\beta$")
    ax.set_ylabel(
        r"$\Delta\mathrm{FID} = \mathrm{FID}(\beta) - \mathrm{FID}_{\mathrm{baseline}}$"
    )
    ax.set_title(
        r"DiT-B/4 / ImageNet-$256$: FID-vs-MSE landscape mismatch",
        pad=10,
    )
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-3.5, max(emp_at_1, pred_at_1) * 1.15)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)

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
