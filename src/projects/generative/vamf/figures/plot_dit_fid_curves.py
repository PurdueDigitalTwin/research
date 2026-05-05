"""Four-method FID-vs-step paper figure for the DiT-B/4 ImageNet-256 runs.

The input JSON has shape ``{run_label: [(step, fid), ...], ...}``
with run_label in {"baseline", "beta025", "beta05", "beta1"}.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_fid_curves -- \\
        --fid_json=logs/vamf/dit_probe/four_method_fid.json \\
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
    help="JSON file with {run_label: [[step, fid], ...]} for the 3 runs.",
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)
flags.DEFINE_integer(
    name="skip_below_step",
    default=15000,
    help=(
        "Drop early-training datapoints below this step from the plot, "
        "since the post-fix transient spike at step 10k dominates the "
        "y-axis and obscures the converged-regime ordering."
    ),
)


def _filter(
    trajectory: typing.List[typing.Tuple[int, typing.Any]],
    min_step: int,
) -> typing.List[typing.Tuple]:
    r"""Drop points below min_step."""
    return [(s, f) for s, f in trajectory if s >= min_step]


def main(argv: typing.List[str]) -> int:
    r"""Render the three-method FID convergence figure."""
    del argv  # unused arguments
    F = flags.FLAGS

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    with open(F.fid_json) as f:
        data = json.load(f)

    # Each label is keyed in the input JSON with the (run_label, key, marker, color, plot_kwargs) ordering.
    series = [
        ("baseline", r"$\beta\!=\!0$ (baseline)", "s", palette["meanflow"]),
        (
            "beta025",
            r"$\beta\!=\!0.25$ (interior)",
            "D",
            palette.get("beta025", "#1f77b4"),
        ),
        ("beta05", r"$\beta\!=\!0.5$ (interior)", "^", palette["vamf_tw"]),
        ("beta1", r"$\beta\!=\!1$ (corner)", "o", palette["vamf_l2"]),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))
    for key, label, marker, color in series:
        if key not in data:
            continue
        traj = _filter(data[key], F.skip_below_step)
        steps = np.asarray([s for s, _ in traj]) / 1000.0
        fids = np.asarray([f for _, f in traj])
        ax.semilogy(
            steps,
            fids,
            marker=marker,
            markersize=4,
            linewidth=1.6,
            color=color,
            label=label,
        )

    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel(r"FID (lower is better)")
    ax.set_title(
        r"DiT-B/4 ImageNet-$256$ FID convergence at four $\beta$ values"
    )
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

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
