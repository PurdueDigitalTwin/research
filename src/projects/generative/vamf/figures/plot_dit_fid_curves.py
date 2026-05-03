"""Three-method FID-vs-step paper figure for the DiT-B/4 ImageNet-256 runs.

The input JSON has shape ``{run_label: [(step, fid), ...], ...}``
with run_label in {"baseline", "vamf_l2", "beta05"}.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_fid_curves -- \\
        --fid_json=logs/vamf/dit_probe/three_method_fid.json \\
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

    base = _filter(data["baseline"], F.skip_below_step)
    vamf = _filter(data["vamf_l2"], F.skip_below_step)
    b05 = _filter(data["beta05"], F.skip_below_step)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))

    base_steps = np.asarray([s for s, _ in base]) / 1000.0
    base_fids = np.asarray([f for _, f in base])
    vamf_steps = np.asarray([s for s, _ in vamf]) / 1000.0
    vamf_fids = np.asarray([f for _, f in vamf])
    b05_steps = np.asarray([s for s, _ in b05]) / 1000.0
    b05_fids = np.asarray([f for _, f in b05])

    ax.semilogy(
        base_steps,
        base_fids,
        marker="s",
        markersize=4,
        linewidth=1.6,
        color=palette["meanflow"],
        label=r"MeanFlow baseline ($\beta\!=\!0$)",
    )
    ax.semilogy(
        b05_steps,
        b05_fids,
        marker="^",
        markersize=4,
        linewidth=1.6,
        color=palette["vamf_tw"],
        label=r"$\beta\!=\!0.5$ (interior)",
    )
    ax.semilogy(
        vamf_steps,
        vamf_fids,
        marker="o",
        markersize=4,
        linewidth=1.6,
        color=palette["vamf_l2"],
        label=r"VaMF-L$_{2}$ ($\beta\!=\!1$)",
    )

    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel(r"FID (lower is better)")
    ax.set_title(
        r"DiT-B/4 ImageNet-$256$ FID convergence at three $\beta$ values"
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
