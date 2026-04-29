"""Plot the teaser figure for the VaMF paper.

Inputs (paths set via flags):
  --diagnostic_results : logs/vamf/diagnostics/diagnostics_<dataset>_<seed>.json
  --work_dir           : output directory for the PDF
"""

import json
import os
import typing

from absl import app
from absl import flags
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

from src.projects.generative.vamf.figures import _style

# ==============================================================================
# Flags
flags.DEFINE_enum(
    "style",
    _style.DEFAULT_STYLE,
    list(_style.STYLES),
    "Render target. 'paper' = light/serif (camera-ready); "
    "'slides' = dark/sans-serif (talks).",
)
flags.DEFINE_string(
    name="diagnostic_results",
    default=None,
    required=True,
    help="Path to the diagnostic JSON (panels a + b).",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Output directory for the figure.",
    required=True,
)
flags.DEFINE_string(
    name="filename",
    default="teaser.pdf",
    help="Output filename.",
)


# ==============================================================================
# Helper functions
def _sorted_t_keys(d: typing.Dict[str, typing.Any]) -> typing.List[str]:
    return sorted(d.keys(), key=lambda k: float(k.split("=")[1]))


# ==============================================================================
# Panel constructors
def panel_phenomenon(ax, diag, palette):
    r"""Plot variance amplification ratio vs t (semilog y)."""
    data = diag["exp1_variance_amplification"]
    keys = _sorted_t_keys(data)
    t = np.array([float(k.split("=")[1]) for k in keys])
    ratio = np.array([data[k]["variance_ratio"] for k in keys])

    ax.semilogy(
        t,
        ratio,
        "o-",
        color=palette["ratio"],
        markersize=4,
        linewidth=1.4,
    )
    ax.axhline(
        1.0,
        color=palette["ref"],
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    # Annotate the peak.
    peak_i = int(np.argmax(ratio))
    ax.annotate(
        f"$\\sim {ratio[peak_i]:.0f}\\times$",
        xy=(t[peak_i], ratio[peak_i]),
        xytext=(t[peak_i] + 0.15, ratio[peak_i] * 0.7),
        fontsize=8,
        ha="left",
        color=plt.rcParams["text.color"],
        arrowprops=dict(
            arrowstyle="-",
            color=plt.rcParams["text.color"],
            lw=0.5,
            shrinkA=2,
            shrinkB=2,
        ),
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel(
        r"$\mathrm{Var}[\mathcal{L}_{\mathrm{stoch}}]\;/"
        r"\;\mathrm{Var}[\mathcal{L}_{\mathrm{det}}]$"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))


def panel_mechanism(ax, diag, palette):
    r"""Co-plot per-sample loss variance and ||(t-r)J - I||_F vs t."""
    e1 = diag["exp1_variance_amplification"]
    e4 = diag["exp4_jacobian_norm"]

    keys = _sorted_t_keys(e4)
    t = np.array([float(k.split("=")[1]) for k in keys])
    j_norm = np.array([e4[k]["J_norm_mean"] for k in keys])

    # Match exp1 keys to exp4 t-grid (both are tenths between 0.1 and 0.9).
    def _match(k4: str) -> str:
        tv = float(k4.split("=")[1])
        for k1 in e1:
            if abs(float(k1.split("=")[1]) - tv) < 1e-6:
                return k1
        raise KeyError(f"No exp1 entry matching {k4}")

    loss_var = np.array([e1[_match(k)]["deterministic_var"] for k in keys])

    color_loss = palette["loss"]
    color_jac = palette["trace"]

    (line_loss,) = ax.plot(
        t,
        loss_var,
        "o-",
        color=color_loss,
        markersize=4,
        linewidth=1.4,
        label=r"$\mathrm{Var}[\mathcal{L}_{\mathrm{det}}]$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(
        r"$\mathrm{Var}[\mathcal{L}_{\mathrm{det}}]$",
        color=color_loss,
    )
    ax.tick_params(axis="y", labelcolor=color_loss)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))

    ax_r = ax.twinx()
    (line_jac,) = ax_r.plot(
        t,
        j_norm,
        "s--",
        color=color_jac,
        markersize=4,
        linewidth=1.4,
        label=r"$\|(t-r)\,\partial_x u_\theta - I\|_F$",
    )
    ax_r.set_ylabel(
        r"$\|(t-r)\,\partial_x u_\theta - I\|_F$",
        color=color_jac,
    )
    ax_r.tick_params(axis="y", labelcolor=color_jac)

    # Single legend for both axes.
    ax.legend(
        handles=[line_loss, line_jac],
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.6,
    )


# ==============================================================================
# Main entry point
def main(argv: typing.List[str]) -> None:
    del argv  # unused arguments

    F = flags.FLAGS
    os.makedirs(F.work_dir, exist_ok=True)

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    with open(F.diagnostic_results) as f:
        diag = json.load(f)

    fig = plt.figure(figsize=(5.5, 2.0))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 1.2],
        wspace=0.5,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_phenomenon(ax_a, diag, palette)
    panel_mechanism(ax_b, diag, palette)

    out_path = os.path.join(F.work_dir, F.filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}  (style={F.style})")


if __name__ == "__main__":
    app.run(main=main)
