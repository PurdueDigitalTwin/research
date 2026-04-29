"""Plot the teaser figure for the VaMF paper.

Three-panel hero figure:
  (a) The phenomenon — variance amplification ratio Var(L_stoch) / Var(L_det)
      across t, showing how the stochastic MeanFlow loss estimator's variance
      explodes by orders of magnitude in the middle of the integration window.
  (b) The mechanism — ||(t - r) J - I||_F vs t, the Frobenius norm of the
      flow Jacobian deviation that drives panel (a) and that VaMF's trace
      weight directly downweights.
  (c) The fix — generated samples on swiss_roll for vanilla MeanFlow vs
      VaMF-TW (with sigma_t = t^2), against the reference distribution.

Inputs (paths set via flags, defaults aligned with the canonical layout):
  --diagnostic_results : logs/vamf/diagnostics/diagnostics_<dataset>_<seed>.json
  --samples_dir        : logs/vamf/toy_exp/200k
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
    name="samples_dir",
    default=None,
    required=True,
    help="Directory with <dataset>_<method>_<seed>.npz sample files.",
)
flags.DEFINE_string(
    name="samples_dataset",
    default="swiss_roll",
    help="Toy dataset to draw the panel-(c) sample comparison from.",
)
flags.DEFINE_integer(
    name="samples_seed",
    default=42,
    help="Seed for the panel-(c) sample comparison.",
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


def _load_npz(samples_dir, dataset, method, seed):
    path = os.path.join(samples_dir, f"{dataset}_{method}_{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _load_swd(samples_dir, dataset, method, seed):
    path = os.path.join(samples_dir, f"{dataset}_{method}_{seed}.json")
    with open(path) as f:
        d = json.load(f)
    final = d.get("final", {})
    return final.get("swd1", final.get("swd", float("nan")))


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
        1.0, color=palette["ref"], linestyle="--", linewidth=0.6, alpha=0.7
    )

    # Annotate the peak.
    peak_i = int(np.argmax(ratio))
    ax.annotate(
        f"$\\sim {ratio[peak_i]:.0f}\\times$",
        xy=(t[peak_i], ratio[peak_i]),
        xytext=(t[peak_i] + 0.05, ratio[peak_i] * 0.35),
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
    ax.set_title("(a) phenomenon", loc="left", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))


def panel_mechanism(ax, diag, palette):
    r"""Plot ||(t-r) J - I||_F vs t (linear y)."""
    data = diag["exp4_jacobian_norm"]
    keys = _sorted_t_keys(data)
    t = np.array([float(k.split("=")[1]) for k in keys])
    j_mean = np.array([data[k]["J_norm_mean"] for k in keys])
    j_std = np.sqrt(np.array([data[k]["J_norm_sq_std"] for k in keys]))

    ax.plot(
        t,
        j_mean,
        "s-",
        color=palette["trace"],
        markersize=4,
        linewidth=1.4,
    )
    ax.fill_between(
        t,
        j_mean - 0.5 * j_std,
        j_mean + 0.5 * j_std,
        color=palette["trace"],
        alpha=0.18,
        linewidth=0,
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|(t-r)\,J_z u_\theta - I\|_F$")
    ax.set_title("(b) mechanism", loc="left", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))


def panel_fix(axes, samples_dir, dataset, seed, palette):
    r"""Plot side-by-side scatter: MeanFlow vs VaMF-TW samples."""
    # Reference is identical across methods; load from MF file.
    npz_mf = _load_npz(samples_dir, dataset, "meanflow", seed)
    npz_tw = _load_npz(samples_dir, dataset, "vamf_tw", seed)
    ref = npz_mf["reference"]
    swd_mf = _load_swd(samples_dir, dataset, "meanflow", seed)
    swd_tw = _load_swd(samples_dir, dataset, "vamf_tw", seed)

    pad = 0.15
    lo, hi = float(ref[:, 0].min()), float(ref[:, 0].max())
    px = (hi - lo) * pad
    xlim = (lo - px, hi + px)
    lo, hi = float(ref[:, 1].min()), float(ref[:, 1].max())
    py = (hi - lo) * pad
    ylim = (lo - py, hi + py)

    for ax, gen, swd, label, color in zip(
        axes,
        [npz_mf["generated"], npz_tw["generated"]],
        [swd_mf, swd_tw],
        ["MeanFlow", "VaMF (TW)"],
        [palette["mf"], palette["vamf"]],
    ):
        # Reference as a translucent backdrop.
        ax.scatter(
            ref[:, 0],
            ref[:, 1],
            s=3.5,
            alpha=0.22,
            c=palette["ref"],
            edgecolors="none",
            rasterized=True,
        )
        # Generated samples on top.
        ax.scatter(
            gen[:, 0],
            gen[:, 1],
            s=3.5,
            alpha=0.6,
            c=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Per-axis tag with method + SWD; styled to read on either background.
        ax.text(
            0.04,
            0.96,
            f"{label}\n$\\mathrm{{SW}}_1 = {swd:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=palette["bbox_text"],
            bbox=dict(
                facecolor=palette["bbox_face"],
                edgecolor="none",
                alpha=palette["bbox_alpha"],
                pad=2.0,
            ),
        )
        # Hide spines for a clean sample-grid look.
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Title spans the first sample axis only (kept short to match panels a/b).
    axes[0].set_title("(c) fix", loc="left", fontweight="bold")


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

    # Layout: (a) | (b) | (c1, c2). Panel c is wider so the two sample
    # subplots fit side by side without squashing.
    fig = plt.figure(figsize=(6.85, 2.0))
    gs = gridspec.GridSpec(
        1,
        4,
        width_ratios=[1.0, 1.0, 0.85, 0.85],
        wspace=0.55,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_c2 = fig.add_subplot(gs[0, 3])

    panel_phenomenon(ax_a, diag, palette)
    panel_mechanism(ax_b, diag, palette)
    panel_fix(
        [ax_c1, ax_c2],
        F.samples_dir,
        F.samples_dataset,
        F.samples_seed,
        palette,
    )

    out_path = os.path.join(F.work_dir, F.filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}  (style={F.style})")


if __name__ == "__main__":
    app.run(main=main)
