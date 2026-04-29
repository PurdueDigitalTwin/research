import json
import os
import typing

from absl import app
from absl import flags
from matplotlib import axes as mpl_axes
from matplotlib import pyplot as plt
import numpy as np

from src.projects.generative.vamf.figures import _style

# ==============================================================================
# Flags
flags.DEFINE_enum(
    name="style",
    default=_style.DEFAULT_STYLE,
    enum_values=list(_style.STYLES),
    help=(
        "Render target. 'paper' = light/serif (camera-ready); "
        "'slides' = dark/sans-serif (talks)."
    ),
)
flags.DEFINE_string(
    name="base_dir",
    default=None,
    required=True,
    help=(
        "Root directory containing experiment subdirectories: "
        "200k/, 200k_t2/, dgmm_scaling/, dgmm_sigma_none/, "
        "dgmm_sigma_t2/, dgmm_sigma_learned/."
    ),
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Output directory for figures.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Seed used in the experiments.",
)

# ==============================================================================
# Constants
DATASETS = [
    "checkerboard",
    "eight_gaussians",
    "two_moons",
    "swiss_roll",
]
METHODS = ["meanflow", "vamf_l2", "vamf_tw"]

DATASET_LABELS = {
    "checkerboard": "Checkerboard",
    "eight_gaussians": "8-Gaussians",
    "two_moons": "Two Moons",
    "swiss_roll": "Swiss Roll",
}
METHOD_LABELS = {
    "meanflow": "MeanFlow",
    "vamf_l2": r"VaMF ($\ell_2$)",
    "vamf_tw": r"VaMF (TW, $\sigma_t{=}t^2$)",
}
DGMM_DIMS = [2, 4, 8, 16, 32, 64]

# Labels for the σ_t ablation (DGMM scaling figure).
SIGMA_LABELS = {
    "sigma_none": r"TW ($\sigma_t{=}1$)",
    "sigma_t2": r"TW ($\sigma_t{=}t^2$)",
    "sigma_learned": r"TW ($\sigma_t$ learned)",
}

# Colors are populated from the active style palette in ``main()`` so the
# same plotting code renders both the light (paper) and dark (slides)
# variants. They start empty and are filled by ``_apply_style_palette``.
METHOD_COLORS: typing.Dict[str, str] = {}
SIGMA_COLORS: typing.Dict[str, str] = {}


def _apply_style_palette(name: str) -> None:
    """Activate the named matplotlib style and populate color dicts."""
    _style.apply_style(name)
    palette = _style.palette(name)
    METHOD_COLORS.clear()
    METHOD_COLORS.update({m: palette[m] for m in METHODS})
    SIGMA_COLORS.clear()
    SIGMA_COLORS.update({k: palette[k] for k in SIGMA_LABELS})


# ==============================================================================
# Helpers
def _axis_lim(values: np.ndarray, margin: float = 0.15):
    """Compute axis limits with padding."""
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _load_history(path: str) -> typing.Optional[typing.List[dict]]:
    """Load a JSON result file and return its history list."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["history"]


def _swd_key(record: dict, p: int = 1) -> str:
    """Pick the SW_p key in a history record (back-compat with legacy ``swd``)."""
    new_key = f"swd{p}"
    if new_key in record:
        return new_key
    if p == 1 and "swd" in record:
        return "swd"
    raise KeyError(f"No {new_key} or legacy swd key in record")


# ==============================================================================
# Figure 1: Samples grid (datasets x methods)
#   MeanFlow & VaMF-L2 from 200k/; VaMF-TW from 200k_t2/
def plot_samples_grid(
    base_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """4-row by 4-col grid: Reference + MeanFlow + VaMF-L2 + VaMF-TW(t^2)."""
    n_rows = len(DATASETS)
    n_cols = 1 + len(METHODS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.4),
        squeeze=False,
    )

    for row, dataset in enumerate(DATASETS):
        # Load reference from any available npz
        ref = None
        for sub in ["200k", "200k_t2"]:
            for method in METHODS:
                ref_path = os.path.join(
                    base_dir, sub, f"{dataset}_{method}_{seed}.npz"
                )
                if os.path.exists(ref_path):
                    ref = np.load(ref_path)["reference"]
                    break
            if ref is not None:
                break

        if ref is None:
            for col in range(n_cols):
                axes[row, col].text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=axes[row, col].transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="gray",
                )
                axes[row, col].tick_params(
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False,
                )
            axes[row, 0].set_ylabel(
                DATASET_LABELS[dataset],
                fontsize=11,
                fontweight="bold",
            )
            continue

        xlim = _axis_lim(ref[:, 0])
        ylim = _axis_lim(ref[:, 1])

        # Reference column
        ax = axes[row, 0]
        ax.scatter(
            ref[:, 0],
            ref[:, 1],
            s=3,
            alpha=0.5,
            c="#333333",
            edgecolors="none",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        if row == 0:
            ax.set_title("Reference", fontweight="bold")
        ax.set_ylabel(
            DATASET_LABELS[dataset],
            fontsize=11,
            fontweight="bold",
        )

        # Method columns
        for col, method in enumerate(METHODS):
            ax = axes[row, 1 + col]
            # VaMF-TW uses 200k_t2/ (σ_t=t²); others use 200k/
            subdir = "200k_t2" if method == "vamf_tw" else "200k"
            npz_path = os.path.join(
                base_dir,
                subdir,
                f"{dataset}_{method}_{seed}.npz",
            )
            if not os.path.exists(npz_path):
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="gray",
                )
            else:
                gen = np.load(npz_path)["generated"]
                ax.scatter(
                    gen[:, 0],
                    gen[:, 1],
                    s=3,
                    alpha=0.5,
                    c=METHOD_COLORS[method],
                    edgecolors="none",
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
            if row == 0:
                ax.set_title(METHOD_LABELS[method], fontweight="bold")

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_samples.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Figure 2: Training SWD curves (all 4 datasets, 3 methods)
#   MeanFlow & VaMF-L2 from 200k/; VaMF-TW from 200k_t2/
def plot_training_curves(
    base_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """One subplot per dataset, all methods overlaid."""
    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.0))
    if n == 1:
        axes = [axes]

    for idx, dataset in enumerate(DATASETS):
        ax: mpl_axes.Axes = axes[idx]  # type: ignore
        for method in METHODS:
            subdir = "200k_t2" if method == "vamf_tw" else "200k"
            hist = _load_history(
                os.path.join(
                    base_dir,
                    subdir,
                    f"{dataset}_{method}_{seed}.json",
                )
            )
            if hist is None:
                continue
            steps = np.array([h["step"] for h in hist]) / 1000
            try:
                key = _swd_key(hist[0], p=1)
            except KeyError:
                key = "raw_loss"
            values = np.array([h[key] for h in hist])
            smoothed = _smooth(values, window=7)

            ax.plot(
                steps,
                values,
                color=METHOD_COLORS[method],
                linewidth=0.4,
                alpha=0.25,
            )
            ax.plot(
                steps,
                smoothed,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.5,
                alpha=0.9,
            )

        ax.set_xlabel("Step (k)")
        if idx == 0:
            ax.set_ylabel("Sliced Wasserstein Dist.")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_training_curves.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Figure 3: Swiss Roll stability — σ_t=1 vs σ_t=t^2
def plot_swiss_roll_stability(
    base_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """Side-by-side: Swiss Roll SWD curves for TW(σ=1) vs TW(σ=t^2),
    plus MeanFlow and VaMF-L2 baselines."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    curves = [
        ("200k", "meanflow", "MeanFlow", "#1f77b4", "-"),
        ("200k", "vamf_l2", r"VaMF ($\ell_2$)", "#ff7f0e", "-"),
        ("200k", "vamf_tw", r"VaMF TW ($\sigma_t{=}1$)", "#d62728", "--"),
        ("200k_t2", "vamf_tw", r"VaMF TW ($\sigma_t{=}t^2$)", "#2ca02c", "-"),
    ]
    for subdir, method, label, color, ls in curves:
        hist = _load_history(
            os.path.join(
                base_dir,
                subdir,
                f"swiss_roll_{method}_{seed}.json",
            )
        )
        if hist is None:
            continue
        steps = np.array([h["step"] for h in hist]) / 1000
        swds = np.array([h[_swd_key(h, p=1)] for h in hist])
        smoothed = _smooth(swds, window=7)

        ax.plot(
            steps,
            swds,
            color=color,
            linewidth=0.4,
            alpha=0.2,
        )
        ax.plot(
            steps,
            smoothed,
            color=color,
            label=label,
            linewidth=1.8,
            alpha=0.9,
            linestyle=ls,
        )

    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Sliced Wasserstein Dist.")
    ax.set_title("Swiss Roll: Convergence Stability", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_swiss_roll_stability.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Figure 4: DGMM dimension scaling — σ_t ablation
def plot_dgmm_scaling(
    base_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """Two-panel figure: best SWD and convergence speed vs dimension.

    Shows MeanFlow, VaMF-L2 baselines alongside three TW σ_t variants.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # --- Baselines (MeanFlow, VaMF-L2) from dgmm_scaling/ ---
    baseline_dir = os.path.join(base_dir, "dgmm_scaling")
    for method, label, color, marker in [
        ("meanflow", "MeanFlow", "#1f77b4", "^"),
        ("vamf_l2", r"VaMF ($\ell_2$)", "#ff7f0e", "D"),
    ]:
        best_swds, conv_steps = [], []
        dims_data, dims_conv = [], []
        for d in DGMM_DIMS:
            hist = _load_history(
                os.path.join(
                    baseline_dir,
                    f"dgmm_{d}_{method}_{seed}.json",
                )
            )
            if hist is None:
                continue
            swds = np.array([h[_swd_key(h, p=1)] for h in hist])
            steps = np.array([h["step"] for h in hist])
            best_swds.append(float(swds.min()))
            dims_data.append(d)
            below = np.where(swds < 0.10)[0]
            if len(below) > 0:
                conv_steps.append(int(steps[below[0]]) // 1000)
                dims_conv.append(d)

        if dims_data:
            ax1.plot(
                dims_data,
                best_swds,
                f"{marker}-",
                color=color,
                label=label,
                linewidth=1.5,
                markersize=5,
                alpha=0.7,
            )
        if dims_conv:
            ax2.plot(
                dims_conv,
                conv_steps,
                f"{marker}-",
                color=color,
                label=label,
                linewidth=1.5,
                markersize=5,
                alpha=0.7,
            )

    # --- TW σ_t variants from dgmm_sigma_{none,t2,learned}/ ---
    sigma_dirs = {
        "sigma_none": os.path.join(base_dir, "dgmm_sigma_none"),
        "sigma_t2": os.path.join(base_dir, "dgmm_sigma_t2"),
        "sigma_learned": os.path.join(base_dir, "dgmm_sigma_learned"),
    }
    sigma_markers = {
        "sigma_none": "s",
        "sigma_t2": "o",
        "sigma_learned": "P",
    }
    for key, ddir in sigma_dirs.items():
        best_swds, conv_steps = [], []
        dims_data, dims_conv = [], []
        for d in DGMM_DIMS:
            hist = _load_history(
                os.path.join(
                    ddir,
                    f"dgmm_{d}_vamf_tw_{seed}.json",
                )
            )
            if hist is None:
                continue
            swds = np.array([h[_swd_key(h, p=1)] for h in hist])
            steps = np.array([h["step"] for h in hist])
            best_swds.append(float(swds.min()))
            dims_data.append(d)
            below = np.where(swds < 0.10)[0]
            if len(below) > 0:
                conv_steps.append(int(steps[below[0]]) // 1000)
                dims_conv.append(d)

        if dims_data:
            ax1.plot(
                dims_data,
                best_swds,
                f"{sigma_markers[key]}-",
                color=SIGMA_COLORS[key],
                label=SIGMA_LABELS[key],
                linewidth=1.5,
                markersize=6,
            )
        if dims_conv:
            ax2.plot(
                dims_conv,
                conv_steps,
                f"{sigma_markers[key]}-",
                color=SIGMA_COLORS[key],
                label=SIGMA_LABELS[key],
                linewidth=1.5,
                markersize=6,
            )

    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("Best SWD")
    ax1.set_title("(a) Peak Quality vs Dimension", fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(DGMM_DIMS)
    ax1.set_xticklabels([str(d) for d in DGMM_DIMS])
    ax1.legend(fontsize=7, loc="upper right")

    ax2.set_xlabel("Dimension $d$")
    ax2.set_ylabel("Steps to SWD < 0.10 (k)")
    ax2.set_title("(b) Convergence Speed vs Dimension", fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(DGMM_DIMS)
    ax2.set_xticklabels([str(d) for d in DGMM_DIMS])
    ax2.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_dgmm_scaling.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Main
def main(argv: typing.List[str]) -> None:
    del argv
    FLAGS = flags.FLAGS
    os.makedirs(FLAGS.work_dir, exist_ok=True)

    _apply_style_palette(FLAGS.style)
    print(f"Plotting toy experiment results (style={FLAGS.style})...")
    plot_samples_grid(FLAGS.base_dir, FLAGS.work_dir, FLAGS.seed)
    plot_training_curves(FLAGS.base_dir, FLAGS.work_dir, FLAGS.seed)
    plot_swiss_roll_stability(FLAGS.base_dir, FLAGS.work_dir, FLAGS.seed)
    plot_dgmm_scaling(FLAGS.base_dir, FLAGS.work_dir, FLAGS.seed)
    print("Done.")


if __name__ == "__main__":
    app.run(main=main)
