"""Plot toy experiment results for paper figures."""

import json
import os
import typing

from absl import app
from absl import flags
import matplotlib
from matplotlib import axes as mpl_axes
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "text.usetex": False,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ==============================================================================
# Flags
flags.DEFINE_string(
    name="results_dir",
    default=None,
    required=True,
    help="Directory containing .npz and .json result files.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Output directory for figures.",
)
flags.DEFINE_string(
    name="gmm_results_dir",
    default=None,
    help="Directory containing GMM scaling .json files (optional).",
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
    "vamf_tw": "VaMF (TW)",
}
METHOD_COLORS = {
    "meanflow": "#1f77b4",
    "vamf_l2": "#ff7f0e",
    "vamf_tw": "#2ca02c",
}


# ==============================================================================
# Helpers
def _axis_lim(values: np.ndarray, margin: float = 0.15):
    """Compute axis limits with padding."""
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


# ==============================================================================
# Figure 1: Samples grid (datasets x methods)
def plot_samples_grid(
    results_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """4-row by 5-col grid: reference + 4 methods."""
    n_rows = len(DATASETS)
    n_cols = 1 + len(METHODS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.4),
        squeeze=False,
    )

    for row, dataset in enumerate(DATASETS):
        # Load reference from the first method's npz
        ref_npz = os.path.join(
            results_dir,
            f"{dataset}_{METHODS[0]}_{seed}.npz",
        )
        if not os.path.exists(ref_npz):
            for col in range(n_cols):
                ax = axes[row, col]
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
                ax.tick_params(
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
        ref = np.load(ref_npz)["reference"]
        print(
            f"  {dataset}: ref shape={ref.shape}, "
            f"range=[{ref.min():.2f}, {ref.max():.2f}]"
        )
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
            npz_path = os.path.join(
                results_dir,
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
                if row == 0:
                    ax.set_title(
                        METHOD_LABELS[method],
                        fontweight="bold",
                    )
                ax.tick_params(
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False,
                )
                continue
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
                ax.set_title(
                    METHOD_LABELS[method],
                    fontweight="bold",
                )

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_samples.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Figure 2: Training loss curves
def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def plot_training_curves(
    results_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """One subplot per dataset, all methods overlaid."""
    n = len(DATASETS)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(3.5 * n, 3.0),
    )
    if n == 1:
        axes = [axes]

    for idx, dataset in enumerate(DATASETS):
        ax: mpl_axes.Axes = axes[idx]  # type: ignore
        key = "raw_loss"
        for method in METHODS:
            json_path = os.path.join(
                results_dir,
                f"{dataset}_{method}_{seed}.json",
            )
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                result = json.load(f)
            hist = result["history"]
            steps = np.array([h["step"] for h in hist]) / 1000
            # prefer SWD if available, fallback to raw_loss
            key = "swd" if "swd" in hist[0] else "raw_loss"
            values = np.array([h[key] for h in hist])
            smoothed = _smooth(values, window=7)

            # Raw values as faint line
            ax.plot(
                steps,
                values,
                color=METHOD_COLORS[method],
                linewidth=0.4,
                alpha=0.25,
            )
            # Smoothed as main line
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
            ylabel = "Sliced Wasserstein Dist." if key == "swd" else "Raw Loss"
            ax.set_ylabel(ylabel)
        ax.set_title(
            DATASET_LABELS[dataset],
            fontweight="bold",
        )
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_training_curves.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Figure 3: Dimension scaling (GMM)
GMM_DIMS = [2, 4, 8, 16]


def plot_dimension_scaling(
    results_dir: str,
    work_dir: str,
    seed: int,
) -> None:
    """Two-panel figure: best SWD and convergence speed vs dimension."""
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8, 3.2),
    )

    for method in METHODS:
        best_swds = []
        conv_steps = []
        dims_with_data = []
        dims_conv = []

        for d in GMM_DIMS:
            json_path = os.path.join(
                results_dir,
                f"gmm_{d}_{method}_{seed}.json",
            )
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                result = json.load(f)
            hist = result["history"]
            swds = np.array([h["swd"] for h in hist])
            steps = np.array([h["step"] for h in hist])

            best_swds.append(float(swds.min()))
            dims_with_data.append(d)

            below = np.where(swds < 0.10)[0]
            if len(below) > 0:
                conv_steps.append(int(steps[below[0]]) // 1000)
                dims_conv.append(d)

        if not dims_with_data:
            continue

        ax1.plot(
            dims_with_data,
            best_swds,
            "o-",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.5,
            markersize=6,
        )
        if dims_conv:
            ax2.plot(
                dims_conv,
                conv_steps,
                "s-",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.5,
                markersize=6,
            )

    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("Best SWD")
    ax1.set_title("(a) Peak Quality vs Dimension", fontweight="bold")
    ax1.set_xticks(GMM_DIMS)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Dimension $d$")
    ax2.set_ylabel("Steps to SWD < 0.10 (k)")
    ax2.set_title("(b) Convergence Speed vs Dimension", fontweight="bold")
    ax2.set_xticks(GMM_DIMS)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(work_dir, "toy_dimension_scaling.pdf")
    fig.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ==============================================================================
# Main
def main(argv: typing.List[str]) -> None:
    del argv
    FLAGS = flags.FLAGS
    os.makedirs(FLAGS.work_dir, exist_ok=True)

    print("Plotting toy experiment results...")
    plot_samples_grid(
        FLAGS.results_dir,
        FLAGS.work_dir,
        FLAGS.seed,
    )
    plot_training_curves(
        FLAGS.results_dir,
        FLAGS.work_dir,
        FLAGS.seed,
    )
    if FLAGS.gmm_results_dir:
        print("Plotting dimension scaling...")
        plot_dimension_scaling(
            FLAGS.gmm_results_dir,
            FLAGS.work_dir,
            FLAGS.seed,
        )
    print("Done.")


if __name__ == "__main__":
    app.run(main=main)
