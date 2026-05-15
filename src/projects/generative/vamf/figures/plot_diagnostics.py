"""Generate paper figures from diagnostic experiment results."""

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
# Constants

# NOTE: Color dict is populated from the active style palette in ``main()``
# hence the same plotting code renders both the light (paper) and dark (slides)
# variants without per-call changes.
COLORS: typing.Dict[str, str] = {}
_COLOR_ROLES = (
    "stochastic",
    "deterministic",
    "ratio",
    "loss",
    "grad",
    "fid",
    "mf_v0",
    "mf_v1",
    "baseline",
)


def _apply_style_palette(name: str) -> None:
    """Activate the named matplotlib style and populate the COLORS dict."""
    _style.apply_style(name)
    palette = _style.palette(name)
    COLORS.clear()
    COLORS.update({k: palette[k] for k in _COLOR_ROLES})


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
    name="diagnostic_results",
    default=None,
    required=True,
    help="Path to the JSON file of diagnostic runs.",
)
flags.DEFINE_string(
    name="wandb_metrics",
    default=None,
    required=True,
    help="Path to the JSON file of WandB logged metrics.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Output directory.",
)


def plot_variance_amplification(data, work_dir):
    """Figure: Exp 1 — Variance ratio vs t."""
    t_vals = []
    ratios = []
    var_stoch = []
    var_determ = []

    for key in sorted(data.keys()):
        t = float(key.split("=")[1])
        t_vals.append(t)
        ratios.append(data[key]["variance_ratio"])
        var_stoch.append(data[key]["stochastic_var"])
        var_determ.append(data[key]["deterministic_var"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    # Panel (a): variance ratio
    ax1.semilogy(
        t_vals,
        ratios,
        "o-",
        color=COLORS["ratio"],
        linewidth=2,
        markersize=7,
    )
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(
        r"$\mathrm{Var}[\ell_{\mathrm{stoch}}]"
        r" / \mathrm{Var}[\ell_{\mathrm{determ}}]$"
    )
    ax1.set_title("(a) Variance Amplification Ratio")
    ax1.set_xticks(t_vals)

    # Panel (b): absolute variances
    ax2.semilogy(
        t_vals,
        var_stoch,
        "s-",
        color=COLORS["stochastic"],
        label="Stochastic ($v_{\\mathrm{cond}}$)",
        linewidth=2,
        markersize=6,
    )
    ax2.semilogy(
        t_vals,
        var_determ,
        "^-",
        color=COLORS["deterministic"],
        label=r"Deterministic ($u_\theta(z,t,t)$)",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\mathrm{Var}[\ell]$")
    ax2.set_title("(b) Per-Sample Loss Variance")
    ax2.legend()
    ax2.set_xticks(t_vals)

    plt.tight_layout()
    path = os.path.join(work_dir, "exp1_variance_amplification.pdf")
    fig.savefig(path)
    plt.close()
    _logging.rank_zero_info("Figure saved to %s.", path)


def plot_curvature_gap(data, work_dir):
    """Figure: Exp 2 — Curvature gap vs (t-r)."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    cmap = plt.cm.viridis  # type: ignore
    t_keys = sorted(data.keys())
    for idx, key in enumerate(t_keys):
        t_val = float(key.split("=")[1])
        gaps = data[key]

        t_minus_r = [g["t_minus_r"] for g in gaps]
        gap_sq = [g["gap_sq_mean"] for g in gaps]

        color = cmap(idx / max(len(t_keys) - 1, 1))
        ax.plot(
            t_minus_r,
            gap_sq,
            "o-",
            color=color,
            label=f"$t={t_val}$",
            markersize=4,
            linewidth=1.5,
        )

    ax.set_xlabel("$t - r$")
    ax.set_ylabel(r"$\| u_\theta(z,r,t) - v_{\mathrm{cond}} \|^2$")
    ax.set_title("Curvature Gap vs Interval Length")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(work_dir, "exp2_curvature_gap.pdf")
    fig.savefig(path)
    plt.close()
    _logging.rank_zero_info("Figure saved to %s.", path)


def plot_jacobian_norm(data, work_dir):
    """Figure: Exp 4 — ||J||_F vs t."""
    t_vals = []
    j_norms = []
    j_stds = []

    for key in sorted(data.keys()):
        t = float(key.split("=")[1])
        t_vals.append(t)
        j_norms.append(data[key]["J_norm_mean"])
        j_stds.append(data[key]["J_norm_sq_std"] ** 0.5)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    ax.plot(
        t_vals,
        j_norms,
        "o-",
        color=COLORS["ratio"],
        linewidth=2,
        markersize=7,
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\| J \|_F$")
    ax.set_title(r"Jacobian Factor $\| (t{-}r)\partial_z u_\theta - I \|_F$")
    ax.set_xticks(t_vals)

    plt.tight_layout()
    path = os.path.join(work_dir, "exp4_jacobian_norm.pdf")
    fig.savefig(path)
    plt.close()
    _logging.rank_zero_info("Figuer saved to %s.", path)


def plot_training_curves(wandb_data, work_dir):
    """Figure: Exp 3 — Loss, grad norm, FID multi-panel."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    run_styles = {
        "vanilla_mf_v0": {
            "color": COLORS["mf_v0"],
            "label": "MeanFlow v0",
        },
        "vanilla_mf_v1": {
            "color": COLORS["mf_v1"],
            "label": "MeanFlow v1",
        },
        "old_baseline": {
            "color": COLORS["baseline"],
            "label": "Old baseline",
        },
    }

    # (a) Loss
    ax = axes[0]
    for run_key, style in run_styles.items():
        if run_key not in wandb_data:
            continue
        d = wandb_data[run_key]["loss"]
        if d["steps"]:
            steps = np.array(d["steps"]) / 1000
            ax.plot(
                steps,
                d["values"],
                color=style["color"],
                label=style["label"],
                linewidth=0.8,
                alpha=0.7,
            )
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Loss Curve")
    ax.legend(fontsize=8)

    # (b) Grad norm
    ax = axes[1]
    for run_key, style in run_styles.items():
        if run_key not in wandb_data:
            continue
        d = wandb_data[run_key].get("grad_norm", {})
        if isinstance(d, dict) and d.get("steps"):
            steps = np.array(d["steps"]) / 1000
            ax.plot(
                steps,
                d["values"],
                color=style["color"],
                label=style["label"],
                linewidth=0.8,
                alpha=0.7,
            )
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(b) Gradient Norm")
    ax.legend(fontsize=8)

    # (c) FID
    ax = axes[2]
    for run_key, style in run_styles.items():
        if run_key not in wandb_data:
            continue
        d = wandb_data[run_key]["fid"]
        if d["steps"]:
            steps = np.array(d["steps"]) / 1000
            ax.plot(
                steps,
                d["values"],
                "o-",
                color=style["color"],
                label=style["label"],
                markersize=3,
                linewidth=1.2,
            )
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("FID")
    ax.set_title("(c) FID Score")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(work_dir, "exp3_training_curves.pdf")
    fig.savefig(path)
    plt.close()
    _logging.rank_zero_info("Figure saved to %s.", path)


def main(argv: typing.List[str]):
    del argv  # unused arguments

    os.makedirs(flags.FLAGS.work_dir, exist_ok=True)
    _apply_style_palette(flags.FLAGS.style)
    _logging.rank_zero_info(
        "Plotting with figure style %s.", flags.FLAGS.style
    )

    # Plot wandb training curves (Exp 3) — always available
    if os.path.exists(flags.FLAGS.wandb_metrics):
        _logging.rank_zero_info("Plotting Experiment 3 (training curves)...")
        with open(flags.FLAGS.wandb_metrics) as f:
            wandb_data = json.load(f)
        plot_training_curves(wandb_data, flags.FLAGS.work_dir)
    else:
        _logging.rank_zero_warning(
            "Warning: %s not found", flags.FLAGS.wandb_metrics
        )

    # Plot diagnostic experiments (Exp 1, 2, 4)
    if os.path.exists(flags.FLAGS.diagnostic_results):
        _logging.rank_zero_info("Plotting diagnostic experiments...")
        with open(flags.FLAGS.diagnostic_results) as f:
            diag_data = json.load(f)

        if "exp1_variance_amplification" in diag_data:
            _logging.rank_zero_info("Experiment 1 (variance amplification)...")
            plot_variance_amplification(
                diag_data["exp1_variance_amplification"],
                flags.FLAGS.work_dir,
            )

        if "exp2_curvature_gap" in diag_data:
            _logging.rank_zero_info("Experiment 2 (curvature gap)...")
            plot_curvature_gap(
                diag_data["exp2_curvature_gap"],
                flags.FLAGS.work_dir,
            )

        if "exp4_jacobian_norm" in diag_data:
            _logging.rank_zero_info("Experiment 4 (Jacobian norm)...")
            plot_jacobian_norm(
                diag_data["exp4_jacobian_norm"],
                flags.FLAGS.work_dir,
            )
    else:
        _logging.rank_zero_warning(
            "Warning: %s not found — " "skipping diagnostic plots",
            flags.FLAGS.diagnostic_results,
        )

    _logging.rank_zero_info("All Done!")

    return 0


if __name__ == "__main__":
    app.run(main=main)
