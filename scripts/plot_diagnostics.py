"""Generate paper figures from diagnostic experiment results.

Usage:
    python scripts/plot_diagnostics.py \
        --diagnostic_results docs/generative/vamf/results/diagnostic_results.json \
        --wandb_metrics docs/generative/vamf/results/wandb_metrics.json \
        --output_dir docs/generative/vamf/assets
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
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

COLORS = {
    "stochastic": "#d62728",
    "deterministic": "#2ca02c",
    "ratio": "#1f77b4",
    "loss": "#1f77b4",
    "grad": "#ff7f0e",
    "fid": "#2ca02c",
    "mf_v0": "#1f77b4",
    "mf_v1": "#ff7f0e",
    "baseline": "#7f7f7f",
}


def plot_variance_amplification(data, output_dir):
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
    path = os.path.join(output_dir, "exp1_variance_amplification.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_curvature_gap(data, output_dir):
    """Figure: Exp 2 — Curvature gap vs (t-r)."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    cmap = plt.cm.viridis
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
    path = os.path.join(output_dir, "exp2_curvature_gap.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_jacobian_norm(data, output_dir):
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
    path = os.path.join(output_dir, "exp4_jacobian_norm.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_training_curves(wandb_data, output_dir):
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
    path = os.path.join(output_dir, "exp3_training_curves.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic_results",
        type=str,
        default="docs/generative/vamf/results/diagnostic_results.json",
    )
    parser.add_argument(
        "--wandb_metrics",
        type=str,
        default="docs/generative/vamf/results/wandb_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/generative/vamf/assets",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot wandb training curves (Exp 3) — always available
    if os.path.exists(args.wandb_metrics):
        print("Plotting Experiment 3 (training curves)...")
        with open(args.wandb_metrics) as f:
            wandb_data = json.load(f)
        plot_training_curves(wandb_data, args.output_dir)
    else:
        print(f"Warning: {args.wandb_metrics} not found")

    # Plot diagnostic experiments (Exp 1, 2, 4)
    if os.path.exists(args.diagnostic_results):
        print("Plotting diagnostic experiments...")
        with open(args.diagnostic_results) as f:
            diag_data = json.load(f)

        if "exp1_variance_amplification" in diag_data:
            print("  Experiment 1 (variance amplification)...")
            plot_variance_amplification(
                diag_data["exp1_variance_amplification"],
                args.output_dir,
            )

        if "exp2_curvature_gap" in diag_data:
            print("  Experiment 2 (curvature gap)...")
            plot_curvature_gap(
                diag_data["exp2_curvature_gap"],
                args.output_dir,
            )

        if "exp4_jacobian_norm" in diag_data:
            print("  Experiment 4 (Jacobian norm)...")
            plot_jacobian_norm(
                diag_data["exp4_jacobian_norm"],
                args.output_dir,
            )
    else:
        print(
            f"Warning: {args.diagnostic_results} not found — "
            "skipping diagnostic plots"
        )

    print("Done.")


if __name__ == "__main__":
    main()
