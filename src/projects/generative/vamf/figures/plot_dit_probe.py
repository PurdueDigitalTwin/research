"""Paper figure for the DiT-scale variance + gradient probe.

Loads four JSON files (variance-amplification probe and gradient-variance
probe, for both VaMF and baseline DiT-B/4 checkpoints at matched step)
and renders a two-panel figure:

  (a) Per-sample loss-variance amplification ratio vs t, side-by-side for
      VaMF and baseline, with a y=1 reference line.
  (b) Gradient noise-ratio decomposition: Tr(Cov[g]) reduction (left bar
      group) and NR=Tr(Cov[g])/||E[g]||^2 reduction (right bar group),
      both stoch / det. Shows that gradient covariance drops massively
      while NR stays close to 1.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_probe -- \\
        --variance_vamf=/tmp/probe_postfix_vamf_l2_40k.json \\
        --variance_baseline=/tmp/probe_postfix_baseline_40k.json \\
        --grad_vamf=/tmp/grad_var_postfix_vamf_l2_40k.json \\
        --grad_baseline=/tmp/grad_var_postfix_baseline_40k.json \\
        --output=docs/generative/vamf/results/dit_probe_step40k.pdf
"""

import json
import os
import typing

from absl import app
from absl import flags
import jaxtyping
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
    name="variance_vamf",
    default=None,
    required=True,
    help="JSON from probe_dit_checkpoint for VaMF-L2.",
)
flags.DEFINE_string(
    name="variance_baseline",
    default=None,
    required=True,
    help="JSON from probe_dit_checkpoint for baseline.",
)
flags.DEFINE_string(
    name="grad_vamf",
    default=None,
    required=True,
    help="JSON from grad_var_probe for VaMF-L2.",
)
flags.DEFINE_string(
    name="grad_baseline",
    default=None,
    required=True,
    help="JSON from grad_var_probe for baseline.",
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)


def _load_variance(path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    r"""Read a probe_dit_checkpoint JSON and return (t_values, ratios)."""
    with open(path) as f:
        data = json.load(f)
    results = data["results"]["exp1_variance_amplification"]
    t_vals = []
    ratios = []
    for key in sorted(results.keys()):
        t = float(key.split("=")[1])
        t_vals.append(t)
        ratios.append(results[key]["variance_ratio"])
    return np.asarray(t_vals), np.asarray(ratios)


def _load_grad(path: str) -> jaxtyping.PyTree:
    r"""Read a grad_var_probe JSON and return its summary stats."""
    with open(path) as f:
        return json.load(f)


def main(argv: typing.List[str]) -> int:
    """Render the two-panel DiT probe figure."""
    del argv  # unused arguments
    F = flags.FLAGS

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    t_v, ratios_v = _load_variance(F.variance_vamf)
    t_b, ratios_b = _load_variance(F.variance_baseline)
    assert np.allclose(t_v, t_b), "t-grids must match for the comparison."

    grad_v = _load_grad(F.grad_vamf)
    grad_b = _load_grad(F.grad_baseline)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # ----------------------------------------------------------------------
    # Panel (a): variance amplification ratio vs t, both methods overlaid.
    # ----------------------------------------------------------------------
    ax1.semilogy(
        t_v,
        ratios_v,
        marker="o",
        linewidth=2.0,
        markersize=7,
        color=palette["vamf_l2"],
        label=r"VaMF-L$_2$ ($\beta\!=\!1$)",
    )
    ax1.semilogy(
        t_b,
        ratios_b,
        marker="s",
        linewidth=2.0,
        markersize=7,
        color=palette["meanflow"],
        label="MeanFlow baseline ($\\beta\\!=\\!0$ trained)",
    )
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(
        r"$\mathrm{Var}[\ell_{\mathrm{stoch}}]"
        r"\,/\,\mathrm{Var}[\ell_{\mathrm{determ}}]$"
    )
    ax1.set_title(r"(a) Loss-variance amplification (DiT-B/4, step $40$k)")
    ax1.set_xticks(t_v)
    ax1.legend(loc="upper left", frameon=True)

    # ----------------------------------------------------------------------
    # Panel (b): gradient covariance vs gradient NR reduction (stoch/det).
    # Two grouped bars: left group = Tr(Cov[g]) ratio, right group = NR ratio.
    # Within each group, two bars: VaMF-L2 ckpt vs baseline ckpt.
    # ----------------------------------------------------------------------
    metrics = [
        "Tr(Cov[$g$])",
        r"NR $=\!\mathrm{Tr}(\mathrm{Cov})/\|\mathbb{E}[g]\|^2$",
    ]
    vamf_vals = [
        grad_v["ratios"]["trcov_ratio"],
        grad_v["ratios"]["NR_ratio"],
    ]
    base_vals = [
        grad_b["ratios"]["trcov_ratio"],
        grad_b["ratios"]["NR_ratio"],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    bars_v = ax2.bar(
        x - width / 2,
        vamf_vals,
        width,
        color=palette["vamf_l2"],
        label=r"VaMF-L$_2$ ckpt",
    )
    bars_b = ax2.bar(
        x + width / 2,
        base_vals,
        width,
        color=palette["meanflow"],
        label="baseline ckpt",
    )
    ax2.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1.0,
    )
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel("ratio (stochastic / deterministic)")
    ax2.set_title("(b) Gradient noise decomposition")
    ax2.legend(loc="upper right", frameon=True)

    # Annotate each bar with its numeric value.
    for bars, vals in ((bars_v, vamf_vals), (bars_b, base_vals)):
        for bar, val in zip(bars, vals):
            label = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            ax2.annotate(
                label + r"$\times$",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out = F.output
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _logging.rank_zero_info("Figure saved to %s.", out)

    return 0


if __name__ == "__main__":
    app.run(main)
