"""Multi-panel paper figure for DiT-scale variance + gradient diagnostics.

Loads three sets of probe outputs:

  (1) Variance-amplification probes at multiple training steps for both
      VaMF-L₂ and baseline — used for panel (b), the curvature trajectory.
  (2) Variance-amplification probes at a fixed step (default 40k) for both
      methods — used for panel (a), the t-sweep at convergence.
  (3) Gradient-variance probes at multiple t values for both methods — used
      for panel (c), the NR-vs-t comparison.

The figure has three panels:

  (a) Loss-variance amplification ratio vs t (both methods, step 40k).
      Shows the predicted t-dependent shape from Theorem 1.
  (b) Variance amplification ratio at t=0.9 vs training step (both methods).
      Shows VaMF-L₂'s Jacobian grows faster — empirical signature of the
      implicit Jacobian regularization missing from β=1.
  (c) Gradient noise ratio (stoch / det) vs t for both methods. Shows that
      the deterministic tangent yields large NR reductions across t≤0.7 but
      marginally loses at t=0.9.

Usage::

    bazelisk run //src/projects/generative/vamf/figures:plot_dit_diagnostics -- \\
        --probe_dir=logs/vamf/dit_probe \\
        --output=docs/generative/vamf/results/dit_diagnostics.pdf
"""

import json
import os
import re
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
    name="probe_dir",
    default=None,
    required=True,
    help=(
        "Directory containing probe_<SIDE>_<STEP>.json (variance probes) "
        "and grad_<SIDE>_t<T>.json (gradient probes)."
    ),
)
flags.DEFINE_integer(
    name="t_sweep_step",
    default=40000,
    help="Training step used for the t-sweep variance plot (panel a).",
)
flags.DEFINE_string(
    name="output",
    default=None,
    required=True,
    help="Output PDF path.",
)


def _load_variance(path: str) -> typing.Dict[float, float]:
    r"""Read a probe_dit_checkpoint JSON and return {t: ratio}."""
    with open(path) as f:
        data = json.load(f)
    results = data["results"]["exp1_variance_amplification"]
    out = {}
    for key in sorted(results.keys()):
        t = float(key.split("=")[1])
        out[t] = results[key]["variance_ratio"]
    return out


def _load_grad(path: str) -> jaxtyping.PyTree:
    r"""Read a grad_var_probe JSON."""
    with open(path) as f:
        return json.load(f)


def _collect_nr_trajectory(probe_dir: str, side: str, t_str: str = "0.5"):
    """Return arrays (steps, NR_ratio at fixed t) sorted by step."""
    steps, ratios = [], []
    pat = re.compile(
        rf"^grad_step_{re.escape(side)}_(\d+)_t{re.escape(t_str)}\.json$"
    )
    for name in sorted(os.listdir(probe_dir)):
        m = pat.match(name)
        if not m:
            continue
        step = int(m.group(1))
        d = _load_grad(os.path.join(probe_dir, name))
        steps.append(step)
        ratios.append(d["ratios"]["NR_ratio"])
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(ratios)[order]


def _collect_nr_sweep(probe_dir: str, side: str):
    """Return arrays (t, NR_stoch, NR_det) sorted by t."""
    ts, nr_s, nr_d = [], [], []
    pat = re.compile(rf"^grad_{re.escape(side)}_t(\d+(?:\.\d+)?)\.json$")
    for name in sorted(os.listdir(probe_dir)):
        m = pat.match(name)
        if not m:
            continue
        t = float(m.group(1))
        d = _load_grad(os.path.join(probe_dir, name))
        ts.append(t)
        nr_s.append(d["stoch"]["NR"])
        nr_d.append(d["det"]["NR"])
    order = np.argsort(ts)
    return (
        np.asarray(ts)[order],
        np.asarray(nr_s)[order],
        np.asarray(nr_d)[order],
    )


def main(argv: typing.List[str]) -> int:
    r"""Render the three-panel DiT diagnostics figure."""
    del argv  # unused arguments
    F = flags.FLAGS

    _style.apply_style(F.style)
    palette = _style.palette(F.style)
    c_vamf = palette["vamf_l2"]
    c_base = palette["meanflow"]

    # ----- panel (a): variance ratio vs t at fixed step ---------------------
    var_v = _load_variance(
        os.path.join(F.probe_dir, f"probe_vamf_l2_{F.t_sweep_step}.json")
    )
    var_b = _load_variance(
        os.path.join(F.probe_dir, f"probe_baseline_{F.t_sweep_step}.json")
    )
    t_axis = sorted(var_v.keys())

    # ----- panel (b): NR-ratio trajectory at fixed t=0.5 --------------------
    steps_v, ratio_v = _collect_nr_trajectory(F.probe_dir, "vamf_l2")
    steps_b, ratio_b = _collect_nr_trajectory(F.probe_dir, "baseline")

    # ----- panel (c): NR vs t -----------------------------------------------
    t_grad_v, nr_s_v, nr_d_v = _collect_nr_sweep(F.probe_dir, "vamf_l2")
    t_grad_b, nr_s_b, nr_d_b = _collect_nr_sweep(F.probe_dir, "baseline")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    ax1, ax2, ax3 = axes

    # Panel (a)
    ax1.semilogy(
        t_axis,
        [var_v[t] for t in t_axis],
        marker="o",
        linewidth=2.0,
        markersize=7,
        color=c_vamf,
        label=r"VaMF-L$_2$",
    )
    ax1.semilogy(
        t_axis,
        [var_b[t] for t in t_axis],
        marker="s",
        linewidth=2.0,
        markersize=7,
        color=c_base,
        label="MeanFlow baseline",
    )
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(
        r"$\mathrm{Var}[\ell_{\mathrm{stoch}}]"
        r"\,/\,\mathrm{Var}[\ell_{\mathrm{determ}}]$"
    )
    ax1.set_title(
        rf"(a) Variance amplification (step ${F.t_sweep_step // 1000}$k)"
    )
    ax1.set_xticks(t_axis)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Panel (b)
    ax2.semilogy(
        steps_v / 1000,
        ratio_v,
        marker="o",
        linewidth=2.0,
        markersize=7,
        color=c_vamf,
        label=r"VaMF-L$_2$",
    )
    ax2.semilogy(
        steps_b / 1000,
        ratio_b,
        marker="s",
        linewidth=2.0,
        markersize=7,
        color=c_base,
        label="MeanFlow baseline",
    )
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax2.set_xlabel("training step (thousands)")
    ax2.set_ylabel(
        r"$\mathrm{NR}_{\mathrm{stoch}}/\mathrm{NR}_{\mathrm{det}}$ at $t=0.5$"
    )
    ax2.set_title(r"(b) Gradient-NR reduction trajectory")
    ax2.legend(loc="upper left", frameon=True, fontsize=8)

    # Panel (c)
    ax3.semilogy(
        t_grad_v,
        nr_s_v,
        marker="o",
        linewidth=1.6,
        markersize=6,
        color=c_vamf,
        linestyle="-",
        label=r"VaMF-L$_2$ NR$_{\mathrm{stoch}}$",
    )
    ax3.semilogy(
        t_grad_v,
        nr_d_v,
        marker="o",
        linewidth=1.6,
        markersize=6,
        color=c_vamf,
        linestyle="--",
        label=r"VaMF-L$_2$ NR$_{\mathrm{det}}$",
    )
    ax3.semilogy(
        t_grad_b,
        nr_s_b,
        marker="s",
        linewidth=1.6,
        markersize=6,
        color=c_base,
        linestyle="-",
        label=r"baseline NR$_{\mathrm{stoch}}$",
    )
    ax3.semilogy(
        t_grad_b,
        nr_d_b,
        marker="s",
        linewidth=1.6,
        markersize=6,
        color=c_base,
        linestyle="--",
        label=r"baseline NR$_{\mathrm{det}}$",
    )
    ax3.set_xlabel("$t$")
    ax3.set_ylabel(
        r"$\mathrm{NR}=\mathrm{Tr}(\mathrm{Cov}[g])/\|\mathbb{E}[g]\|^2$"
    )
    ax3.set_title(r"(c) Gradient NR vs $t$ (step $40$k)")
    ax3.set_xticks(t_grad_v)
    ax3.legend(loc="upper left", frameon=True, fontsize=7, ncol=2)

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
