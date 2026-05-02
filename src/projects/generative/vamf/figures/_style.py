"""Shared style + palette switcher for VaMF paper figures.

Currently, there are two supported styles:

  - ``paper`` (default): light background, serif typography, NeurIPS-class
    sizing. Use this for the camera-ready PDF.
  - ``slides``: dark background, sans-serif, larger typography, vibrant
    palette. Use this for talks / Twitter / blog embeds.
"""

import os
import typing

from matplotlib import pyplot as plt

STYLES = ("paper", "slides")
DEFAULT_STYLE = "paper"

_PARENT = os.path.dirname(os.path.abspath(__file__))
_STYLE_DIR = os.path.join(_PARENT, "constants")


def style_path(name: str) -> str:
    r"""Absolute path to the ``.mplstyle`` file for the named style."""
    if name not in STYLES:
        raise ValueError(f"Unknown style {name!r}; expected one of {STYLES}.")
    return os.path.join(_STYLE_DIR, f"{name}.mplstyle")


def apply_style(name: str = DEFAULT_STYLE) -> None:
    r"""Activate the named matplotlib style."""
    plt.rcParams.update(
        {"text.usetex": True, "text.latex.preamble": r"\usepackage{bm}"}
    )
    plt.style.use(style_path(name))


# Per-style palettes for color references that aren't picked from
# axes.prop_cycle (e.g. fixed roles like "reference scatter" or "VaMF").
# Keys are role names; values are hex strings tuned to the active background.
_PALETTES: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "paper": {
        # --- panel-(a) "phenomenon" / panel-(b) "mechanism" lines ---
        "ratio": "#1f77b4",
        "trace": "#9467bd",
        # --- shared neutrals ---
        "ref": "#888888",
        "baseline": "#7f7f7f",
        # --- illustration roles (conditional-marginal gap figure) ---
        "latent": "#a7d9d5",
        "data": "#f3c6a0",
        "path": "#777777",
        "state": "#ff7f0e",
        "top_curve": "#2ca02c",
        "heatmap_cmap": "inferno",
        # --- methods (must align with plot_toy.METHOD_LABELS keys) ---
        "meanflow": "#1f77b4",
        "vamf_l2": "#ff7f0e",
        "vamf_tw": "#2ca02c",
        # --- aliases for convenience (plot_teaser uses short names) ---
        "mf": "#1f77b4",
        "vamf": "#2ca02c",
        # --- sigma_t schedules (plot_toy SIGMA_COLORS) ---
        "sigma_none": "#d62728",
        "sigma_t2": "#2ca02c",
        "sigma_learned": "#9467bd",
        # --- diagnostic curves (plot_diagnostics.COLORS) ---
        "stochastic": "#d62728",
        "deterministic": "#2ca02c",
        "loss": "#1f77b4",
        "grad": "#ff7f0e",
        "fid": "#2ca02c",
        "mf_v0": "#1f77b4",
        "mf_v1": "#ff7f0e",
        # --- box overlay (used inside sample panels for method + metric) ---
        "bbox_face": "#ffffff",
        "bbox_text": "#000000",
        "bbox_alpha": 0.85,
    },
    "slides": {
        "ratio": "#4cc9f0",
        "trace": "#b388eb",
        "ref": "#666666",
        "baseline": "#aaaaaa",
        # --- illustration roles ---
        "latent": "#a8d3e0",
        "data": "#f1c6d1",
        "path": "#9d9795",
        "state": "#f4e0b0",
        "top_curve": "#c9e5c6",
        "heatmap_cmap": "magma",
        # --- methods (for plot_toy) ---
        "meanflow": "#4cc9f0",
        "vamf_l2": "#ffb627",
        "vamf_tw": "#7ed957",
        # --- aliases for convenience ---
        "mf": "#4cc9f0",
        "vamf": "#7ed957",
        "sigma_none": "#f72585",
        "sigma_t2": "#7ed957",
        "sigma_learned": "#b388eb",
        "stochastic": "#f72585",
        "deterministic": "#7ed957",
        "loss": "#4cc9f0",
        "grad": "#ffb627",
        "fid": "#7ed957",
        "mf_v0": "#4cc9f0",
        "mf_v1": "#ffb627",
        "bbox_face": "#0d1117",
        "bbox_text": "#ffffff",
        "bbox_alpha": 0.75,
    },
}


def palette(name: str = DEFAULT_STYLE) -> typing.Dict[str, str]:
    """Return the role-keyed color palette for the given style."""
    if name not in _PALETTES:
        raise ValueError(f"Unknown style {name!r}; expected one of {STYLES}.")
    return _PALETTES[name]
