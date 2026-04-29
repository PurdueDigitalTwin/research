"""Shared style + palette switcher for VaMF paper figures.

Two styles are supported:
  - ``paper`` (default): light background, serif typography, NeurIPS-class
    sizing. Use this for the camera-ready PDF.
  - ``slides``: dark background, sans-serif, larger typography, vibrant
    palette. Use this for talks / Twitter / blog embeds.

Each plotting script should:
  1. Add an ``--style`` absl flag (or use ``DEFAULT_STYLE`` for non-flagged
     callers).
  2. Call ``apply_style(name)`` once, before constructing any figure.
  3. Resolve hardcoded colors via ``palette(name)`` instead of inlining hex
     strings, so the same figure code renders well on either background.
"""

import os
import typing

from matplotlib import pyplot as plt


STYLES = ("paper", "slides")
DEFAULT_STYLE = "paper"

_HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE_DIR = os.path.join(_HERE, "constants")


def style_path(name: str) -> str:
    """Absolute path to the ``.mplstyle`` file for the named style."""
    if name not in STYLES:
        raise ValueError(
            f"Unknown style {name!r}; expected one of {STYLES}."
        )
    return os.path.join(_STYLE_DIR, f"{name}.mplstyle")


def apply_style(name: str = DEFAULT_STYLE) -> None:
    """Activate the named matplotlib style."""
    plt.style.use(style_path(name))


# Per-style palettes for color references that aren't picked from
# axes.prop_cycle (e.g. fixed roles like "reference scatter" or "VaMF").
# Keys are role names; values are hex strings tuned to the active background.
_PALETTES: typing.Dict[str, typing.Dict[str, str]] = {
    "paper": {
        "ratio":   "#1f77b4",
        "trace":   "#9467bd",
        "ref":     "#888888",
        "mf":      "#1f77b4",
        "vamf":    "#2ca02c",
        # Box overlay (used inside sample panels to show method + metric).
        "bbox_face":  "#ffffff",
        "bbox_text":  "#000000",
        "bbox_alpha": 0.85,
    },
    "slides": {
        "ratio":   "#4cc9f0",
        "trace":   "#b388eb",
        "ref":     "#666666",
        "mf":      "#4cc9f0",
        "vamf":    "#7ed957",
        "bbox_face":  "#0d1117",
        "bbox_text":  "#ffffff",
        "bbox_alpha": 0.75,
    },
}


def palette(name: str = DEFAULT_STYLE) -> typing.Dict[str, str]:
    """Return the role-keyed color palette for the given style."""
    if name not in _PALETTES:
        raise ValueError(
            f"Unknown style {name!r}; expected one of {STYLES}."
        )
    return _PALETTES[name]
