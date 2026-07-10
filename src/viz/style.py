"""Publication-quality matplotlib style shared by all thesis figures.

The goal is a single source of truth for figure aesthetics so that EDA,
forecasting, efficiency and anomaly-detection plots look like they belong to
the same dissertation: consistent font sizes, a colour-blind-safe palette,
light gridlines, no top/right spines, and a uniform export DPI.

Usage
-----
    from src.viz import apply_paper_style, color_for_model

    apply_paper_style()          # call once before creating figures
    ...
    ax.plot(x, y, color=color_for_model("transformer"))

``apply_paper_style`` only mutates ``matplotlib.rcParams`` (global state), so
it is safe to call repeatedly and cheap to import.
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless saving
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe qualitative palette (black moved to the end).
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Stable per-model colours so a model keeps its colour across every figure.
# Ordered from simplest baseline to most complex model.
MODEL_COLORS = {
    "naive": "#999999",       # neutral grey — the reference baseline
    "linear": "#56B4E9",      # sky blue
    "nlinear": "#0072B2",     # blue
    "dlinear": "#009E73",     # green
    "tcn": "#CC79A7",         # reddish purple
    "lstm": "#E69F00",        # orange
    "transformer": "#D55E00",  # vermillion
}

_FALLBACK_COLOR = "#666666"


def color_for_model(name: Optional[str]) -> str:
    """Return the canonical colour for a model name (case-insensitive).

    Falls back to a neutral grey for unknown names so plotting never crashes
    on an unexpected label.
    """
    if not name:
        return _FALLBACK_COLOR
    return MODEL_COLORS.get(str(name).lower(), _FALLBACK_COLOR)


def apply_paper_style() -> None:
    """Apply the shared publication style to ``matplotlib.rcParams``.

    Idempotent: safe to call before every figure. Sets typography, a
    colour-blind-safe cycle, light gridlines, clean spines and a high export
    DPI suitable for a printed dissertation.
    """
    plt.rcParams.update(
        {
            # --- typography -------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            # --- colour -----------------------------------------------------
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # --- spines & grid ---------------------------------------------
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#b0b0b0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            # --- lines & markers -------------------------------------------
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            # --- legend -----------------------------------------------------
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.fancybox": False,
            # --- export -----------------------------------------------------
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
