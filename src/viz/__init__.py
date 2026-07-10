"""Shared visualisation helpers for thesis figures.

Exposes a single publication-quality matplotlib style so that every figure in
the repository shares consistent fonts, sizes, colours and spacing.
"""

from src.viz.style import (
    MODEL_COLORS,
    PALETTE,
    apply_paper_style,
    color_for_model,
)

__all__ = [
    "MODEL_COLORS",
    "PALETTE",
    "apply_paper_style",
    "color_for_model",
]
