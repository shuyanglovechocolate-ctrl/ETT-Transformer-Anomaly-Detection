"""Tests for the shared publication plotting style."""

import matplotlib.pyplot as plt

from src.viz import MODEL_COLORS, PALETTE, apply_paper_style, color_for_model
from src.models.factory import SUPPORTED_MODELS


def test_apply_paper_style_sets_rcparams():
    # Perturb a couple of params, then confirm the style resets them.
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["savefig.dpi"] = 72

    apply_paper_style()

    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["savefig.dpi"] == 200
    assert plt.rcParams["axes.grid"] is True


def test_apply_paper_style_is_idempotent():
    apply_paper_style()
    first = dict(plt.rcParams)
    apply_paper_style()
    # Re-applying must not change anything.
    assert plt.rcParams["savefig.dpi"] == first["savefig.dpi"]
    assert plt.rcParams["font.size"] == first["font.size"]


def test_color_for_model_known_and_unknown():
    assert color_for_model("transformer") == MODEL_COLORS["transformer"]
    # Case-insensitive.
    assert color_for_model("Transformer") == MODEL_COLORS["transformer"]
    # Unknown / empty names get a stable fallback, never a crash.
    fallback = color_for_model("does_not_exist")
    assert fallback == color_for_model(None)
    assert isinstance(fallback, str) and fallback.startswith("#")


def test_every_supported_model_has_a_colour():
    # Keeps the palette in sync with the model registry.
    for name in SUPPORTED_MODELS:
        assert name in MODEL_COLORS, f"missing colour for model '{name}'"


def test_palette_entries_are_hex():
    assert len(PALETTE) >= 6
    for c in PALETTE:
        assert c.startswith("#") and len(c) == 7
