"""Tests for the methodology pipeline diagram (Module 8.2)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.make_methodology_diagram import (
    MODULES, ANALYSIS, COL_X, save_figure,
)


def test_four_modules_and_columns_aligned():
    assert len(MODULES) == 4
    assert len(COL_X) == 4
    for title, role, steps in MODULES:
        assert title and role and len(steps) >= 1


def test_analysis_layers_reference_valid_columns():
    for col_idx, title, items in ANALYSIS:
        assert 0 <= col_idx < len(COL_X)
        assert title and len(items) >= 1


def test_save_figure_creates_nonempty_png(tmp_path):
    out = tmp_path / "sub" / "methodology_pipeline.png"
    returned = save_figure(str(out))
    assert returned == str(out)
    assert out.exists()
    assert out.stat().st_size > 1000  # a real rendered PNG, not an empty file
