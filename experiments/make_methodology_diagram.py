"""Methodology pipeline diagram (Module 8.2).

Generates a single figure that ties the four modules and the analysis / hardening
layers into one view, as a presentation aid for the README and dissertation. Pure
drawing code — it reads no results and trains nothing.

Writes docs/figures/methodology_pipeline.png
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# The four core modules (title, one-line role, key steps).
MODULES = [
    ("Module 1\nData Pipeline",
     "leakage-free inputs",
     ["chronological split", "train-only scaling", "sliding windows"]),
    ("Module 2\nModel Library",
     "one input/output contract",
     ["Naive / Linear", "NLinear / DLinear", "LSTM / Transformer"]),
    ("Module 3\nForecasting",
     "train + evaluate",
     ["experiment matrix", "early stopping", "per-horizon MAE/RMSE"]),
    ("Module 4\nAnomaly Detection",
     "reuse residuals",
     ["synthetic injection", "validation threshold", "point + event metrics"]),
]

# Analysis / hardening layers grouped under the module they extend.
ANALYSIS = [
    (2, "Forecasting analysis",
     ["paired bootstrap significance", "efficiency / complexity (Pareto)",
      "ETTm external validity"]),
    (3, "Anomaly analysis",
     ["threshold-free PR-AUC", "hybrid residual + flatness",
      "accuracy-vs-detection", "detector significance", "extended anomaly types"]),
]

# Column geometry (x centre per module) in a 0..100 canvas.
COL_X = [12.5, 37.5, 62.5, 87.5]


def _box(ax, cx, cy, w, h, title, body_lines, facecolor, title_size=11):
    import matplotlib.patches as mpatches
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.4, edgecolor="#333333", facecolor=facecolor))
    ax.text(cx, y0 + h - 2.2, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold")
    if body_lines:
        ax.text(cx, y0 + h - 6.5, "\n".join(f"• {b}" for b in body_lines),
                ha="center", va="top", fontsize=8.5)


def draw_pipeline(ax):
    import matplotlib.patches as mpatches
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    module_color = "#dce6f2"
    analysis_color = "#e8f0e1"
    data_color = "#f5e6cc"

    # Data source banner.
    _box(ax, 50, 93, 96, 8, "ETT datasets (ETTh1 / ETTh2 core; ETTm1 / ETTm2 external check)",
         [], data_color, title_size=10)

    # Module spine.
    my, mh = 68, 22
    for cx, (title, role, steps) in zip(COL_X, MODULES):
        _box(ax, cx, my, 22, mh, title, steps, module_color)

    # Arrows along the spine (Module 1 -> 2 -> 3 -> 4).
    for a, b in zip(COL_X[:-1], COL_X[1:]):
        ax.annotate("", xy=(b - 11, my), xytext=(a + 11, my),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8))
    # Arrow from data banner into Module 1.
    ax.annotate("", xy=(COL_X[0], my + mh / 2), xytext=(COL_X[0], 89),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    # Analysis layers under Modules 3 and 4 (kept within the 0..100 canvas, no overlap).
    ay, ah, aw = 30, 24, 23
    for col_idx, title, items in ANALYSIS:
        cx = COL_X[col_idx]
        _box(ax, cx, ay, aw, ah, title, items, analysis_color)
        ax.annotate("", xy=(cx, ay + ah / 2), xytext=(cx, my - mh / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#4a7a3a", lw=1.6))

    ax.text(50, 4, "Single leakage-free protocol · multi-seed · committed result snapshots",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")


def save_figure(path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    draw_pipeline(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Render the methodology pipeline diagram.")
    parser.add_argument("--out", default=str(
        PROJECT_ROOT / "docs" / "figures" / "methodology_pipeline.png"))
    args = parser.parse_args()
    save_figure(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
