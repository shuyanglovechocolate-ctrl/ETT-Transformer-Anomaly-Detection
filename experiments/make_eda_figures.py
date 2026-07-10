"""Regenerate the committed EDA figures with the shared publication style.

The exploratory analysis lives in ``notebooks/01_eda.ipynb``; this script is a
headless, reproducible entry point that renders the four EDA figures that are
tracked in git so they share the same typography and palette as every other
figure in the repository. It reads the raw dataset through the same
``src.data.loader`` helpers the notebook uses, so there is no logic drift.

Reads  data/raw/<dataset>.csv
Writes results/figures/eda_ot_trend.png
       results/figures/eda_feature_timeseries.png
       results/figures/eda_ot_distribution.png
       results/figures/eda_correlation_heatmap.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN, load_ett_dataset
from src.viz import PALETTE, apply_paper_style


def make_eda_figures(df, dataset_name: str, fig_dir: str) -> None:
    """Render the four tracked EDA figures for ``dataset_name`` into ``fig_dir``."""
    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)

    # 1) OT trend over time.
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df[TARGET_COLUMN], color=PALETTE[0], linewidth=1.0)
    plt.title(f"{dataset_name} — Oil Temperature (OT) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Oil Temperature (OT)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_ot_trend.png"))
    plt.close()

    # 2) All input features over time (stacked panels).
    fig, axes = plt.subplots(len(FEATURE_COLUMNS), 1, figsize=(14, 12), sharex=True)
    for ax, feature in zip(axes, FEATURE_COLUMNS):
        ax.plot(df["date"], df[feature], color=PALETTE[0], linewidth=0.9)
        ax.set_ylabel(feature)
        ax.set_title(feature, fontsize=10, loc="left")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{dataset_name} — Input Features Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "eda_feature_timeseries.png"))
    plt.close(fig)

    # 3) OT distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_COLUMN], bins=50, color=PALETTE[0], alpha=0.85,
             edgecolor="white", linewidth=0.4)
    plt.title(f"{dataset_name} — Distribution of Oil Temperature (OT)")
    plt.xlabel("OT")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_ot_distribution.png"))
    plt.close()

    # 4) Feature correlation heatmap (seaborn if available, else matplotlib).
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                    vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    except ImportError:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURE_COLUMNS)))
        ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
        ax.set_yticks(range(len(FEATURE_COLUMNS)))
        ax.set_yticklabels(FEATURE_COLUMNS)
        for i in range(len(FEATURE_COLUMNS)):
            for j in range(len(FEATURE_COLUMNS)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{dataset_name} — Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "eda_correlation_heatmap.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate committed EDA figures.")
    parser.add_argument("--data-path",
                        default=str(PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"))
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--fig-dir", default=str(PROJECT_ROOT / "results" / "figures"))
    args = parser.parse_args()

    df = load_ett_dataset(args.data_path)
    make_eda_figures(df, args.dataset_name, args.fig_dir)
    print(f"Wrote 4 EDA figures for {args.dataset_name} to {args.fig_dir}")


if __name__ == "__main__":
    main()
