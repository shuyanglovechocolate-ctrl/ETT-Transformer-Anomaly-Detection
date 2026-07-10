"""Focused TCN comparison (Module 4.6).

Trains the Temporal Convolutional Network on a small, focused slice of the core
protocol (ETTh1, multivariate, input_len 96, seed 42, horizons 24/48/96) in an
ISOLATED results directory, then places its forecasting MAE next to the models
already recorded in the canonical experiment log. The point is a like-for-like,
equal-budget comparison of a convolutional model against the linear, recurrent
and attention baselines — not a new full matrix.

Guardrail: the TCN runs write to ``results_tcn/`` (git-ignored); the canonical
tables under ``results/`` are never modified. Only the comparison summary and a
figure are written under ``results/comparison/`` and ``results/figures/``.

Reads  results/metrics/experiment_log.csv  (canonical baselines)
Writes results_tcn/...                      (isolated TCN training artefacts)
       results/comparison/tcn_comparison.csv
       results/figures/tcn_comparison_mae.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.run_matrix import build_matrix_config  # noqa: E402
from src.training.experiment import (  # noqa: E402
    build_experiment_id, build_output_paths, run_experiment_from_config,
)
from src.viz import apply_paper_style, color_for_model  # noqa: E402

DATASET = "ETTh1"
INPUT_TYPE = "multivariate"
INPUT_LEN = 96
SEED = 42
HORIZONS = [24, 48, 96]
# Models to place alongside the TCN (must exist in the canonical log).
BASELINE_MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]


def train_tcn(horizons, project_root, results_dir):
    """Train the TCN for each horizon into the isolated results_dir; return rows."""
    rows = []
    for horizon in horizons:
        config = build_matrix_config(DATASET, INPUT_TYPE, INPUT_LEN, horizon,
                                     SEED, "tcn")
        eid = build_experiment_id(config)
        metrics_path = build_output_paths(eid, results_dir)["metrics"]
        if os.path.exists(metrics_path):
            print(f"[{eid}] already trained; reusing metrics.")
        else:
            run_experiment_from_config(config, project_root=project_root,
                                       results_dir=results_dir)
        rows.append({"model": "tcn", "horizon": horizon})
    return rows


def load_tcn_mae(horizons, results_dir):
    """Read the isolated TCN experiment_log and return {horizon: mae}."""
    log_path = os.path.join(results_dir, "metrics", "experiment_log.csv")
    log = pd.read_csv(log_path)
    log = log[log["model"].astype(str) == "tcn"]
    out = {}
    for horizon in horizons:
        sub = log[log["horizon"] == horizon]
        if not sub.empty:
            out[horizon] = float(sub.sort_values("timestamp").iloc[-1]["mae"])
    return out


def load_baseline_mae(project_root):
    """Read canonical experiment_log for the focused slice; DataFrame model x horizon."""
    log_path = os.path.join(project_root, "results", "metrics", "experiment_log.csv")
    log = pd.read_csv(log_path)
    log = log[log["model"].astype(str) != "model"]
    log["mae"] = pd.to_numeric(log["mae"], errors="coerce")
    mask = ((log["dataset"] == DATASET) & (log["input_type"] == INPUT_TYPE)
            & (log["input_len"] == INPUT_LEN) & (log["seed"] == SEED)
            & (log["model"].isin(BASELINE_MODELS)) & (log["horizon"].isin(HORIZONS)))
    sub = log[mask].dropna(subset=["mae"])
    # Mean over any duplicate runs of the same cell.
    return sub.groupby(["model", "horizon"])["mae"].mean().reset_index()


def build_comparison(baseline_df, tcn_mae):
    rows = baseline_df.to_dict("records")
    for horizon, mae in tcn_mae.items():
        rows.append({"model": "tcn", "horizon": horizon, "mae": mae})
    df = pd.DataFrame(rows)
    # Per-horizon rank (1 = best/lowest MAE) to make the comparison legible.
    df["rank"] = df.groupby("horizon")["mae"].rank(method="min").astype(int)
    return df.sort_values(["horizon", "mae"]).reset_index(drop=True)


def make_figure(df, path):
    apply_paper_style()
    horizons = sorted(df["horizon"].unique())
    models = ["naive", "linear", "nlinear", "dlinear", "lstm", "tcn", "transformer"]
    models = [m for m in models if m in set(df["model"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(models)
    width = 0.8 / n
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("horizon").reindex(horizons)
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(horizons))]
        # Outline the TCN bars so the model of interest stands out.
        is_tcn = model == "tcn"
        ax.bar(xs, sub["mae"].values, width=width, label=model,
               color=color_for_model(model),
               edgecolor="black" if is_tcn else "none",
               linewidth=1.2 if is_tcn else 0.0)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("MAE (original OT scale)")
    ax.set_title(f"Focused TCN comparison ({DATASET}, {INPUT_TYPE}, "
                 f"input_len {INPUT_LEN}, seed {SEED})")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Focused TCN comparison.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--results-dir",
                        default=str(PROJECT_ROOT / "results_tcn"),
                        help="Isolated directory for TCN training artefacts.")
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    args = parser.parse_args()

    train_tcn(args.horizons, args.project_root, args.results_dir)
    tcn_mae = load_tcn_mae(args.horizons, args.results_dir)
    baseline_df = load_baseline_mae(args.project_root)
    df = build_comparison(baseline_df, tcn_mae)

    out_dir = os.path.join(args.project_root, "results", "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "tcn_comparison.csv")
    df.to_csv(out_csv, index=False)

    fig_dir = os.path.join(args.project_root, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    make_figure(df, os.path.join(fig_dir, "tcn_comparison_mae.png"))

    print(f"Wrote tcn_comparison.csv ({len(df)} rows) and tcn_comparison_mae.png")
    for horizon in args.horizons:
        sub = df[df["horizon"] == horizon]
        if sub.empty:
            continue
        tcn_row = sub[sub["model"] == "tcn"]
        rank = int(tcn_row["rank"].iloc[0]) if not tcn_row.empty else None
        best = sub.iloc[0]
        print(f"  h{horizon}: best={best['model']} (MAE {best['mae']:.3f}); "
              f"TCN rank={rank} of {len(sub)}")


if __name__ == "__main__":
    main()
