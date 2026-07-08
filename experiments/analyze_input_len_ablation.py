"""Input-length ablation analysis (Module 8.3).

Post-processes the ISOLATED input-length ablation run (produced by
run_input_len_ablation.py with `--results-dir results_inputlen`) and writes small
summary tables and one figure to `results/sensitivity/` and `results/figures/`. It
reads a separate results directory and never touches the canonical ETTh tables.

Question: does the model ranking (NLinear / DLinear / Transformer on ETTh1, h96)
depend on the default `input_len=96`, or is it stable across 48 / 96 / 192?

Reads  results_inputlen/metrics/experiment_log.csv
Writes results/sensitivity/input_len_ablation.csv
       results/sensitivity/input_len_ablation_summary.csv
       results/figures/input_len_ablation_mae.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

TABLE_COLS = ["model", "input_len", "num_runs", "mean_mae", "std_mae", "mean_rmse",
              "mean_wape", "total_parameters", "rank_within_input_len"]


def read_csv_retry(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    last_exc = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def build_ablation_table(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, input_len), ranked by MAE within each input_len."""
    df = log_df[log_df["model"].astype(str) != "model"].copy()
    for col in ("mae", "rmse", "wape", "total_parameters", "input_len"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mae", "input_len"])

    rows = []
    for (model, input_len), g in df.groupby(["model", "input_len"]):
        rows.append({
            "model": model, "input_len": int(input_len), "num_runs": int(len(g)),
            "mean_mae": float(g["mae"].mean()),
            "std_mae": float(g["mae"].std(ddof=1)) if len(g) > 1 else 0.0,
            "mean_rmse": float(g["rmse"].mean()),
            "mean_wape": float(g["wape"].mean()),
            "total_parameters": int(g["total_parameters"].iloc[0]),
        })
    out = pd.DataFrame(rows)
    out["rank_within_input_len"] = (out.groupby("input_len")["mean_mae"]
                                    .rank(method="min").astype(int))
    return out.sort_values(["input_len", "mean_mae"])[TABLE_COLS].reset_index(drop=True)


def ranking_by_input_len(table: pd.DataFrame) -> dict:
    """Ordered (best-to-worst by MAE) model list per input_len."""
    return {int(il): list(g.sort_values("mean_mae")["model"])
            for il, g in table.groupby("input_len")}


def is_ranking_stable(table: pd.DataFrame) -> bool:
    """True when every input_len yields the identical model ordering."""
    orderings = list(ranking_by_input_len(table).values())
    return all(o == orderings[0] for o in orderings) if orderings else False


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for input_len, order in ranking_by_input_len(table).items():
        best = order[0]
        rows.append({
            "input_len": input_len,
            "best_model": best,
            "ranking_best_to_worst": " > ".join(order),
        })
    summary = pd.DataFrame(rows).sort_values("input_len").reset_index(drop=True)
    summary["ranking_stable_across_input_len"] = is_ranking_stable(table)
    return summary


def make_figure(table: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, g in table.groupby("model"):
        g = g.sort_values("input_len")
        yerr = g["std_mae"] if "std_mae" in g else None
        ax.errorbar(g["input_len"], g["mean_mae"], yerr=yerr, marker="o",
                    capsize=4, label=model)
    ax.set_xlabel("Input length (past timesteps)")
    ax.set_ylabel("Mean MAE (original OT scale)")
    ax.set_title("Input-length sensitivity (ETTh1, multivariate, horizon 96)")
    ax.set_xticks(sorted(table["input_len"].unique()))
    ax.legend(title="model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Input-length ablation analysis.")
    parser.add_argument("--ablation-results-dir",
                        default=str(PROJECT_ROOT / "results_inputlen"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "sensitivity"))
    parser.add_argument("--fig-dir", default=str(PROJECT_ROOT / "results" / "figures"))
    args = parser.parse_args()

    log_path = os.path.join(args.ablation_results_dir, "metrics", "experiment_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Ablation experiment log not found: {log_path}")

    log_df = read_csv_retry(log_path)
    table = build_ablation_table(log_df)
    summary = build_summary(table)

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "input_len_ablation.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "input_len_ablation_summary.csv"),
                   index=False)
    make_figure(table, os.path.join(args.fig_dir, "input_len_ablation_mae.png"))

    print(f"Wrote input_len_ablation.csv ({len(table)} rows), summary and figure.")
    print(f"Ranking stable across input_len: {is_ranking_stable(table)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
