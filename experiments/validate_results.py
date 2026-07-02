"""Validate forecasting experiment outputs (Module 3.5).

Automates the completeness/consistency checks that were previously done by hand
(e.g. a seed whose metrics JSON existed but never reached experiment_log). Reads
existing outputs only; runs no experiments. Prints a report and writes
results/metrics/result_validation_report.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.run_matrix import build_matrix_configs  # noqa: E402
from src.training.experiment import build_experiment_id  # noqa: E402

EXPECTED_MATRICES = {"core-light": 144, "core-deep": 36, "robustness-deep": 12}
METRIC_COLS = ["mae", "rmse", "wape"]


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").drop_duplicates("experiment_id", keep="last")


def validate(results_dir: str) -> dict:
    """Run all validation checks and return a structured report."""
    metrics_dir = os.path.join(results_dir, "metrics")
    predictions_dir = os.path.join(results_dir, "predictions")
    log_path = os.path.join(metrics_dir, "experiment_log.csv")

    report = {"checks": [], "ok": True}

    def add(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["ok"] = False

    if not os.path.exists(log_path):
        add("experiment_log_exists", False, log_path)
        return report
    add("experiment_log_exists", True)

    raw = pd.read_csv(log_path)
    log = _dedup(raw)

    # Duplicate experiment_id rows in the raw (append-only) log are expected;
    # we only flag if dedup actually removed nothing yet ids repeat (shouldn't).
    add("no_duplicate_ids_after_dedup",
        log["experiment_id"].is_unique,
        f"{len(raw)} raw rows -> {len(log)} unique")

    # Each declared matrix is complete (every expected experiment_id is logged
    # and has a predictions file).
    logged_ids = set(log["experiment_id"])
    for matrix, expected_n in EXPECTED_MATRICES.items():
        expected_ids = {build_experiment_id(c) for c in build_matrix_configs(matrix)}
        missing_log = sorted(expected_ids - logged_ids)
        add(f"{matrix}_complete_in_log", not missing_log,
            f"expected {expected_n}, missing {len(missing_log)}: {missing_log[:5]}")
        missing_pred = [
            eid for eid in expected_ids
            if not os.path.exists(os.path.join(predictions_dir, f"{eid}_predictions.csv"))
        ]
        add(f"{matrix}_predictions_exist", not missing_pred,
            f"missing {len(missing_pred)} prediction files: {missing_pred[:5]}")

    # Every (dataset, model, input_type, horizon) group has 3 seeds.
    grp = log.groupby(["dataset", "model", "input_type", "horizon"])["seed"].nunique()
    bad_seed_counts = grp[grp != 3]
    add("all_groups_have_3_seeds", bad_seed_counts.empty,
        f"{len(bad_seed_counts)} groups with !=3 seeds: "
        f"{bad_seed_counts.head().to_dict()}")

    # Metrics are present, non-NaN and positive.
    metric_na = log[METRIC_COLS].isna().any().any()
    add("no_nan_metrics", not metric_na)
    nonpos = (log[METRIC_COLS] <= 0).any().any()
    add("metrics_positive", not nonpos)

    # Summary tables (if present) have sensible num_runs.
    mc_path = os.path.join(metrics_dir, "model_comparison.csv")
    if os.path.exists(mc_path):
        mc = pd.read_csv(mc_path)
        add("model_comparison_num_runs_ok",
            bool((mc["num_runs"] >= 1).all()),
            f"num_runs range {int(mc['num_runs'].min())}-{int(mc['num_runs'].max())}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate experiment results.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    report = validate(args.results_dir)

    print("Result validation:", "PASS" if report["ok"] else "FAIL")
    for c in report["checks"]:
        flag = "ok " if c["ok"] else "XX "
        print(f"  [{flag}] {c['check']}" + (f" — {c['detail']}" if c["detail"] else ""))

    out = os.path.join(args.results_dir, "metrics", "result_validation_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")

    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
