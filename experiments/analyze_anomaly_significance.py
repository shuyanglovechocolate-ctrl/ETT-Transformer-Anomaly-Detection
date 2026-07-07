"""Anomaly-detection significance analysis (Module 7.4).

Pure post-hoc analysis of existing threshold-free results (no model or detector
re-runs). Answers a single question with the same statistical rigour used for the
forecasting comparison (Module 3.5): is the residual detector *significantly* better
than the causal statistical baselines?

For every anomaly type it compares the residual detector against each baseline
(raw_zscore, diff_score, rolling_zscore) on two threshold-free metrics:

- pr_auc          (PR-AUC / average precision; primary, robust to class imbalance)
- oracle_best_f1  (best F1 over all thresholds; sourced from the `best_f1` column)

Comparisons are strictly *paired*: only matched
`dataset x model x horizon x aggregation_method x anomaly_type x injection_seed`
units for which the residual detector and all baselines are present are kept
(complete-case inner join), so a difference reflects the detector, not differing
experimental conditions. Each paired difference (residual - baseline, higher is
better) is summarised with a paired bootstrap 95% CI and a one-sided Wilcoxon
signed-rank test (residual > baseline). A comparison is significant when the
bootstrap CI lower bound is above zero.

Reads  results/anomaly/metrics/anomaly_threshold_free_results.csv
Writes results/anomaly/metrics/anomaly_significance_tests.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RESIDUAL_DETECTOR = "residual"
BASELINE_DETECTORS = ["raw_zscore", "diff_score", "rolling_zscore"]
# Output metric label -> source column in the threshold-free results table.
METRICS = {"pr_auc": "pr_auc", "oracle_best_f1": "best_f1"}
PAIR_KEY = ["dataset", "model", "horizon", "aggregation_method",
            "anomaly_type", "injection_seed"]


def read_csv_retry(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    """Read a CSV, retrying on transient sandbox filesystem errors."""
    last_exc = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Matched (paired) units
# ---------------------------------------------------------------------------

def matched_metric_frame(df: pd.DataFrame, metric_col: str,
                         detectors) -> pd.DataFrame:
    """Wide frame of one metric per detector, keeping only complete-case units.

    Returns a dataframe indexed by PAIR_KEY with one column per detector; rows
    where any requested detector is missing or has a NaN metric are dropped, so
    every remaining row is a fully matched paired unit.
    """
    detectors = list(detectors)
    sub = df[df["detector_type"].isin(detectors)]
    wide = sub.pivot_table(index=PAIR_KEY, columns="detector_type",
                           values=metric_col, aggfunc="first", dropna=False)
    # A detector that is entirely absent (or all-NaN) leaves no column; add it
    # back as NaN so the complete-case filter drops those units uniformly.
    for det in detectors:
        if det not in wide.columns:
            wide[det] = np.nan
    wide = wide[detectors].dropna(subset=detectors)
    return wide


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42,
                        alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for the mean of `diffs` (residual - baseline).

    Higher metric is better, so a positive mean_diff favours the residual
    detector. The comparison is 'significant' when the CI lower bound is > 0.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "significant": False,
                "win_rate": float("nan"), "num_pairs": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0),
        "win_rate": float(np.mean(diffs > 0)),
        "num_pairs": int(n),
    }


def wilcoxon_greater_pvalue(diffs: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p-value for (residual > baseline).

    Returns NaN when the test is undefined (all-zero differences) or scipy is
    unavailable, so the script degrades gracefully to bootstrap + win-rate only.
    """
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_significance(df: pd.DataFrame, n_boot: int = 10000,
                         seed: int = 42) -> pd.DataFrame:
    """Paired residual-vs-baseline significance per anomaly type and metric."""
    rows = []
    anomaly_types = sorted(df["anomaly_type"].unique())
    for metric_label, metric_col in METRICS.items():
        for baseline in BASELINE_DETECTORS:
            wide = matched_metric_frame(df, metric_col,
                                        [RESIDUAL_DETECTOR, baseline])
            for anomaly_type in anomaly_types:
                cell = wide.xs(anomaly_type, level="anomaly_type") \
                    if anomaly_type in wide.index.get_level_values("anomaly_type") \
                    else wide.iloc[0:0]
                res_vals = cell[RESIDUAL_DETECTOR].to_numpy()
                base_vals = cell[baseline].to_numpy()
                diffs = res_vals - base_vals
                boot = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
                rows.append({
                    "anomaly_type": anomaly_type,
                    "baseline_detector": baseline,
                    "metric": metric_label,
                    "num_pairs": boot["num_pairs"],
                    "mean_residual": float(res_vals.mean()) if len(res_vals) else float("nan"),
                    "mean_baseline": float(base_vals.mean()) if len(base_vals) else float("nan"),
                    "mean_diff": boot["mean_diff"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "significant": boot["significant"],
                    "wilcoxon_pvalue": wilcoxon_greater_pvalue(diffs),
                    "win_rate": boot["win_rate"],
                    "n_boot": n_boot,
                    "bootstrap_seed": seed,
                })
    cols = ["anomaly_type", "baseline_detector", "metric", "num_pairs",
            "mean_residual", "mean_baseline", "mean_diff", "ci_low", "ci_high",
            "significant", "wilcoxon_pvalue", "win_rate", "n_boot",
            "bootstrap_seed"]
    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance tests for anomaly detectors.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, "anomaly", "metrics")
    in_path = os.path.join(metrics_dir, "anomaly_threshold_free_results.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Threshold-free results not found: {in_path}")

    df = read_csv_retry(in_path)
    sig = compute_significance(df, n_boot=args.n_boot, seed=args.seed)
    out_path = os.path.join(metrics_dir, "anomaly_significance_tests.csv")
    sig.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(sig)} rows)")


if __name__ == "__main__":
    main()
