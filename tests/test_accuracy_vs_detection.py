"""Tests for the accuracy-vs-detection analysis (Module 6.11)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies
from experiments.prepare_multimodel_residuals import (
    experiment_id, build_multimodel_scenarios, MODELS,
)
from experiments.analyze_accuracy_vs_detection import (
    detection_metrics_for, compute_correlation_table,
)


def test_experiment_id_format():
    assert experiment_id("ETTh1", "linear", 24) == \
        "ETTh1_linear_multivariate_len96_h24_seed42"


def test_build_multimodel_scenarios_count():
    scenarios = build_multimodel_scenarios()
    assert len(scenarios) == len(MODELS) * 2 * 2  # 6 models x 2 datasets x 2 horizons
    assert ("ETTh1", "transformer", 96) in scenarios


def _residual_df(n=300, pred_noise=0.2):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y = rng.normal(10.0, 1.0, size=n)
    yp = y + np.random.default_rng(1).normal(0.0, pred_noise, size=n)
    return pd.DataFrame({"target_date": dates, "y_true": y, "y_pred": yp,
                         "residual": y - yp, "abs_residual": np.abs(y - yp),
                         "anomaly_score": np.abs(y - yp)})


def test_same_injection_positions_across_models():
    # Two "models" share identical y_true, so injected labels must match.
    base = _residual_df()
    model_a = base.copy()
    model_b = base.copy()
    model_b["y_pred"] = base["y_pred"] + 0.5  # different predictions
    inj_a = inject_synthetic_anomalies(model_a, "spike", anomaly_ratio=0.03,
                                       duration_range=(1, 3), seed=42)
    inj_b = inject_synthetic_anomalies(model_b, "spike", anomaly_ratio=0.03,
                                       duration_range=(1, 3), seed=42)
    assert np.array_equal(inj_a["is_anomaly"].to_numpy(),
                          inj_b["is_anomaly"].to_numpy())


def test_detection_metrics_for_keys():
    val = _residual_df()
    injected = inject_synthetic_anomalies(val, "spike", anomaly_ratio=0.03,
                                          duration_range=(1, 3), seed=42)
    m = detection_metrics_for(val, injected)
    for key in ("average_precision", "oracle_best_f1", "fixed_f1", "event_recall"):
        assert key in m


def test_compute_correlation_table_detects_relationship():
    # Construct 6 models where lower MAE -> higher PR-AUC (perfect negative rank).
    models = MODELS
    rows = []
    for i, m in enumerate(models):
        rows.append({"dataset": "ETTh1", "horizon": 24, "anomaly_type": "spike",
                     "model": m, "injection_seed": 42,
                     "forecast_mae": float(i), "forecast_rmse": float(i),
                     "forecast_wape": float(i),
                     "average_precision": float(len(models) - i),  # inverse
                     "oracle_best_f1": 0.5, "fixed_f1": 0.5, "event_recall": 0.5})
    corr = compute_correlation_table(pd.DataFrame(rows))
    row = corr[(corr.forecast_metric == "forecast_mae")
               & (corr.detection_metric == "average_precision")].iloc[0]
    assert row["spearman_correlation"] == -1.0  # perfect inverse rank
    assert row["num_models"] == len(models)
