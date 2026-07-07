"""Anomaly detection package (Module 4).

Residual-based synthetic anomaly detection built on Module 3 forecasting outputs.
"""

from src.anomaly.residuals import (
    aggregate_residuals,
    load_prediction_file,
    save_aggregated_residuals,
    AGGREGATION_METHODS,
)
from src.anomaly.injection import inject_synthetic_anomalies, ANOMALY_TYPES
from src.anomaly.thresholds import compute_threshold, validate_scores, THRESHOLD_METHODS
from src.anomaly.detector import detect_anomalies
from src.anomaly.metrics import calculate_detection_metrics
from src.anomaly.baselines import compute_baseline_scores, BASELINE_DETECTORS
from src.anomaly.event_metrics import calculate_event_detection_metrics
from src.anomaly.diagnostics import (
    summarize_score_distribution, compute_causal_rolling_std,
    compute_flatness_score, compare_scores_by_label,
)
from src.anomaly.hybrid import (
    rank_normalize, hybrid_rankmax_score, detect_hybrid_or,
)
from src.anomaly.scoring import score_detector, DETECTOR_TYPES
from src.anomaly.threshold_free import (
    pr_auc, roc_auc, best_f1_threshold, f1_at_target_fpr,
    calculate_threshold_free_metrics,
)

__all__ = [
    "aggregate_residuals",
    "load_prediction_file",
    "save_aggregated_residuals",
    "AGGREGATION_METHODS",
    "inject_synthetic_anomalies",
    "ANOMALY_TYPES",
    "compute_threshold",
    "validate_scores",
    "THRESHOLD_METHODS",
    "detect_anomalies",
    "calculate_detection_metrics",
    "compute_baseline_scores",
    "BASELINE_DETECTORS",
    "calculate_event_detection_metrics",
    "summarize_score_distribution",
    "compute_causal_rolling_std",
    "compute_flatness_score",
    "compare_scores_by_label",
    "score_detector",
    "DETECTOR_TYPES",
    "rank_normalize",
    "hybrid_rankmax_score",
    "detect_hybrid_or",
    "pr_auc",
    "roc_auc",
    "best_f1_threshold",
    "f1_at_target_fpr",
    "calculate_threshold_free_metrics",
]
