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

__all__ = [
    "aggregate_residuals",
    "load_prediction_file",
    "save_aggregated_residuals",
    "AGGREGATION_METHODS",
    "inject_synthetic_anomalies",
    "ANOMALY_TYPES",
]
