"""Model factory: build a forecaster from a config dictionary.

Module 3 only needs:

    model = build_model(config, num_features=data["num_features"],
                        feature_cols=data["feature_cols"])
"""

from typing import Any, Dict, List

from src.models.base import BaseForecaster
from src.models.naive import NaiveForecaster
from src.models.linear import LinearForecaster
from src.models.dlinear import DLinearForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster


def build_model(
    config: Dict[str, Any],
    num_features: int,
    feature_cols: List[str],
) -> BaseForecaster:
    """Build a forecasting model from config and Module 1 metadata.

    Parameters
    ----------
    config : Dict[str, Any]
        Full config; uses config["model"] and config["window"].
    num_features : int
        From Module 1 (1 univariate / 7 multivariate). Never hard-coded.
    feature_cols : List[str]
        Ordered input feature names (needed by the Naive baseline).

    Returns
    -------
    BaseForecaster
        An instantiated model following the standard input-output contract.
    """
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    input_len = config["window"]["input_len"]
    horizon = config["window"]["horizon"]

    if name == "naive":
        return NaiveForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col=config["dataset"].get("target", "OT"),
        )

    if name == "linear":
        return LinearForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
        )

    if name == "dlinear":
        return DLinearForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            kernel_size=model_cfg.get("kernel_size", 25),
        )

    if name == "lstm":
        return LSTMForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )

    if name == "transformer":
        return TransformerForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            dropout=model_cfg.get("dropout", 0.1),
            pooling=model_cfg.get("pooling", "last"),
        )

    raise ValueError(
        f"Unknown model name: '{name}'. "
        "Expected one of: naive, linear, dlinear, lstm, transformer."
    )
