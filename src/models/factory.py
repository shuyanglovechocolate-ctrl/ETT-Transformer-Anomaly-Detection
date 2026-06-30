"""Model factory: build a forecaster from a config dictionary.

Module 3 only needs:

    model = build_model(config, num_features=data["num_features"],
                        feature_cols=data["feature_cols"])
"""

from typing import Any, Dict, List

from src.models.base import BaseForecaster
from src.models.naive import NaiveForecaster
from src.models.linear import LinearForecaster
from src.models.nlinear import NLinearForecaster
from src.models.dlinear import DLinearForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

# Single source of truth for supported model names -> classes.
MODEL_REGISTRY = {
    "naive": NaiveForecaster,
    "linear": LinearForecaster,
    "nlinear": NLinearForecaster,
    "dlinear": DLinearForecaster,
    "lstm": LSTMForecaster,
    "transformer": TransformerForecaster,
}

SUPPORTED_MODELS = list(MODEL_REGISTRY)


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate the model section of a config, raising on invalid values.

    Checks the presence and validity of model.name and the model-specific
    hyper-parameters, so a bad config fails at build time rather than mid-training.
    """
    if "model" not in config:
        raise ValueError("Config must contain a 'model' section.")

    model_cfg = config["model"]
    if "name" not in model_cfg:
        raise ValueError("config['model'] must contain a 'name' field.")

    name = model_cfg["name"].lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model name: {name}. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS)}."
        )

    if name == "dlinear":
        kernel_size = model_cfg.get("kernel_size", 25)
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("dlinear.kernel_size must be a positive integer.")
        if kernel_size % 2 == 0:
            raise ValueError("dlinear.kernel_size must be odd.")
        if not isinstance(model_cfg.get("channel_independent", False), bool):
            raise ValueError("dlinear.channel_independent must be a boolean.")

    if name == "lstm":
        if model_cfg.get("hidden_dim", 64) <= 0:
            raise ValueError("lstm.hidden_dim must be > 0.")
        if model_cfg.get("num_layers", 2) <= 0:
            raise ValueError("lstm.num_layers must be > 0.")
        dropout = model_cfg.get("dropout", 0.2)
        if not 0 <= dropout < 1:
            raise ValueError("lstm.dropout must be in [0, 1).")

    if name == "transformer":
        d_model = model_cfg.get("d_model", 64)
        nhead = model_cfg.get("nhead", 4)
        if d_model <= 0:
            raise ValueError("transformer.d_model must be > 0.")
        if nhead <= 0:
            raise ValueError("transformer.nhead must be > 0.")
        if model_cfg.get("num_layers", 2) <= 0:
            raise ValueError("transformer.num_layers must be > 0.")
        if model_cfg.get("dim_feedforward", 128) <= 0:
            raise ValueError("transformer.dim_feedforward must be > 0.")
        dropout = model_cfg.get("dropout", 0.1)
        if not 0 <= dropout < 1:
            raise ValueError("transformer.dropout must be in [0, 1).")
        if d_model % nhead != 0:
            raise ValueError(
                f"transformer.d_model ({d_model}) must be divisible by "
                f"nhead ({nhead})."
            )
        if model_cfg.get("pooling", "last") not in ("last", "mean"):
            raise ValueError("transformer.pooling must be 'last' or 'mean'.")


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
    validate_model_config(config)

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

    if name == "nlinear":
        return NLinearForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col=config["dataset"].get("target", "OT"),
        )

    if name == "dlinear":
        return DLinearForecaster(
            input_len=input_len,
            num_features=num_features,
            horizon=horizon,
            kernel_size=model_cfg.get("kernel_size", 25),
            channel_independent=model_cfg.get("channel_independent", False),
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
        f"Expected one of: {', '.join(SUPPORTED_MODELS)}."
    )
