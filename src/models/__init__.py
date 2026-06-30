"""Forecasting model library (Module 2).

All models share the contract:
    x      : [batch_size, input_len, num_features]
    y_pred : [batch_size, horizon]
"""

from src.models.base import BaseForecaster
from src.models.naive import NaiveForecaster
from src.models.dlinear import DLinearForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
from src.models.factory import build_model

__all__ = [
    "BaseForecaster",
    "NaiveForecaster",
    "DLinearForecaster",
    "LSTMForecaster",
    "TransformerForecaster",
    "build_model",
]
