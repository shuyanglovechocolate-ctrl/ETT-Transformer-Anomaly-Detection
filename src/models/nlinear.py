"""NLinear-style forecaster.

Inspired by NLinear (Zeng et al., AAAI 2023): subtract the last observed value
from the window, predict the change with a linear layer, then add the last value
back. The simple normalization makes it a surprisingly strong baseline on
distribution-shifting series.

This is a variant-A (flattened) adaptation for single-target OT forecasting: the
last value of every channel is subtracted, a linear layer maps the flattened
normalized window to the horizon, and the last OT value is added back.
"""

from typing import List

from src.models.base import BaseForecaster

import torch.nn as nn


class NLinearForecaster(BaseForecaster):
    """Subtract-last / linear / add-back-last forecaster."""

    model_type = "linear"
    requires_feature_cols = True
    description = "NLinear-inspired linear baseline with last-value normalization."

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        feature_cols: List[str],
        target_col: str = "OT",
    ):
        super().__init__(input_len, num_features, horizon)
        if target_col not in feature_cols:
            raise ValueError(
                f"target_col '{target_col}' not found in feature_cols {feature_cols}."
            )
        self.ot_index = feature_cols.index(target_col)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_len * num_features, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        last = x[:, -1:, :]                       # [batch, 1, num_features]
        normalized = x - last                      # subtract last value per channel
        delta = self.linear(self.flatten(normalized))  # [batch, horizon]
        last_ot = x[:, -1, self.ot_index].unsqueeze(1)  # [batch, 1]
        return delta + last_ot                     # [batch, horizon]
