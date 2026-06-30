"""Direct linear forecaster.

A single linear layer that maps the flattened historical window straight to the
multi-horizon forecast. No decomposition; this is the plain linear baseline that
sits between the Naive persistence model and the decomposition-based DLinear.
"""

from src.models.base import BaseForecaster

import torch.nn as nn


class LinearForecaster(BaseForecaster):
    """Flatten [B, L, F] -> Linear -> [B, horizon]."""

    def __init__(self, input_len: int, num_features: int, horizon: int):
        super().__init__(input_len, num_features, horizon)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_len * num_features, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        flat = self.flatten(x)        # [batch, input_len * num_features]
        return self.linear(flat)      # [batch, horizon]
