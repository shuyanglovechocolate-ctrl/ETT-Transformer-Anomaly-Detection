"""Linear baseline (DLinear-style, simple first version).

This first version flattens the input window and maps it directly to the
horizon with a single linear layer. It is intentionally simple and reliable: a
strong linear challenger to the deep models, fast to train across many
horizons / seeds. It is NOT the full trend/seasonal decomposition DLinear; that
can be added later if needed.
"""

from src.models.base import BaseForecaster

import torch.nn as nn


class DLinearForecaster(BaseForecaster):
    """Flatten [B, L, F] -> Linear -> [B, horizon]."""

    def __init__(self, input_len: int, num_features: int, horizon: int):
        super().__init__(input_len, num_features, horizon)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_len * num_features, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        flat = self.flatten(x)        # [batch, input_len * num_features]
        return self.linear(flat)      # [batch, horizon]
