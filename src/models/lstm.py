"""LSTM forecaster.

Classic recurrent baseline: an LSTM encodes the input window, and the last
hidden state is projected to the forecast horizon.
"""

from src.models.base import BaseForecaster

import torch.nn as nn


class LSTMForecaster(BaseForecaster):
    """LSTM -> last hidden state -> linear head -> [B, horizon].

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    hidden_dim : int
        LSTM hidden size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only active when num_layers > 1).
    """

    model_type = "recurrent"
    description = "LSTM encoder with a linear head over the last hidden state."

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(input_len, num_features, horizon)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        output, _ = self.lstm(x)       # output: [batch, input_len, hidden_dim]
        last_hidden = output[:, -1, :]  # [batch, hidden_dim]
        return self.head(last_hidden)   # [batch, horizon]
