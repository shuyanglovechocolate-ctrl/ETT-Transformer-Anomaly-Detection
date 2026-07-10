"""Temporal Convolutional Network (TCN) forecaster.

A convolutional alternative to the recurrent and attention baselines. Stacked
residual blocks of dilated **causal** 1-D convolutions give an exponentially
growing receptive field with depth, so the model can reach far back over the
input window without recurrence. The representation at the final (most recent)
time step is projected to the forecast horizon.

Flow:
    x [B, L, F] -> transpose to [B, F, L]
    -> N residual blocks of (dilated causal Conv1d -> ReLU -> dropout) x2
       with dilations 1, 2, 4, ... and a 1x1 residual projection
    -> take the last time step [B, C]
    -> linear head [B, horizon]

Causality (no leakage from future steps into the prediction of a given step) is
enforced by left-padding each convolution by ``(kernel_size - 1) * dilation`` and
cropping the extra right-hand outputs (the "chomp" step).
"""

import torch.nn as nn

from src.models.base import BaseForecaster


class _Chomp1d(nn.Module):
    """Remove the ``chomp_size`` right-most steps added by causal padding."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual (identity) connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )
        # 1x1 projection so the residual matches the channel count when needed.
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    """Dilated causal TCN for multi-horizon OT forecasting.

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    num_channels : int
        Number of convolution channels in every residual block.
    num_layers : int
        Number of stacked residual blocks; dilation doubles each block, so the
        receptive field is ``1 + 2 * (kernel_size - 1) * (2**num_layers - 1)``.
    kernel_size : int
        Convolution kernel size (>= 2 for a non-trivial receptive field).
    dropout : float
        Dropout probability inside each block.
    """

    model_type = "convolutional"
    description = "Temporal Convolutional Network with dilated causal convolutions."

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        num_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(input_len, num_features, horizon)
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        blocks = []
        in_ch = num_features
        for i in range(num_layers):
            dilation = 2 ** i
            blocks.append(
                _TemporalBlock(in_ch, num_channels, kernel_size, dilation, dropout)
            )
            in_ch = num_channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(num_channels, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features] -> [batch, num_features, input_len]
        h = x.transpose(1, 2)
        h = self.tcn(h)              # [batch, num_channels, input_len]
        last = h[:, :, -1]           # most recent time step: [batch, num_channels]
        return self.head(last)       # [batch, horizon]
