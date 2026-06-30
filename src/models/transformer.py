"""Encoder-only Transformer forecaster with positional encoding.

Flow:
    input projection (num_features -> d_model)
    + sinusoidal positional encoding
    -> stack of self-attention encoder layers
    -> pooling (last token or mean)
    -> linear head (d_model -> horizon)

A custom encoder layer is used so that attention weights can be returned
(``return_attention=True``) for the exploratory RQ4 analysis. Attention is an
exploratory cue about which historical steps the model attends to, not a causal
explanation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


class _EncoderLayer(nn.Module):
    """Post-norm self-attention encoder layer that can return attention."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention: bool = False):
        attn_out, attn_weights = self.self_attn(
            x, x, x, need_weights=return_attention, average_attn_weights=True
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x, attn_weights


class TransformerForecaster(BaseForecaster):
    """Encoder-only Transformer for multi-horizon OT forecasting.

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    d_model : int
        Embedding size. Must be divisible by nhead.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        Feed-forward hidden size.
    dropout : float
        Dropout probability.
    pooling : str
        "last" (use the final time step) or "mean" (average over time).
    """

    model_type = "attention"
    supports_attention = True
    description = "Encoder-only Transformer with positional encoding and pooling head."

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__(input_len, num_features, horizon)

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got '{pooling}'.")

        self.pooling = pooling
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(input_len, 1))
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x, return_attention: bool = False):
        # x: [batch, input_len, num_features]
        h = self.input_proj(x)          # [batch, input_len, d_model]
        h = self.pos_encoder(h)

        attentions = []
        for layer in self.layers:
            h, attn = layer(h, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)

        if self.pooling == "last":
            pooled = h[:, -1, :]        # [batch, d_model]
        else:
            pooled = h.mean(dim=1)      # [batch, d_model]

        out = self.head(pooled)         # [batch, horizon]

        if return_attention:
            return out, attentions
        return out
