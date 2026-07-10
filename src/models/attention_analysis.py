"""Aggregation helpers for exploratory Transformer attention analysis (RQ4).

The Transformer forecaster can return per-layer self-attention maps
(``forward(..., return_attention=True)``). These functions turn a stream of
per-batch attention maps into interpretable summaries: the mean attention
matrix, the key distribution feeding the pooled representation, an
attention-by-lag profile, and simple concentration statistics (entropy and
recent-mass).

Attention is treated as an *exploratory cue* about which historical timesteps
the model relies on, not as a causal explanation of its forecasts. All
functions here are pure (NumPy in, NumPy/float out) so they are unit-testable
without a trained model.
"""

from typing import Dict, Iterable, List

import numpy as np


def mean_attention_matrix(batches: Iterable[np.ndarray]) -> np.ndarray:
    """Mean self-attention matrix over all windows.

    Parameters
    ----------
    batches : iterable of arrays, each ``[batch, L, L]``
        Head-averaged attention maps (rows = query positions, cols = key
        positions). Rows are assumed to be (softmax) probability distributions.

    Returns
    -------
    np.ndarray, shape ``[L, L]``
        Attention averaged over every window in every batch.
    """
    total = None
    count = 0
    for b in batches:
        b = np.asarray(b, dtype=float)
        if b.ndim != 3 or b.shape[1] != b.shape[2]:
            raise ValueError(f"Expected [batch, L, L] maps, got shape {b.shape}.")
        summed = b.sum(axis=0)
        total = summed if total is None else total + summed
        count += b.shape[0]
    if count == 0:
        raise ValueError("No attention maps were provided.")
    return total / count


def pooled_key_distribution(mean_matrix: np.ndarray, pooling: str = "last") -> np.ndarray:
    """Attention distribution over key positions that feeds the pooled state.

    With ``pooling="last"`` the forecasting head reads the final time step, so
    the relevant distribution is that query's row. With ``pooling="mean"`` every
    query contributes equally, so the column-mean (mean over query rows) is used.
    """
    m = np.asarray(mean_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Expected a square [L, L] matrix, got shape {m.shape}.")
    if pooling == "last":
        return m[-1].copy()
    if pooling == "mean":
        return m.mean(axis=0)
    raise ValueError(f"pooling must be 'last' or 'mean', got '{pooling}'.")


def attention_by_lag(key_distribution: np.ndarray) -> np.ndarray:
    """Re-index a key-position distribution by lag (0 = most recent step).

    Key positions run oldest→newest (index 0 is the oldest input step); lag runs
    newest→oldest, so ``attention_by_lag(dist)[0]`` is the weight on the most
    recent input step.
    """
    dist = np.asarray(key_distribution, dtype=float)
    if dist.ndim != 1:
        raise ValueError("key_distribution must be 1-D.")
    return dist[::-1].copy()


def attention_entropy(key_distribution: np.ndarray) -> float:
    """Shannon entropy (nats) of a key distribution.

    High entropy → attention spread across many timesteps; low entropy →
    concentrated on a few. The distribution is renormalised defensively.
    """
    p = np.asarray(key_distribution, dtype=float)
    total = p.sum()
    if total <= 0:
        raise ValueError("key_distribution must have positive mass.")
    p = np.clip(p / total, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def recent_mass(key_distribution: np.ndarray, k: int) -> float:
    """Fraction of attention on the ``k`` most recent key positions."""
    p = np.asarray(key_distribution, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    total = p.sum()
    if total <= 0:
        raise ValueError("key_distribution must have positive mass.")
    p = p / total
    return float(p[-k:].sum())


def summarize_attention(
    mean_matrices: List[np.ndarray],
    pooling: str = "last",
    recent_k: int = 8,
) -> List[Dict[str, float]]:
    """Per-layer summary rows for the pooled key distribution.

    Returns one dict per layer with the peak lag, entropy, recent-mass and the
    number of input steps, ready to be written to a CSV.
    """
    rows = []
    for i, m in enumerate(mean_matrices):
        dist = pooled_key_distribution(m, pooling=pooling)
        lag = attention_by_lag(dist)
        input_len = int(dist.shape[0])
        rows.append({
            "layer": i,
            "input_len": input_len,
            "pooling": pooling,
            "peak_lag": int(np.argmax(lag)),
            "peak_attention": float(lag.max()),
            "entropy_nats": attention_entropy(dist),
            "max_entropy_nats": float(np.log(input_len)),
            f"recent_{recent_k}_mass": recent_mass(dist, min(recent_k, input_len)),
        })
    return rows
