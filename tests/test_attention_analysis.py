"""Tests for the exploratory attention-analysis aggregation helpers."""

import numpy as np
import pytest

from src.models.attention_analysis import (
    attention_by_lag,
    attention_entropy,
    mean_attention_matrix,
    pooled_key_distribution,
    recent_mass,
    summarize_attention,
)


def _uniform_maps(batch, L, n_batches=2):
    """A stream of uniform attention maps: every row sums to 1, all equal."""
    m = np.full((batch, L, L), 1.0 / L)
    return [m for _ in range(n_batches)]


def test_mean_attention_matrix_averages_over_windows():
    b1 = np.zeros((2, 3, 3))
    b1[:, 2, 0] = 1.0  # last query attends fully to oldest key
    b2 = np.zeros((2, 3, 3))
    b2[:, 2, 2] = 1.0  # last query attends fully to newest key
    mean = mean_attention_matrix([b1, b2])
    assert mean.shape == (3, 3)
    # 4 windows total: two put mass on key 0, two on key 2 → 0.5 each.
    assert mean[2, 0] == pytest.approx(0.5)
    assert mean[2, 2] == pytest.approx(0.5)


def test_mean_attention_matrix_rejects_bad_shape_and_empty():
    with pytest.raises(ValueError):
        mean_attention_matrix([np.zeros((2, 3, 4))])
    with pytest.raises(ValueError):
        mean_attention_matrix([])


def test_pooled_key_distribution_last_vs_mean():
    m = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_allclose(pooled_key_distribution(m, "last"), [0.6, 0.4])
    np.testing.assert_allclose(pooled_key_distribution(m, "mean"), [0.4, 0.6])
    with pytest.raises(ValueError):
        pooled_key_distribution(m, "bogus")


def test_attention_by_lag_reverses_position_order():
    # key positions oldest→newest; lag index 0 must be the most recent key.
    dist = np.array([0.1, 0.2, 0.7])  # newest key (pos 2) has 0.7
    lag = attention_by_lag(dist)
    assert lag[0] == pytest.approx(0.7)  # lag 0 = most recent
    assert lag[-1] == pytest.approx(0.1)


def test_attention_entropy_uniform_is_log_L():
    dist = np.full(4, 0.25)
    assert attention_entropy(dist) == pytest.approx(np.log(4))
    # A one-hot distribution has ~zero entropy.
    onehot = np.array([1.0, 0.0, 0.0, 0.0])
    assert attention_entropy(onehot) == pytest.approx(0.0, abs=1e-6)


def test_recent_mass_counts_from_the_end():
    dist = np.array([0.1, 0.2, 0.3, 0.4])  # newest keys at the end
    assert recent_mass(dist, 1) == pytest.approx(0.4)
    assert recent_mass(dist, 2) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        recent_mass(dist, 0)


def test_summarize_attention_shapes_and_peak_lag():
    L = 5
    # Two layers: one concentrated on the most recent key, one uniform.
    recent = np.zeros((L, L))
    recent[:, -1] = 1.0            # every query attends to newest key
    uniform = np.full((L, L), 1.0 / L)
    rows = summarize_attention([recent, uniform], pooling="last", recent_k=2)
    assert [r["layer"] for r in rows] == [0, 1]
    # Layer 0: newest key → peak lag 0 and high recent mass.
    assert rows[0]["peak_lag"] == 0
    assert rows[0]["recent_2_mass"] == pytest.approx(1.0)
    # Layer 1 uniform: entropy equals max entropy log(L).
    assert rows[1]["entropy_nats"] == pytest.approx(np.log(L))
    assert rows[1]["max_entropy_nats"] == pytest.approx(np.log(L))
