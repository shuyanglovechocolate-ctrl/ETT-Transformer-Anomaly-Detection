"""Tests for the inference-latency analysis (Module 8.4)."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_inference_latency import summarize_latency, time_model


def test_summarize_latency_math():
    # 1 second per full pass over 1000 windows in 10 batches.
    s = summarize_latency([1.0, 1.0], num_windows=1000, num_batches=10)
    assert s["num_repeats"] == 2
    assert s["mean_latency_ms_per_1k_windows"] == 1000.0   # 1s / 1000 windows -> 1000 ms/1k
    assert s["mean_latency_ms_per_batch"] == 100.0         # 1s / 10 batches -> 100 ms/batch
    assert s["std_latency_ms_per_1k_windows"] == 0.0


def test_summarize_latency_single_repeat_zero_std():
    s = summarize_latency([0.5], num_windows=500, num_batches=5)
    assert s["num_repeats"] == 1
    assert s["std_latency_ms_per_1k_windows"] == 0.0


class _Dummy(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=1)  # [B, L, F] -> [B, F]


def test_time_model_smoke_cpu():
    ds = TensorDataset(torch.randn(20, 5, 3), torch.randn(20, 3))
    loader = DataLoader(ds, batch_size=8)  # 20 windows in 3 batches
    rep_s, nw, nb = time_model(_Dummy(), loader, torch.device("cpu"),
                               warmup=1, repeats=3)
    assert nw == 20 and nb == 3
    assert len(rep_s) == 3
    assert all(t >= 0 for t in rep_s)

    stats = summarize_latency(rep_s, nw, nb)
    assert stats["mean_latency_ms_per_1k_windows"] >= 0
    assert not np.isnan(stats["mean_latency_ms_per_batch"])
