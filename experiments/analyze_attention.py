"""Exploratory Transformer attention analysis (Module 4.5 / RQ4).

Loads a saved Transformer checkpoint (no re-training), runs inference on the
test windows with ``return_attention=True``, and aggregates the per-layer
self-attention maps into interpretable summaries:

  * a mean attention heatmap for each layer;
  * the key distribution feeding the pooled (last-step) representation,
    plotted as attention-by-lag;
  * a per-layer CSV with peak lag, entropy and recent-attention mass.

The goal is a defensible, exploratory answer to "which past timesteps does the
Transformer attend to?", explicitly framed as a cue rather than a causal
explanation. The linear-family models remain more accurate under the tested
protocol (see the efficiency and forecasting analyses); this simply
characterises the attention model's internal behaviour.

Reads  results/logs/<eid>_config.yaml, results/checkpoints/<eid>_best.pt
Writes results/metrics/attention_summary.csv
       results/figures/attention_last_layer_heatmap.png
       results/figures/attention_by_lag.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.data.pipeline import build_data_pipeline  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.models.attention_analysis import (  # noqa: E402
    attention_by_lag, mean_attention_matrix, pooled_key_distribution,
    summarize_attention,
)
from src.viz import PALETTE, apply_paper_style  # noqa: E402


def collect_attention(model, loader, device, max_batches=None):
    """Run the model over ``loader`` and collect per-layer mean attention maps.

    Returns a list (one entry per encoder layer) of ``[L, L]`` mean matrices.
    """
    per_layer_batches = None
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0].to(device)
            _, attentions = model(x, return_attention=True)
            if per_layer_batches is None:
                per_layer_batches = [[] for _ in attentions]
            for li, attn in enumerate(attentions):
                per_layer_batches[li].append(attn.detach().cpu().numpy())
    if not per_layer_batches:
        raise RuntimeError("Loader produced no batches; cannot analyse attention.")
    return [mean_attention_matrix(batches) for batches in per_layer_batches]


def _plot_heatmap(mean_matrix, path, title):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_matrix, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xlabel("key position (input step; 0 = oldest)")
    ax.set_ylabel("query position (input step; 0 = oldest)")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean attention weight")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_by_lag(mean_matrices, pooling, path):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(mean_matrices):
        lag = attention_by_lag(pooled_key_distribution(m, pooling=pooling))
        ax.plot(np.arange(len(lag)), lag, marker="o", markersize=3,
                color=PALETTE[i % len(PALETTE)], label=f"layer {i}")
    ax.set_xlabel("lag (steps before the forecast origin; 0 = most recent)")
    ax.set_ylabel("mean attention weight")
    ax.set_title(f"Attention over input lags feeding the pooled state "
                 f"(pooling='{pooling}')")
    ax.legend(title="encoder layer")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(eid, project_root, device, max_batches=None):
    results = os.path.join(project_root, "results")
    config = load_config(os.path.join(results, "logs", f"{eid}_config.yaml"))
    if config["model"]["name"].lower() != "transformer":
        raise ValueError(f"{eid} is not a Transformer run; attention analysis "
                         f"requires a Transformer checkpoint.")
    set_seed(config["training"]["seed"])

    data = build_data_pipeline(config, project_root=project_root)
    model = build_model(config, num_features=data["num_features"],
                        feature_cols=data["feature_cols"]).to(device)

    ckpt = os.path.join(results, "checkpoints", f"{eid}_best.pt")
    if count_parameters(model) > 0 and os.path.exists(ckpt):
        load_checkpoint(model, ckpt)
        model.to(device)
    else:
        raise FileNotFoundError(f"Missing checkpoint for {eid}: {ckpt}")

    pooling = config["model"].get("pooling", "last")
    mean_matrices = collect_attention(model, data["test_loader"], device,
                                      max_batches=max_batches)

    fig_dir = os.path.join(results, "figures")
    metrics_dir = os.path.join(results, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    _plot_heatmap(mean_matrices[-1],
                  os.path.join(fig_dir, "attention_last_layer_heatmap.png"),
                  f"Mean self-attention — last encoder layer ({eid})")
    _plot_by_lag(mean_matrices, pooling,
                 os.path.join(fig_dir, "attention_by_lag.png"))

    rows = summarize_attention(mean_matrices, pooling=pooling)
    for r in rows:
        r["experiment_id"] = eid
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(metrics_dir, "attention_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[{eid}] wrote attention_summary.csv ({len(summary)} layers), "
          f"attention_last_layer_heatmap.png, attention_by_lag.png")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory Transformer attention analysis (post-hoc).")
    parser.add_argument(
        "--experiment-id",
        default="ETTh1_transformer_multivariate_len96_h24_seed42",
        help="Experiment id whose config/checkpoint to load.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional cap on test batches (for a quick run).")
    args = parser.parse_args()

    device = get_device()
    analyze(args.experiment_id, args.project_root, device,
            max_batches=args.max_batches)


if __name__ == "__main__":
    main()
