"""CLI wrapper for a single recorded forecasting experiment (Module 3).

Usage:
    python src/train.py --config configs/ETTh1_multivariate_h24.yaml
    python src/train.py --config configs/ETTh1_multivariate_h24.yaml --epochs 30 --lr 0.0005
    python src/train.py --config configs/ETTh1_univariate_h24.yaml --patience 5 --grad-clip 0.5

The orchestration lives in src/training/experiment.py.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run one forecasting experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (e.g. 1 for a smoke test).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the learning rate.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stopping patience.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Override gradient-clipping max norm.")
    args = parser.parse_args()

    overrides = {
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
    }
    run_experiment(
        config_path=args.config,
        overrides=overrides,
        project_root=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
