"""Batch-generate experiment YAML configs for Module 3.

Writes one config per (dataset x input_type x input_len x horizon x seed)
combination under ``configs/generated/`` so that the many forecasting
experiments do not have to be hand-written. The generated directory is
git-ignored; regenerate it any time with:

    python experiments/generate_configs.py
    python experiments/generate_configs.py --datasets ETTh1 ETTh2 --horizons 24 48 96
"""

import argparse
import itertools
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "configs" / "generated"

# Dataset name -> (relative CSV path, expected frequency for reference).
DATASETS = {
    "ETTh1": "data/raw/ETTh1.csv",
    "ETTh2": "data/raw/ETTh2.csv",
    "ETTm1": "data/raw/ETTm1.csv",
    "ETTm2": "data/raw/ETTm2.csv",
}

DEFAULT_DATASETS = ["ETTh1", "ETTh2"]
DEFAULT_INPUT_TYPES = ["univariate", "multivariate"]
DEFAULT_INPUT_LENS = [96]
DEFAULT_HORIZONS = [24, 48, 96]
DEFAULT_SEEDS = [42]


def build_config(name, input_type, input_len, horizon, seed):
    """Build one config dictionary matching the configs/*.yaml schema."""
    return {
        "dataset": {
            "name": name,
            "path": DATASETS[name],
            "target": "OT",
            "input_type": input_type,
        },
        "split": {
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "window": {
            "input_len": input_len,
            "horizon": horizon,
        },
        "training": {
            "batch_size": 64,
            "seed": seed,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
    }


def generate(datasets, input_types, input_lens, horizons, seeds):
    """Write all config combinations to OUT_DIR and return the file paths."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for name, input_type, input_len, horizon, seed in itertools.product(
        datasets, input_types, input_lens, horizons, seeds
    ):
        config = build_config(name, input_type, input_len, horizon, seed)
        file_name = (
            f"{name}_{input_type}_len{input_len}_h{horizon}_seed{seed}.yaml"
        )
        out_path = OUT_DIR / file_name
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        written.append(out_path)

    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment configs.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=list(DATASETS.keys()))
    parser.add_argument("--input-types", nargs="+", default=DEFAULT_INPUT_TYPES,
                        choices=["univariate", "multivariate"])
    parser.add_argument("--input-lens", nargs="+", type=int, default=DEFAULT_INPUT_LENS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    written = generate(
        datasets=args.datasets,
        input_types=args.input_types,
        input_lens=args.input_lens,
        horizons=args.horizons,
        seeds=args.seeds,
    )
    print(f"Generated {len(written)} config(s) in {OUT_DIR}")
    for path in written:
        print(" -", path.name)


if __name__ == "__main__":
    main()
