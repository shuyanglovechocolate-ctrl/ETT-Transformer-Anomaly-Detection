"""Batch experiment runner (Module 3.4a: pilot matrix).

Builds experiment configs in memory (no temp YAML on disk) and runs them through
the 5.3 experiment runner. Fail-soft: one failing experiment does not stop the
rest; a success/skip/fail summary is printed at the end.

Examples:
    python experiments/run_matrix.py --matrix pilot
    python experiments/run_matrix.py --matrix pilot --models naive linear nlinear dlinear --epochs 50 --skip-existing
    python experiments/run_matrix.py --matrix pilot --models lstm transformer --epochs 2 --force
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.experiment import (
    build_experiment_id,
    build_output_paths,
    run_experiment_from_config,
)

DATASET_PATHS = {
    "ETTh1": "data/raw/ETTh1.csv",
    "ETTh2": "data/raw/ETTh2.csv",
    "ETTm1": "data/raw/ETTm1.csv",
    "ETTm2": "data/raw/ETTm2.csv",
}

PILOT_MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]

PILOT_DEFAULTS = {
    "dataset": "ETTh1",
    "input_type": "multivariate",
    "input_len": 96,
    "horizon": 24,
    "seed": 42,
}


def get_default_model_config(model_name: str) -> dict:
    """Return a fresh model config (name + sensible default hyper-parameters)."""
    defaults = {
        "naive": {"name": "naive"},
        "linear": {"name": "linear"},
        "nlinear": {"name": "nlinear"},
        "dlinear": {"name": "dlinear", "kernel_size": 25, "channel_independent": False},
        "lstm": {"name": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.2},
        "transformer": {
            "name": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_feedforward": 128, "dropout": 0.1, "pooling": "last",
        },
    }
    if model_name not in defaults:
        raise ValueError(
            f"Unknown model '{model_name}'. Known: {', '.join(defaults)}."
        )
    return dict(defaults[model_name])


def build_matrix_config(dataset, input_type, input_len, horizon, seed, model_name) -> dict:
    """Build a complete config dict for one matrix cell."""
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    return {
        "dataset": {
            "name": dataset,
            "path": DATASET_PATHS[dataset],
            "target": "OT",
            "input_type": input_type,
        },
        "split": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "window": {"input_len": input_len, "horizon": horizon},
        "training": {
            "batch_size": 64, "seed": seed, "epochs": 100,
            "learning_rate": 0.001, "weight_decay": 0.0, "grad_clip": 1.0,
            "early_stopping": {"patience": 10, "min_delta": 0.0},
            "scheduler": {"name": "reduce_on_plateau", "factor": 0.5, "patience": 5},
        },
        "model": get_default_model_config(model_name),
    }


def build_pilot_matrix(models=None) -> list:
    """Build the pilot matrix configs (ETTh1 multivariate len96 h24 seed42)."""
    models = models or PILOT_MODELS
    return [
        build_matrix_config(
            PILOT_DEFAULTS["dataset"], PILOT_DEFAULTS["input_type"],
            PILOT_DEFAULTS["input_len"], PILOT_DEFAULTS["horizon"],
            PILOT_DEFAULTS["seed"], m,
        )
        for m in models
    ]


def experiment_exists(config: dict, results_dir: str) -> bool:
    """True if this experiment's metrics JSON already exists."""
    eid = build_experiment_id(config)
    return os.path.exists(build_output_paths(eid, results_dir)["metrics"])


def run_matrix(
    configs,
    project_root: str = ".",
    results_dir: str = None,
    epochs: int = None,
    skip_existing: bool = False,
    force: bool = False,
) -> dict:
    """Run a list of experiment configs fail-soft; return a run summary."""
    if skip_existing and force:
        raise ValueError("--skip-existing and --force are mutually exclusive.")
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")

    overrides = {"epochs": epochs} if epochs is not None else None
    succeeded, skipped, failed = [], [], []

    for config in configs:
        eid = build_experiment_id(config)
        if skip_existing and experiment_exists(config, results_dir):
            print(f"[skip] {eid} (metrics already exist)")
            skipped.append(eid)
            continue
        try:
            run_experiment_from_config(
                config, overrides=overrides,
                project_root=project_root, results_dir=results_dir,
            )
            succeeded.append(eid)
        except Exception as exc:  # fail-soft: record and continue
            print(f"[FAIL] {eid}: {exc}")
            failed.append({"experiment_id": eid,
                           "model": config.get("model", {}).get("name"),
                           "error": str(exc)})

    print("\nMatrix finished.")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    for f in failed:
        print(f"  FAIL {f['experiment_id']} ({f['model']}): {f['error']}")

    return {"succeeded": succeeded, "skipped": skipped, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Run a batch experiment matrix.")
    parser.add_argument("--matrix", choices=["pilot"], default="pilot",
                        help="Which matrix to run (currently only 'pilot').")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=PILOT_MODELS, help="Subset of models to run.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs for all runs.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments whose metrics JSON already exists.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if metrics already exist (opposite of --skip-existing).")
    args = parser.parse_args()

    if args.skip_existing and args.force:
        parser.error("--skip-existing and --force are mutually exclusive.")

    configs = build_pilot_matrix(models=args.models)
    run_matrix(
        configs,
        project_root=str(PROJECT_ROOT),
        epochs=args.epochs,
        skip_existing=args.skip_existing,
        force=args.force,
    )


if __name__ == "__main__":
    main()
