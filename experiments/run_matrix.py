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
import json
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

# Core matrix defaults.
CORE_LIGHT_MODELS = ["naive", "linear", "nlinear", "dlinear"]
CORE_DEEP_MODELS = ["lstm", "transformer"]
CORE_DATASETS = ["ETTh1", "ETTh2"]
CORE_INPUT_TYPES = ["univariate", "multivariate"]
CORE_HORIZONS = [24, 48, 96]
CORE_SEEDS = [42, 2024, 3407]
CORE_INPUT_LEN = 96

# Robustness check: regularized deep models on the longest horizon.
ROBUSTNESS_DEEP_MODELS = ["lstm", "transformer"]
REGULARIZED_MODEL_CONFIGS = {
    "lstm": {"name": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3},
    "transformer": {
        "name": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 128, "dropout": 0.2, "pooling": "last",
    },
}
REGULARIZED_TRAINING = {
    "lstm": {"learning_rate": 0.001, "weight_decay": 0.0001},
    "transformer": {"learning_rate": 0.0001, "weight_decay": 0.0001},
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


def build_matrix(models, datasets, input_types, horizons, seeds, input_len=CORE_INPUT_LEN) -> list:
    """Build the full cartesian product of experiment configs."""
    return [
        build_matrix_config(dataset, input_type, input_len, horizon, seed, model)
        for dataset in datasets
        for input_type in input_types
        for horizon in horizons
        for seed in seeds
        for model in models
    ]


def build_core_light_matrix(datasets=None, input_types=None, horizons=None,
                            seeds=None, models=None) -> list:
    """Light-model core matrix (default 4 x 2 x 2 x 3 x 3 = 144 runs)."""
    return build_matrix(
        models or CORE_LIGHT_MODELS,
        datasets or CORE_DATASETS,
        input_types or CORE_INPUT_TYPES,
        horizons or CORE_HORIZONS,
        seeds or CORE_SEEDS,
    )


def build_core_deep_matrix(datasets=None, input_types=None, horizons=None,
                           seeds=None, models=None) -> list:
    """Deep-model core matrix (default multivariate only, 2 x 2 x 1 x 3 x 3 = 36 runs)."""
    return build_matrix(
        models or CORE_DEEP_MODELS,
        datasets or CORE_DATASETS,
        input_types or ["multivariate"],
        horizons or CORE_HORIZONS,
        seeds or CORE_SEEDS,
    )


def build_regularized_config(dataset, horizon, seed, model_name) -> dict:
    """Build a regularized deep-model config, tagged 'regularized'."""
    config = build_matrix_config(dataset, "multivariate", CORE_INPUT_LEN,
                                 horizon, seed, model_name)
    config["model"] = dict(REGULARIZED_MODEL_CONFIGS[model_name])
    reg = REGULARIZED_TRAINING[model_name]
    config["training"]["learning_rate"] = reg["learning_rate"]
    config["training"]["weight_decay"] = reg["weight_decay"]
    config["experiment"] = {"tag": "regularized"}
    return config


def build_robustness_deep_matrix(datasets=None, horizons=None, seeds=None,
                                 models=None) -> list:
    """Regularized deep models on the longest horizon (default 2 x 1 x 3 x 2 = 12)."""
    datasets = datasets or CORE_DATASETS
    horizons = horizons or [96]
    seeds = seeds or CORE_SEEDS
    models = models or ROBUSTNESS_DEEP_MODELS
    return [
        build_regularized_config(dataset, horizon, seed, model)
        for dataset in datasets
        for horizon in horizons
        for seed in seeds
        for model in models
    ]


def build_matrix_configs(matrix, datasets=None, input_types=None, horizons=None,
                         seeds=None, models=None) -> list:
    """Dispatch to the requested matrix builder."""
    if matrix == "pilot":
        return build_pilot_matrix(models=models)
    if matrix == "core-light":
        return build_core_light_matrix(datasets, input_types, horizons, seeds, models)
    if matrix == "core-deep":
        return build_core_deep_matrix(datasets, input_types, horizons, seeds, models)
    if matrix == "robustness-deep":
        return build_robustness_deep_matrix(datasets, horizons, seeds, models)
    raise ValueError(f"Unknown matrix '{matrix}'.")


REQUIRED_METRICS_KEYS = [
    "experiment_id", "dataset", "model", "metrics", "training", "paths",
]


def experiment_exists(config: dict, results_dir: str) -> bool:
    """True if this experiment's metrics JSON file exists (existence only)."""
    eid = build_experiment_id(config)
    return os.path.exists(build_output_paths(eid, results_dir)["metrics"])


def is_completed_metrics_file(metrics_path: str) -> bool:
    """True only if the metrics file exists, is non-empty, parses as JSON and
    contains all required keys.

    This guards --skip-existing against empty or half-written metrics files left
    by an interrupted run, which must be re-run rather than skipped.
    """
    if not os.path.exists(metrics_path):
        return False
    if os.path.getsize(metrics_path) == 0:
        return False
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return all(key in data for key in REQUIRED_METRICS_KEYS)


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
        metrics_path = build_output_paths(eid, results_dir)["metrics"]
        if skip_existing and is_completed_metrics_file(metrics_path):
            print(f"[skip] {eid} (completed metrics exist)")
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
    parser.add_argument("--matrix",
                        choices=["pilot", "core-light", "core-deep", "robustness-deep"],
                        default="pilot", help="Which matrix to run.")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=PILOT_MODELS, help="Subset of models to run.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(DATASET_PATHS), help="Subset of datasets.")
    parser.add_argument("--input-types", nargs="+", default=None,
                        choices=["univariate", "multivariate"], help="Input types.")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Forecast horizons.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs for all runs.")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory (default: <project>/results).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments whose metrics JSON already exists.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if metrics already exist (opposite of --skip-existing).")
    args = parser.parse_args()

    if args.skip_existing and args.force:
        parser.error("--skip-existing and --force are mutually exclusive.")

    configs = build_matrix_configs(
        args.matrix, datasets=args.datasets, input_types=args.input_types,
        horizons=args.horizons, seeds=args.seeds, models=args.models,
    )
    print(f"Matrix '{args.matrix}': {len(configs)} experiment(s).")
    run_matrix(
        configs,
        project_root=str(PROJECT_ROOT),
        results_dir=args.results_dir,
        epochs=args.epochs,
        skip_existing=args.skip_existing,
        force=args.force,
    )


if __name__ == "__main__":
    main()
