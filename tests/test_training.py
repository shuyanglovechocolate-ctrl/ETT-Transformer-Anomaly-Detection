"""Tests for the Module 3.1 minimal training pipeline.

Run with: pytest tests/
Uses small synthetic data and a fitted scaler so the tests stay fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.models import LinearForecaster, NaiveForecaster, count_parameters
from src.training import (
    train_one_epoch,
    fit_model,
    build_scheduler,
    evaluate_loss,
    predict,
    create_prediction_dataframe,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    build_experiment_id,
    build_output_paths,
    save_config_snapshot,
    append_experiment_log,
    run_experiment_from_config,
)


def _full_config(model_section, input_type="multivariate", epochs=1):
    return {
        "dataset": {"name": "ETTh1", "path": "data/raw/ETTh1.csv",
                    "target": "OT", "input_type": input_type},
        "split": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "window": {"input_len": 96, "horizon": 24},
        "training": {"batch_size": 64, "seed": 42, "epochs": epochs,
                     "learning_rate": 0.001, "weight_decay": 0.0},
        "model": model_section,
    }

INPUT_LEN = 16
HORIZON = 4
NUM_FEATURES = 7
N = 64
DEVICE = torch.device("cpu")


def _loader():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, INPUT_LEN, NUM_FEATURES)).astype(np.float32)
    y = rng.normal(size=(N, HORIZON)).astype(np.float32)
    return create_dataloader(X, y, batch_size=16, shuffle=True)


def _scaler_y():
    s = StandardScaler()
    s.fit(np.random.default_rng(1).normal(size=(100, 1)))
    return s


def test_train_one_epoch_returns_float():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, _loader(), optimizer, nn.MSELoss(), DEVICE)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_training_loss_not_increasing_much():
    torch.manual_seed(0)
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader()
    criterion = nn.MSELoss()
    first = train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    second = train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    assert np.isfinite(first) and np.isfinite(second)
    # Stable convergence: second epoch should not blow up.
    assert second <= first + 1e-3


def test_evaluate_loss_returns_float():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    loss = evaluate_loss(model, _loader(), nn.MSELoss(), DEVICE)
    assert isinstance(loss, float) and np.isfinite(loss)


def test_predict_shape():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    y_true, y_pred = predict(model, _loader(), DEVICE)
    assert y_true.shape == (N, HORIZON)
    assert y_pred.shape == (N, HORIZON)


def test_create_prediction_dataframe_columns():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    y_true, y_pred = predict(model, _loader(), DEVICE)
    dates = np.tile(
        np.arange(HORIZON), (N, 1)
    ).astype("datetime64[h]")  # dummy datetimes [N, horizon]
    df = create_prediction_dataframe(y_true, y_pred, dates, _scaler_y())
    expected = {
        "sample_index", "horizon_index", "target_date",
        "y_true", "y_pred", "residual", "abs_residual",
    }
    assert expected.issubset(df.columns)
    assert len(df) == N * HORIZON


def test_save_and_load_checkpoint(tmp_path):
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, epoch=1, val_loss=0.5, path=str(path))
    assert path.exists()

    model2 = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    ckpt = load_checkpoint(model2, str(path))
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == 0.5


def test_naive_has_no_parameters_so_no_optimizer():
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    model = NaiveForecaster(INPUT_LEN, NUM_FEATURES, HORIZON, feature_cols)
    assert count_parameters(model) == 0
    # Building an Adam optimizer on an empty parameter list must fail, which is
    # exactly why the training entry point gates on count_parameters > 0.
    with pytest.raises(ValueError):
        torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# 5.2: early stopping, scheduler, fit_model history, grad clipping
# ---------------------------------------------------------------------------

def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=3, mode="min")
    assert es.step(1.0) is True          # first value -> best
    for _ in range(2):                    # two non-improving
        es.step(2.0)
    assert es.should_stop is False
    es.step(2.0)                          # third non-improving -> stop
    assert es.should_stop is True
    assert es.best_epoch == 1


def test_early_stopping_resets_counter_on_improvement():
    es = EarlyStopping(patience=2, mode="min")
    es.step(1.0)
    es.step(2.0)                          # counter = 1
    assert es.counter == 1
    assert es.step(0.5) is True           # improvement resets counter
    assert es.counter == 0
    assert es.best_epoch == 3


def test_train_one_epoch_supports_grad_clip():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(
        model, _loader(), optimizer, nn.MSELoss(), DEVICE, grad_clip=1.0
    )
    assert np.isfinite(loss)


def test_fit_model_returns_summary_and_saves_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "best.pt"
    history = fit_model(
        model, _loader(), _loader(), optimizer, nn.MSELoss(), DEVICE,
        epochs=3, checkpoint_path=str(ckpt), early_stopping_patience=5,
    )
    for key in ("best_epoch", "best_val_loss", "stopped_early", "epochs_ran"):
        assert key in history
    assert history["epochs_ran"] == 3
    assert ckpt.exists()
    # Best checkpoint can be reloaded and used for prediction.
    model2 = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    load_checkpoint(model2, str(ckpt))
    y_true, y_pred = predict(model2, _loader(), DEVICE)
    assert y_pred.shape == (N, HORIZON)


def test_fit_model_stops_early_with_tiny_patience(tmp_path):
    torch.manual_seed(0)
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-9)  # ~no improvement
    history = fit_model(
        model, _loader(), _loader(), optimizer, nn.MSELoss(), DEVICE,
        epochs=50, checkpoint_path=str(tmp_path / "b.pt"),
        early_stopping_patience=2,
    )
    assert history["epochs_ran"] < 50


def test_build_scheduler_none_when_absent():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert build_scheduler(optimizer, {"training": {}}) is None


def test_build_and_run_reduce_on_plateau():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {"training": {"scheduler": {"name": "reduce_on_plateau",
                                         "factor": 0.5, "patience": 1}}}
    scheduler = build_scheduler(optimizer, config)
    assert scheduler is not None
    history = fit_model(
        model, _loader(), _loader(), optimizer, nn.MSELoss(), DEVICE,
        epochs=2, scheduler=scheduler,
    )
    assert history["epochs_ran"] == 2


# ---------------------------------------------------------------------------
# 5.3: experiment runner and logging
# ---------------------------------------------------------------------------

def test_build_experiment_id_format():
    eid = build_experiment_id(_full_config({"name": "dlinear"}))
    assert eid == "ETTh1_dlinear_multivariate_len96_h24_seed42"


def test_build_experiment_id_channel_independent_marker():
    eid = build_experiment_id(
        _full_config({"name": "dlinear", "channel_independent": True})
    )
    assert eid == "ETTh1_dlinear_ci_multivariate_len96_h24_seed42"


def test_build_output_paths_keys():
    paths = build_output_paths("expid", results_dir="results")
    for key in ("checkpoint", "metrics", "predictions", "loss_curve",
                "prediction_plot", "history", "config_snapshot"):
        assert key in paths
    assert paths["config_snapshot"].endswith("expid_config.yaml")


def test_save_config_snapshot_writes_yaml(tmp_path):
    import yaml
    config = _full_config({"name": "linear"})
    path = tmp_path / "snap.yaml"
    save_config_snapshot(config, str(path))
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded["model"]["name"] == "linear"


def test_append_experiment_log_header_then_append(tmp_path):
    import csv
    log = tmp_path / "experiment_log.csv"
    append_experiment_log({"experiment_id": "a", "mae": 1.0}, str(log))
    append_experiment_log({"experiment_id": "b", "mae": 2.0}, str(log))
    with open(log) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "a"
    assert rows[1]["experiment_id"] == "b"


def test_run_experiment_end_to_end(tmp_path):
    # Real ETTh1, linear model, 1 epoch, results redirected to tmp_path.
    config = _full_config({"name": "linear"}, epochs=1)
    record = run_experiment_from_config(
        config,
        overrides={"epochs": 1},
        project_root=str(PROJECT_ROOT),
        results_dir=str(tmp_path),
    )
    # Structured metrics record.
    for key in ("experiment_id", "dataset", "model", "metrics", "training", "paths"):
        assert key in record
    assert "mae" in record["metrics"]
    # Files written under the redirected results dir.
    eid = record["experiment_id"]
    assert (tmp_path / "metrics" / f"{eid}_metrics.json").exists()
    assert (tmp_path / "predictions" / f"{eid}_predictions.csv").exists()
    assert (tmp_path / "logs" / f"{eid}_config.yaml").exists()
    assert (tmp_path / "metrics" / "experiment_log.csv").exists()
