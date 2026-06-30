"""Forward-shape and factory tests for the Module 2 model library.

Run with: pytest tests/
Every model must map [B, input_len, num_features] -> [B, horizon] for both
univariate (F=1) and multivariate (F=7) inputs.
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models import (
    NaiveForecaster,
    LinearForecaster,
    NLinearForecaster,
    DLinearForecaster,
    LSTMForecaster,
    TransformerForecaster,
    build_model,
    validate_model_config,
    count_parameters,
    get_model_summary,
    MODEL_REGISTRY,
)
from src.models.dlinear import SeriesDecomposition, MovingAverage

BATCH = 8
INPUT_LEN = 96
HORIZON = 24
MULTI_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
UNI_FEATURES = ["OT"]


def _x(num_features):
    return torch.randn(BATCH, INPUT_LEN, num_features)


@pytest.mark.parametrize("num_features", [1, 7])
def test_naive_shape(num_features):
    feature_cols = UNI_FEATURES if num_features == 1 else MULTI_FEATURES
    model = NaiveForecaster(INPUT_LEN, num_features, HORIZON, feature_cols)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


@pytest.mark.parametrize("num_features", [1, 7])
def test_linear_shape(num_features):
    model = LinearForecaster(INPUT_LEN, num_features, HORIZON)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


@pytest.mark.parametrize("num_features", [1, 7])
def test_nlinear_shape(num_features):
    feature_cols = UNI_FEATURES if num_features == 1 else MULTI_FEATURES
    model = NLinearForecaster(INPUT_LEN, num_features, HORIZON, feature_cols)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


def test_nlinear_is_shift_equivariant():
    # The defining property of NLinear: subtracting then adding back the last
    # value makes the forecast shift-equivariant, f(x + c) = f(x) + c, for any
    # constant offset c, regardless of the learned weights.
    model = NLinearForecaster(INPUT_LEN, 7, HORIZON, MULTI_FEATURES)
    model.eval()
    x = torch.randn(BATCH, INPUT_LEN, 7)
    c = 3.0
    with torch.no_grad():
        out = model(x)
        out_shifted = model(x + c)
    assert torch.allclose(out_shifted, out + c, atol=1e-4)


@pytest.mark.parametrize("num_features", [1, 7])
def test_dlinear_shape(num_features):
    model = DLinearForecaster(INPUT_LEN, num_features, HORIZON)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


@pytest.mark.parametrize("num_features", [1, 7])
def test_dlinear_channel_independent_shape(num_features):
    model = DLinearForecaster(
        INPUT_LEN, num_features, HORIZON, channel_independent=True
    )
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


def test_dlinear_has_two_linear_components():
    # Decomposition DLinear has a trend AND a seasonal linear head, so it holds
    # roughly twice the parameters of the plain Linear baseline.
    linear = LinearForecaster(INPUT_LEN, 7, HORIZON)
    dlinear = DLinearForecaster(INPUT_LEN, 7, HORIZON)
    n_linear = sum(p.numel() for p in linear.parameters())
    n_dlinear = sum(p.numel() for p in dlinear.parameters())
    assert n_dlinear == 2 * n_linear


@pytest.mark.parametrize("kernel_size", [3, 25])
def test_dlinear_kernel_configurable(kernel_size):
    model = DLinearForecaster(INPUT_LEN, 7, HORIZON, kernel_size=kernel_size)
    assert model.kernel_size == kernel_size
    out = model(_x(7))
    assert out.shape == (BATCH, HORIZON)


def test_series_decomposition_reconstructs_input():
    # seasonal + trend must reconstruct the original series, and a moving average
    # of a constant series returns the constant (seasonal ~ 0).
    decomp = SeriesDecomposition(kernel_size=25)
    x = torch.randn(BATCH, INPUT_LEN, 7)
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape and trend.shape == x.shape
    assert torch.allclose(seasonal + trend, x, atol=1e-5)

    const = torch.full((BATCH, INPUT_LEN, 7), 3.0)
    smoothed = MovingAverage(25)(const)
    assert torch.allclose(smoothed, const, atol=1e-5)


@pytest.mark.parametrize("num_features", [1, 7])
def test_lstm_shape(num_features):
    model = LSTMForecaster(INPUT_LEN, num_features, HORIZON)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


@pytest.mark.parametrize("num_features", [1, 7])
def test_transformer_shape(num_features):
    model = TransformerForecaster(INPUT_LEN, num_features, HORIZON)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


def test_naive_uses_ot_index_not_hardcoded():
    # Put OT in the middle to ensure the index is resolved from feature_cols.
    feature_cols = ["A", "OT", "B"]
    model = NaiveForecaster(INPUT_LEN, 3, HORIZON, feature_cols)
    x = torch.randn(BATCH, INPUT_LEN, 3)
    out = model(x)
    # Every horizon step equals the last OT value of the input window.
    expected_last_ot = x[:, -1, 1]
    assert torch.allclose(out[:, 0], expected_last_ot)
    assert torch.allclose(out[:, -1], expected_last_ot)


def test_transformer_rejects_bad_d_model_nhead():
    with pytest.raises(ValueError):
        TransformerForecaster(INPUT_LEN, 7, HORIZON, d_model=64, nhead=5)


def test_transformer_return_attention():
    num_layers = 2
    model = TransformerForecaster(
        INPUT_LEN, 7, HORIZON, num_layers=num_layers
    )
    out, attentions = model(_x(7), return_attention=True)
    assert out.shape == (BATCH, HORIZON)
    assert len(attentions) == num_layers
    # Each attention map is [batch, input_len, input_len] (heads averaged).
    assert attentions[0].shape == (BATCH, INPUT_LEN, INPUT_LEN)


def _config(model_section, num_features):
    input_type = "univariate" if num_features == 1 else "multivariate"
    return {
        "dataset": {"name": "ETTh1", "path": "x", "target": "OT",
                    "input_type": input_type},
        "split": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "window": {"input_len": INPUT_LEN, "horizon": HORIZON},
        "training": {"batch_size": 64, "seed": 42},
        "model": model_section,
    }


@pytest.mark.parametrize("model_section", [
    {"name": "naive"},
    {"name": "linear"},
    {"name": "nlinear"},
    {"name": "dlinear", "kernel_size": 25},
    {"name": "dlinear", "kernel_size": 25, "channel_independent": True},
    {"name": "lstm", "hidden_dim": 32, "num_layers": 1, "dropout": 0.1},
    {"name": "transformer", "d_model": 32, "nhead": 4, "num_layers": 1},
])
def test_build_model_factory(model_section):
    num_features = 7
    config = _config(model_section, num_features)
    model = build_model(config, num_features=num_features,
                        feature_cols=MULTI_FEATURES)
    out = model(_x(num_features))
    assert out.shape == (BATCH, HORIZON)


def test_build_model_unknown_raises():
    config = _config({"name": "bogus"}, 7)
    with pytest.raises(ValueError):
        build_model(config, num_features=7, feature_cols=MULTI_FEATURES)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def test_count_parameters_naive_is_zero():
    model = NaiveForecaster(INPUT_LEN, 7, HORIZON, MULTI_FEATURES)
    assert count_parameters(model) == 0
    assert count_parameters(model, trainable_only=False) == 0


def test_count_parameters_nonzero_for_linear():
    model = LinearForecaster(INPUT_LEN, 7, HORIZON)
    expected = (INPUT_LEN * 7 + 1) * HORIZON  # weights + bias
    assert count_parameters(model) == expected


def test_get_model_summary_fields():
    model = DLinearForecaster(INPUT_LEN, 7, HORIZON)
    summary = get_model_summary(model, model_name="dlinear")
    assert summary["model_name"] == "dlinear"
    assert summary["class_name"] == "DLinearForecaster"
    assert summary["total_parameters"] == count_parameters(model, trainable_only=False)
    assert summary["trainable_parameters"] == count_parameters(model)
    assert summary["input_len"] == INPUT_LEN
    assert summary["num_features"] == 7
    assert summary["horizon"] == HORIZON


# ---------------------------------------------------------------------------
# Model config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_section", [
    {"name": "naive"},
    {"name": "linear"},
    {"name": "dlinear", "kernel_size": 25},
    {"name": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.2},
    {"name": "transformer", "d_model": 64, "nhead": 4, "pooling": "mean"},
])
def test_validate_model_config_accepts_valid(model_section):
    validate_model_config(_config(model_section, 7))  # should not raise


def test_validate_missing_model_section():
    config = _config({"name": "linear"}, 7)
    del config["model"]
    with pytest.raises(ValueError):
        validate_model_config(config)


def test_validate_missing_model_name():
    with pytest.raises(ValueError):
        validate_model_config(_config({"kernel_size": 25}, 7))


def test_validate_unknown_model_name():
    with pytest.raises(ValueError):
        validate_model_config(_config({"name": "bogus"}, 7))


def test_validate_dlinear_even_kernel_raises():
    with pytest.raises(ValueError):
        validate_model_config(_config({"name": "dlinear", "kernel_size": 24}, 7))


def test_validate_lstm_bad_hidden_dim_raises():
    with pytest.raises(ValueError):
        validate_model_config(_config({"name": "lstm", "hidden_dim": 0}, 7))


def test_validate_lstm_bad_dropout_raises():
    with pytest.raises(ValueError):
        validate_model_config(_config({"name": "lstm", "dropout": 1.0}, 7))


def test_validate_transformer_bad_d_model_nhead_raises():
    with pytest.raises(ValueError):
        validate_model_config(
            _config({"name": "transformer", "d_model": 64, "nhead": 5}, 7)
        )


def test_validate_transformer_bad_pooling_raises():
    with pytest.raises(ValueError):
        validate_model_config(
            _config({"name": "transformer", "pooling": "sum"}, 7)
        )


def test_build_model_invokes_validation():
    # An invalid model config must fail in build_model (via validate_model_config),
    # not silently build something wrong.
    config = _config({"name": "transformer", "d_model": 64, "nhead": 5}, 7)
    with pytest.raises(ValueError):
        build_model(config, num_features=7, feature_cols=MULTI_FEATURES)


# ---------------------------------------------------------------------------
# Registry and model metadata
# ---------------------------------------------------------------------------

def test_model_registry_contents():
    assert set(MODEL_REGISTRY) == {
        "naive", "linear", "nlinear", "dlinear", "lstm", "transformer"
    }


def test_model_metadata_attributes():
    transformer = TransformerForecaster(INPUT_LEN, 7, HORIZON)
    assert transformer.model_type == "attention"
    assert transformer.supports_attention is True

    naive = NaiveForecaster(INPUT_LEN, 7, HORIZON, MULTI_FEATURES)
    assert naive.model_type == "baseline"
    assert naive.requires_feature_cols is True
    assert naive.supports_attention is False

    dlinear = DLinearForecaster(INPUT_LEN, 7, HORIZON)
    assert dlinear.model_type == "linear"


def test_get_model_summary_includes_metadata():
    model = TransformerForecaster(INPUT_LEN, 7, HORIZON)
    summary = get_model_summary(model, model_name="transformer")
    assert summary["model_type"] == "attention"
    assert summary["supports_attention"] is True
    assert summary["supports_multivariate"] is True
    assert "description" in summary
