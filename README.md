# ETT Transformer Anomaly Detection

This project investigates deep learning-based time series forecasting and anomaly detection using the Electricity Transformer Temperature (ETT) datasets. The main goal is to predict oil temperature (OT) using historical multivariate time series data and identify potential anomalies based on prediction errors.

## Project Objectives

- Explore and preprocess ETT time series datasets.
- Build an LSTM baseline model for oil temperature forecasting.
- Develop a Transformer-based forecasting model.
- Compare model performance using MAE and RMSE.
- Detect anomalies using prediction errors.
- Visualize forecasting results and detected anomalies.

## Dataset

### Source

The project uses the four public ETT (Electricity Transformer Temperature) datasets released with the Informer paper (Zhou et al., AAAI 2021): https://github.com/zhouhaoyi/ETDataset. The CSV files are stored locally under `data/raw/`.

- `ETTh1`, `ETTh2` — hourly data (1 record per hour)
- `ETTm1`, `ETTm2` — 15-minute data (1 record per 15 minutes)

This project primarily uses `ETTh1` and `ETTh2`, and may extend to `ETTm1` / `ETTm2`.

### Fields

| Field | Meaning |
| --- | --- |
| `date` | Timestamp |
| `HUFL` / `HULL` | High UseFul Load / High UseLess Load |
| `MUFL` / `MULL` | Middle UseFul Load / Middle UseLess Load |
| `LUFL` / `LULL` | Low UseFul Load / Low UseLess Load |
| `OT` | Oil Temperature — the forecasting target |

The forecasting target is `OT`.

## Methods

- Exploratory Data Analysis
- Chronological train / validation / test splitting
- Leakage-free feature and target scaling
- Sliding window sequence generation
- LSTM forecasting
- Transformer forecasting
- Prediction-error-based anomaly detection

## Module 1: Data Pipeline

Module 1 establishes a reproducible and leakage-free ETT data processing pipeline. It includes chronological train/validation/test splitting, training-only feature and target scaling, configurable univariate and multivariate input construction, independent sliding-window generation within each split, and PyTorch DataLoader preparation for multi-horizon oil temperature forecasting.

### Train / Validation / Test split

The data is split **in chronological order** (never shuffled) into:

- Train: 70%
- Validation: 10%
- Test: 20%

**Rationale.** This ratio is broadly consistent with common ETT benchmark settings, which makes the results easier to compare with existing literature. 70% of the data provides sufficient training samples for the LSTM and Transformer models; the validation set is used for model selection and early stopping; the test set is used only for final evaluation. Shuffling is avoided because randomly splitting a time series would let future observations leak into the training set.

### Leakage-free processing order

The order of scaling and windowing is strict and must not be reordered:

```text
raw data
-> time-based split (train / val / test)
-> fit scaler on train only
-> transform train / val / test with the same scalers
-> create sliding windows independently inside each split
```

Fitting the scaler on the full dataset would expose validation/test distribution information to training. Building windows on the full series before splitting would create windows crossing split boundaries, leaking future data into training. Both are avoided.

### `scaler_x` vs `scaler_y`

Two separate `StandardScaler`s are maintained, both fitted on the training split only:

- **`scaler_x`** standardises the input features. For multivariate input it is fitted on all 7 columns `[HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]`; for univariate input on `[OT]` only.
- **`scaler_y`** is fitted only on the training `OT` column. It is used to inverse-transform model predictions and ground truth back to the original oil-temperature scale.

`scaler_x` cannot be used to inverse-transform OT: in the multivariate setting it expects 7 columns, while model outputs are a single OT column, which would cause a dimension mismatch and unreliable metrics. Therefore **all predictions, ground truth and residuals are inverse-transformed with `scaler_y`**, and MAE / RMSE and anomaly residuals are computed in the original OT scale for interpretability.

### `input_len` and `horizon`

- `input_len`: number of past time steps used as input (default 96).
- `horizon`: number of future OT steps to predict (e.g. 24 / 48 / 96).

A sample uses the past `input_len` steps of input features to predict the next `horizon` steps of OT.

### Univariate vs multivariate input

- **Univariate**: `features = [OT]`, `num_features = 1`. Tests whether OT history alone is enough to forecast future OT.
- **Multivariate**: `features = [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]`, `num_features = 7`. Tests whether load-related covariates improve OT forecasting.

`num_features` is derived dynamically from `input_type` and never hard-coded, so downstream models must read it from the DataLoader / config.

### Sliding window output shapes

```text
X : [num_samples, input_len, num_features]
y : [num_samples, horizon]
```

Each split yields `len(split) - input_len - horizon + 1` samples; windows that would cross a split boundary are dropped by design (this prevents leakage and is not a bug).

### Evaluation interface and acceptance check

`src/evaluate.py` provides `calculate_forecasting_metrics()` and `calculate_residuals()`, both of which inverse-transform predictions and ground truth with `scaler_y` and compute MAE / RMSE / residuals in the original OT scale.

A lightweight interface check script is provided in `experiments/check_module1_pipeline.py`. It verifies that Module 1 can produce DataLoaders, metadata, `scaler_y`, and original-scale MAE / RMSE / residual calculations for the downstream forecasting and anomaly-detection modules. Run it with:

```bash
python experiments/check_module1_pipeline.py
```

> **Correlation caveat.** The EDA correlation matrix is exploratory only. High correlation does not guarantee a variable improves forecasting, and low correlation does not mean it is useless, because correlation captures only linear, non-lagged relationships. The real contribution of load covariates is evaluated later through input ablation experiments (OT-only vs. OT + all covariates).

## Project Structure

```text
data/
  raw/            # ETTh1/h2/m1/m2 CSV files
  processed/      # optional saved windows (.npz), git-ignored
notebooks/
  01_eda.ipynb            # EDA + data-quality checks + split/scaling demo
  02_data_pipeline.ipynb  # end-to-end config-driven pipeline
src/
  data/
    loader.py        # load ETT CSV, parse dates, quality checks
    splitter.py      # chronological train/val/test split
    preprocessing.py # scaler_x / scaler_y, feature selection, inverse transform
    dataset.py       # sliding windows (with y_dates), ETTDataset, DataLoaders
    pipeline.py      # build_data_pipeline(): unified entry + metadata
  models/
    base.py          # BaseForecaster
    naive.py         # Naive / persistence baseline
    linear.py        # plain linear baseline
    nlinear.py       # NLinear-style baseline (last-value normalization)
    dlinear.py       # decomposition-based DLinear (trend + seasonal, +channel-independent)
    lstm.py          # LSTM forecaster
    transformer.py   # Transformer forecaster + positional encoding
    factory.py       # build_model() + validate_model_config() + MODEL_REGISTRY
    utils.py         # count_parameters(), get_model_summary()
  utils/
    seed.py          # global random seed
    device.py        # CPU / CUDA / MPS selection
    config.py        # YAML config loading + validate_config()
configs/
  ETTh1_multivariate_h24.yaml
  ETTh1_univariate_h24.yaml
experiments/
  check_module1_pipeline.py  # Module 1 interface acceptance check
tests/
  test_data_pipeline.py      # minimal pytest suite for Module 1
results/
  figures/          # EDA figures
  metrics/          # summary statistics, metrics, pipeline metadata JSON
```

### Unified pipeline entry

`src/data/pipeline.py` exposes `build_data_pipeline(config)`, which runs the
whole leakage-free flow and returns ready-to-use DataLoaders, scalers, metadata
and per-window target dates. Downstream modules only need:

```python
from src.utils.config import load_config
from src.data.pipeline import build_data_pipeline

config = load_config("configs/ETTh1_multivariate_h24.yaml")
data = build_data_pipeline(config, save_metadata=True)

train_loader = data["train_loader"]
val_loader = data["val_loader"]
test_loader = data["test_loader"]
num_features = data["num_features"]   # model input_dim
horizon = data["horizon"]             # model output_dim
scaler_y = data["scaler_y"]           # original-scale metrics
```

The config is validated up front (`validate_config()` checks input_type, split
ratios, positive window/batch sizes and dataset path). Each window also carries
`y_dates` (shape `[num_samples, horizon]`) so predictions, residuals and
anomalies can be plotted against real timestamps. Pipeline metadata
(sample counts, features, horizon, ...) is saved to `results/metrics/` for
reproducibility.

Run the tests with:

```bash
pytest tests/
```

To batch-generate experiment configs for Module 3:

```bash
python experiments/generate_configs.py
```

This writes one YAML per `dataset x input_type x input_len x horizon x seed`
combination into `configs/generated/` (git-ignored, regenerate on demand).

### Data Contract

Module 1 guarantees the following interface to downstream modules:

| Name | Shape / type | Meaning |
| --- | --- | --- |
| `X` (batch) | `[batch_size, input_len, num_features]` | Model input window |
| `y` (batch) | `[batch_size, horizon]` | Forecast target (OT, scaled) |
| `y_dates` | `[num_samples, horizon]` | Real timestamp of every target step |
| `num_features` | `int` (1 univariate / 7 multivariate) | Model `input_dim` |
| `horizon` | `int` | Model `output_dim` |
| `feature_cols` | `list[str]` | Ordered input feature names |
| `input_type` | `str` | `"univariate"` or `"multivariate"` |
| `scaler_y` | fitted scaler | Inverse-transform predictions to original OT scale |
| `train_loader` / `val_loader` / `test_loader` | `DataLoader` | Batched samples (train shuffled; val/test not) |

Module 2 (models) reads `num_features` and `horizon`; Module 3 (training) reads
the three DataLoaders; Modules 3/4 use `scaler_y` for original-scale MAE / RMSE /
residuals and `y_dates` for time-aligned plots. All values are produced by
`build_data_pipeline(config)`.

## Module 2: Forecasting Model Library

Module 2 implements a unified forecasting model library spanning models of
increasing capacity: Naive, Linear, NLinear, DLinear, LSTM and Transformer
forecasters. All models follow the same input-output contract, taking windowed
ETT sequences with shape `[batch_size, input_len, num_features]` and producing
multi-horizon OT forecasts with shape `[batch_size, horizon]`. Module 2 only
defines the models; training, metrics and anomaly detection belong to later
modules.

The forecasting library compares models with increasing modelling assumptions
and capacity: a non-parametric persistence baseline, direct and normalized
linear models, a decomposition-based linear model, a recurrent neural network,
and an attention-based Transformer encoder. This supports the research question
of whether attention-based models are consistently better than simpler linear
and recurrent baselines for ETT oil-temperature forecasting (cf. Zeng et al.,
AAAI 2023).

### Models

| Model | File | Idea |
| --- | --- | --- |
| `naive` | `src/models/naive.py` | Repeat the last input OT value across the horizon (persistence sanity check, no training). |
| `linear` | `src/models/linear.py` | Flatten the window and map it to the horizon with one linear layer (plain linear baseline). |
| `nlinear` | `src/models/nlinear.py` | Subtract the last value, predict the change with a linear layer, add the last OT value back (NLinear-inspired, robust to level shifts). |
| `dlinear` | `src/models/dlinear.py` | Decompose into trend (moving average) + seasonal and project each to the horizon. Channel-mixing by default; `channel_independent: true` uses per-channel temporal projection plus a linear mixing head. |
| `lstm` | `src/models/lstm.py` | LSTM encoder, last hidden state projected to the horizon. |
| `transformer` | `src/models/transformer.py` | Input projection + sinusoidal positional encoding + self-attention encoder + pooling + linear head. Supports `forward(x, return_attention=True)` for exploratory RQ4 analysis. |

Models are registered in `MODEL_REGISTRY` (`src/models/factory.py`), the single
source of truth for supported names. Each model also exposes metadata
attributes (`model_type`, `supports_attention`, `supports_multivariate`,
`requires_feature_cols`, `description`) surfaced by `get_model_summary()`.

All models subclass `BaseForecaster` (`src/models/base.py`) and read
`num_features` / `horizon` dynamically (never hard-coded). The Naive baseline
resolves the OT column index from `feature_cols`, so it works for both
univariate and multivariate inputs.

> **Note on DLinear.** `DLinearForecaster` is a decomposition-based,
> DLinear-inspired forecaster: it splits the input into trend and seasonal
> components and projects each to the target horizon using the flattened
> multivariate window (variant A, shared/flattened). It is not a full
> reproduction of the original paper; a channel-independent variant could be
> added later if required.

### Model factory and config

A `model` section selects the model and its hyper-parameters:

```yaml
# Naive
model:
  name: naive

# Linear
model:
  name: linear

# NLinear
model:
  name: nlinear

# DLinear (decomposition)
model:
  name: dlinear
  kernel_size: 25            # moving-average window for trend extraction
  channel_independent: false # true -> per-channel temporal projection + mixing head

# LSTM
model:
  name: lstm
  hidden_dim: 64
  num_layers: 2
  dropout: 0.2

# Transformer
model:
  name: transformer
  d_model: 64       # must be divisible by nhead
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.1
  pooling: last     # "last" or "mean"
```

Module 3 builds any model in one call:

```python
from src.models import build_model, get_model_summary

model = build_model(config, num_features=data["num_features"],
                    feature_cols=data["feature_cols"])
y_pred = model(x)   # [batch_size, input_len, num_features] -> [batch_size, horizon]
summary = get_model_summary(model, model_name=config["model"]["name"])
# -> {"model_name", "class_name", "total_parameters", "trainable_parameters",
#     "input_len", "num_features", "horizon"}
```

### Model Configuration Contract

Every model config must contain a `model.name` field. Supported names:
`naive`, `linear`, `nlinear`, `dlinear`, `lstm`, `transformer`. `build_model()`
calls `validate_model_config()` first, so an invalid config fails at build time
rather than mid-training. Model-specific rules:

- **dlinear**: `kernel_size` must be a positive odd integer (default 25);
  `channel_independent` must be a boolean (default false).
- **lstm**: `hidden_dim > 0`, `num_layers > 0`, `0 <= dropout < 1`
  (dropout has no effect when `num_layers == 1`).
- **transformer**: `d_model > 0`, `nhead > 0`, `num_layers > 0`,
  `dim_feedforward > 0`, `0 <= dropout < 1`, `d_model % nhead == 0`,
  `pooling in {"last", "mean"}`.

`get_model_summary(model)` returns a serialisable dict (parameter counts +
dimensions) for Module 3 experiment logs.

Model utilities, forward-shape, factory and config-validation tests live in
`tests/test_models.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the EDA notebook (`notebooks/01_eda.ipynb`) for data-quality checks and figures, then `notebooks/02_data_pipeline.ipynb` to build the config-driven, leakage-free DataLoaders consumed by the forecasting models.
