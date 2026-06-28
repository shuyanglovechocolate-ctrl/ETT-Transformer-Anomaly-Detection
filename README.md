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

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the EDA notebook (`notebooks/01_eda.ipynb`) for data-quality checks and figures, then `notebooks/02_data_pipeline.ipynb` to build the config-driven, leakage-free DataLoaders consumed by the forecasting models.
