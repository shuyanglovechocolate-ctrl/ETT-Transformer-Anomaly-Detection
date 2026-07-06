# ETT Transformer Oil Temperature Forecasting and Anomaly Detection

## Project Overview

This project builds a unified, reproducible framework that connects **transformer
oil-temperature (OT) forecasting** on the Electricity Transformer Temperature (ETT)
datasets with **residual-based synthetic anomaly detection**. It systematically
compares strong simple baselines (Naive, Linear, NLinear, DLinear) against a
recurrent model (LSTM) and an attention-based model (Transformer), then reuses the
resulting forecast residuals to detect injected anomalies and analyses where this
residual signal is effective and where it fails.

## Research Question

> How effectively can a unified forecasting-residual framework predict transformer
> oil temperature and detect synthetic anomalies on ETT datasets, while assessing
> whether complex deep-learning models outperform simpler linear baselines?

## Research Gap and Contribution

Although ETT datasets are widely used for time-series forecasting, there remains
room for a focused and reproducible study that **connects OT forecasting with
residual-based anomaly detection**, while comparing strong simple baselines against
recurrent and Transformer models under a single leakage-free protocol.

This project contributes:

- a leakage-free, config-driven data and training pipeline (Modules 1–2);
- a systematic forecasting comparison with per-horizon analysis and paired
  bootstrap significance testing (Module 3);
- a residual-based synthetic anomaly-detection framework with causal statistical
  baselines, multi-seed robustness, event-wise evaluation and diagnostics
  (Module 4).

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

## Methodology

The project is organised into four modules, each with pytest coverage and
committed result snapshots:

- **Module 1 — Data Pipeline:** leakage-free chronological split, train-only
  scaling, sliding windows, PyTorch DataLoaders.
- **Module 2 — Forecasting Model Library:** Naive, Linear, NLinear, DLinear, LSTM
  and Transformer under one input-output contract.
- **Module 3 — Forecasting Experiments:** batch experiment matrix, early stopping,
  per-horizon and paired-bootstrap significance analysis, best-model selection.
- **Module 4 — Synthetic Anomaly Detection:** residual aggregation, anomaly
  injection, validation-based thresholding, detection, point-wise and event-wise
  evaluation, causal baselines and diagnostics.

Each module is detailed below.

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
data/raw/                     # ETTh1/h2/m1/m2 CSV files
notebooks/                    # 01_eda.ipynb, 02_data_pipeline.ipynb
src/
  data/                       # Module 1: loader, splitter, preprocessing, dataset, pipeline
  models/                     # Module 2: base, naive, linear, nlinear, dlinear, lstm,
                              #           transformer, factory, utils
  training/                   # Module 3: trainer, evaluator, predictor, checkpoint,
                              #           early_stopping, experiment, plots
  anomaly/                    # Module 4: residuals, injection, thresholds, detector,
                              #           metrics, event_metrics, baselines, diagnostics, plots
  evaluate.py                 # original-scale MAE / RMSE / residuals
configs/                      # experiment configs (+ generated/, git-ignored)
experiments/                  # runnable scripts (matrix runners, summaries, diagnostics)
tests/                        # pytest suites for all four modules
results/
  metrics/                    # forecasting summary tables (committed)
  figures/                    # figures (mostly git-ignored; representative ones kept)
  anomaly/metrics/            # anomaly result tables + summaries (committed)
  anomaly/figures/            # representative anomaly figures (committed)
  checkpoints/ predictions/ logs/   # heavy artifacts (git-ignored)
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
| `transformer` | `src/models/transformer.py` | Input projection + sinusoidal positional encoding + self-attention encoder + pooling + linear head. Supports `forward(x, return_attention=True)` for exploratory attention analysis. |

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

## Module 3: Forecasting Experiments

Module 3 turns the pipeline and model library into a reproducible experiment
system. A single experiment is run with:

```bash
python src/train.py --config configs/ETTh1_multivariate_h24.yaml
```

It trains with **early stopping**, **gradient clipping** and an optional LR
scheduler, restores the **best-validation checkpoint** before testing, computes
MAE / RMSE / WAPE in the **original OT scale**, and records everything (metrics
JSON, prediction CSV with `residual` and `target_date`, config snapshot, and a
row in `results/metrics/experiment_log.csv`).

Batch experiments are run through a matrix runner (fail-soft, `--skip-existing`):

```bash
python experiments/run_matrix.py --matrix core-light   # 4 light models x 2 datasets x 2 input types x 3 horizons x 3 seeds
python experiments/run_matrix.py --matrix core-deep     # LSTM/Transformer, multivariate
python experiments/summarize_results.py                 # comparison tables (deduped by experiment_id)
python experiments/analyze_forecasting_results.py       # per-horizon metrics, best model, bootstrap significance
python experiments/validate_results.py                  # completeness / consistency checks
```

The full study covers **192 forecasting runs** (core-light 144 + core-deep 36 +
a regularized deep-model robustness check 12).

## Module 4: Synthetic Anomaly Detection

Module 4 reuses Module 3 forecast residuals to detect injected anomalies, without
retraining any model. The pipeline is deliberately **leakage-free**:

```text
validation residual  -> threshold (percentile / mean+kstd / IQR / MAD)
test residual        -> inject anomaly into y_true (y_pred unchanged)
                     -> recompute anomalous residual -> detect (score > threshold)
                     -> point-wise and event-wise evaluation
```

Anomalies are injected only into `y_true` (spike / level_shift / frozen), so the
anomalous residual `y_true_anomalous - y_pred_clean` grows where anomalies occur.
Thresholds are learned **only from validation residuals**. Three causal
statistical baselines (`raw_zscore`, `diff_score`, `rolling_zscore`) are compared
against the residual detector under an identical protocol.

```bash
python experiments/prepare_anomaly_residuals.py    # validation inference + residual aggregation (6.1)
python experiments/run_anomaly_detection.py        # detector x threshold x anomaly x injection-seed matrix
python experiments/summarize_anomaly_results.py    # detector / type / threshold summaries
python experiments/run_magnitude_sensitivity.py    # F1 vs anomaly magnitude
python experiments/diagnose_anomaly_residuals.py   # residual + frozen-flatness diagnostics
```

## Key Results

### Forecasting (Module 3)

Linear-family models, especially **NLinear and DLinear, consistently outperform
LSTM and Transformer** under the tested ETT settings. Paired bootstrap confidence
intervals over per-window absolute errors support this: linear-family models
significantly beat the deep models in **all tested multivariate comparisons**
(36/36, non-overlapping 95% CIs). A regularized robustness check (lower LR, weight
decay, higher dropout) reduced deep-model variance but did **not** change the
ranking. Adding load covariates (multivariate) did not consistently improve OT
forecasting over univariate input.

### Anomaly detection (Module 4)

The residual-based detector **outperformed the causal statistical baselines** across
synthetic anomaly types and injection seeds. It was particularly effective for
**spike and level-shift** anomalies (best F1 ≈ 0.86 and 0.90), while **frozen-value**
anomalies remained challenging (point-wise F1 ≈ 0.43). Event-wise evaluation showed
partial monitoring usefulness for frozen events (event recall 0.18 → 0.43, with a
mean detection delay of ~6 steps), and flatness diagnostics indicated that
sensor-stuck behaviours are **better characterised by temporal flatness than by
residual magnitude** (anomaly/normal separation ratio ≈ 23× vs ≈ 1–4× for residual
score). Overall, forecast residuals provide a useful but **not universal** anomaly
signal.

## Setup

```bash
pip install -r requirements.txt
```

Tests (fast; synthetic data where possible):

```bash
pytest
```

## How to Reproduce

```bash
# 1. Module 1 sanity check
python experiments/check_module1_pipeline.py

# 2. Module 3 forecasting experiments + analysis
python experiments/run_matrix.py --matrix core-light --skip-existing
python experiments/run_matrix.py --matrix core-deep  --skip-existing
python experiments/run_matrix.py --matrix robustness-deep --skip-existing
python experiments/summarize_results.py
python experiments/analyze_forecasting_results.py

# 3. Module 4 anomaly detection
python experiments/prepare_anomaly_residuals.py
python experiments/run_anomaly_detection.py
python experiments/summarize_anomaly_results.py
python experiments/run_magnitude_sensitivity.py
python experiments/diagnose_anomaly_residuals.py
```

Heavy artifacts (checkpoints, per-run predictions, logs, most figures) are
git-ignored and regenerated by the scripts above.

## Main Result Files

Forecasting:

```text
results/metrics/model_comparison.csv
results/metrics/per_horizon_summary.csv
results/metrics/model_significance_tests.csv
results/metrics/best_model_by_dataset_horizon.csv
```

Anomaly detection:

```text
results/anomaly/metrics/anomaly_detection_results_v3.csv
results/anomaly/metrics/anomaly_summary_by_detector.csv
results/anomaly/metrics/anomaly_event_summary_by_type.csv
results/anomaly/metrics/anomaly_magnitude_sensitivity.csv
results/anomaly/metrics/frozen_flatness_diagnostics.csv
results/anomaly/metrics/residual_diagnostics.csv
```

## Limitations and Future Work

- **Synthetic anomalies.** ETT has no ground-truth anomaly labels, so anomalies are
  injected synthetically. Results describe detectability of controlled anomaly types,
  not real-world faults.
- **Frozen anomalies need extra features.** Residual magnitude alone is insufficient
  for sensor-stuck behaviours; a temporal-flatness feature is a natural complement.
- **Scope.** The core study uses ETTh1/ETTh2 at `input_len=96`. Future work could add
  minute-level ETTm1/ETTm2 (frequency effect), an `input_len` ablation, more seeds,
  training-time/efficiency benchmarks, and a full decomposition/channel-independent
  DLinear or patch-based Transformer.
