export type ComparisonRow = {
  dataset: string;
  model: string;
  input_type: "univariate" | "multivariate";
  horizon: number;
  mae: number;
  mae_std: number;
  rmse: number;
  rmse_std: number;
  wape: number;
  wape_std: number;
  num_runs: number;
};

export type PredIndexEntry = { key: string; dataset: string; model: string };

export type PredSeries = {
  dataset: string;
  model: string;
  horizon: number;
  dates: string[];
  y_true: (number | null)[];
  y_pred: (number | null)[];
};

export type AnomalyData = {
  by_type: {
    anomaly_type: string;
    best_detector: string;
    best_mean_f1: number;
    mean_recall: number;
    mean_precision: number;
  }[];
  threshold_free: {
    detector_type: string;
    anomaly_type: string;
    pr_auc: number;
    pr_auc_std: number;
    roc_auc: number;
    best_f1: number;
    num_runs: number;
  }[];
  accuracy_vs_detection: {
    dataset: string;
    model: string;
    horizon: number;
    anomaly_type: string;
    forecast_mae: number;
    average_precision: number;
    event_recall: number;
  }[];
  magnitude_sensitivity: {
    anomaly_type: string;
    magnitude_scale: number;
    f1: number;
  }[];
  stress_test: {
    anomaly_type: string;
    detector_type: string;
    pr_auc: number | null;
    roc_auc: number | null;
    best_f1: number | null;
    event_recall: number | null;
    detection_delay: number | null;
  }[];
};

export type SignificanceData = {
  comparisons: {
    dataset: string;
    input_type: string;
    horizon: number;
    model_a: string;
    model_b: string;
    mae_a: number;
    mae_b: number;
    delta: number; // MAE(model_a) - MAE(model_b); negative => model_a better
    ci_low: number;
    ci_high: number;
    a_better: boolean;
    significant: boolean;
    num_points: number;
  }[];
};

export type EttmRow = {
  dataset: string;
  model: string;
  horizon: number;
  input_type: string;
  mae: number;
  mae_std: number;
  rmse: number;
  wape: number;
  params: number;
  rank: number;
};

export type InputLengthData = {
  rows: {
    model: string;
    input_len: number;
    mae: number;
    mae_std: number;
    rmse: number;
    wape: number;
    params: number;
    rank: number;
  }[];
  summary: { input_len: number; best_model: string; ranking: string }[];
};

export type EfficiencyRow = {
  dataset: string;
  model: string;
  horizon: number;
  input_type: "univariate" | "multivariate";
  mae: number;
  mae_std: number;
  params: number;
  checkpoint_mb: number | null;
  epochs: number | null;
};

export type FrozenData = {
  detectors: {
    name: string;
    label: string;
    f1: number;
    recall: number;
    precision: number;
    event_recall: number;
  }[];
  diagnosis: { flatness_ratio: number; residual_ratio: number };
  contrast: Record<string, number>;
};

export type AttentionData = {
  layers: {
    layer: number;
    peak_lag: number;
    peak_attention: number;
    entropy: number;
    max_entropy: number;
    recent_8_mass: number;
  }[];
  experiment_id: string | null;
  input_len: number;
  figures: { by_lag: string; heatmap: string };
};

export type Manifest = {
  headline: {
    best_mae: number;
    num_models: number;
    num_datasets: number;
    num_horizons: number;
    num_seeds: number;
    num_anomaly_types: number;
  };
  reproducibility: {
    compute_device: string;
    generated_at: string;
    git_commit: string;
    package_versions: Record<string, string>;
    python_version: string;
    platform: Record<string, string>;
  };
};
