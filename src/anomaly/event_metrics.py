"""Event-wise anomaly detection metrics (Module 6.7).

Complements the point-wise metrics (Module 6.4) with a monitoring-oriented view:
was each anomaly EVENT (a contiguous injected segment) detected at all, and how
quickly? A true event counts as detected if any point inside its segment is
flagged. Detection delay is measured from the segment start to the first flagged
point. False positives in normal regions do not affect event recall (this is
intentional; point-wise precision / FPR already capture false alarms).
"""

import numpy as np
import pandas as pd


def calculate_event_detection_metrics(
    df: pd.DataFrame,
    label_col: str = "is_anomaly",
    pred_col: str = "predicted_anomaly",
    segment_col: str = "anomaly_segment_id",
) -> dict:
    """Compute event-wise recall and detection-delay statistics.

    A true event is a group of anomalous rows sharing an ``anomaly_segment_id``.
    Rows are assumed to be in temporal order (as produced by Module 6.1).

    Returns
    -------
    dict
        num_true_events, num_detected_events, event_recall,
        mean_detection_delay, median_detection_delay, max_detection_delay.
        Delay statistics are NaN when no event was detected.
    """
    for col in (label_col, pred_col, segment_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if len(df) == 0:
        raise ValueError("dataframe must not be empty.")
    if df[pred_col].isna().any():
        raise ValueError(f"Column '{pred_col}' contains NaN.")

    labels = df[label_col].to_numpy().astype(bool)
    preds = df[pred_col].to_numpy().astype(bool)
    seg = df[segment_col].to_numpy()

    event_ids = np.unique(seg[labels])  # segment ids of true anomaly events
    num_true = int(len(event_ids))

    num_detected = 0
    delays = []
    for eid in event_ids:
        idx = np.where((seg == eid) & labels)[0]
        start = idx.min()
        detected_idx = idx[preds[idx]]
        if detected_idx.size > 0:
            num_detected += 1
            delays.append(int(detected_idx.min() - start))

    event_recall = float(num_detected / num_true) if num_true > 0 else 0.0
    if delays:
        mean_delay = float(np.mean(delays))
        median_delay = float(np.median(delays))
        max_delay = int(np.max(delays))
    else:
        mean_delay = median_delay = max_delay = float("nan")

    return {
        "num_true_events": num_true,
        "num_detected_events": int(num_detected),
        "event_recall": event_recall,
        "mean_detection_delay": mean_delay,
        "median_detection_delay": median_delay,
        "max_detection_delay": max_delay,
    }
