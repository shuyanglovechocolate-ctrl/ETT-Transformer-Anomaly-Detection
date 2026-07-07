"""Threshold-free anomaly detection evaluation (Module 6.9).

Anomaly detection is highly class-imbalanced, so a single validation-thresholded
F1 can be misleading. These metrics evaluate the continuous anomaly score against
the labels independently of any threshold choice: PR-AUC (primary), ROC-AUC, the
best achievable F1, and F1 at target false-positive rates.

Note: best_f1 and the target-FPR operating points use the test labels to select a
threshold, so they characterise the score's *separability* (an upper bound), not
a deployable threshold. Deployable operating points still come from validation
(Modules 6.3-6.4).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
)

_EPS = 1e-12
DEFAULT_TARGET_FPRS = (0.01, 0.02, 0.05)


def _validate(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if len(labels) == 0:
        raise ValueError("labels/scores must not be empty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite (no NaN/inf).")
    return labels, scores


def pr_auc(labels, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0 or labels.all():
        return float("nan")  # undefined without both classes
    return float(average_precision_score(labels, scores))


def roc_auc(labels, scores) -> float:
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0 or labels.all():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def best_f1_threshold(labels, scores) -> dict:
    """Best achievable F1 over all thresholds (oracle upper bound)."""
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0:
        return {"best_f1": 0.0, "best_threshold": float("nan")}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    if len(f1) == 0:
        return {"best_f1": 0.0, "best_threshold": float("nan")}
    idx = int(np.argmax(f1))
    return {"best_f1": float(f1[idx]), "best_threshold": float(thresholds[idx])}


def f1_at_target_fpr(labels, scores, target_fpr: float) -> dict:
    """Operating point with the highest recall whose FPR <= target_fpr."""
    labels, scores = _validate(labels, scores)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return {"threshold": float("nan"), "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "fpr": float("nan")}

    order = np.argsort(-scores)  # descending score
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    tp = fp = 0
    best = None
    for i in range(len(scores_sorted)):
        if labels_sorted[i]:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr <= target_fpr:
            recall = tp / n_pos
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall + _EPS)
            best = {"threshold": float(scores_sorted[i]), "precision": precision,
                    "recall": recall, "f1": f1, "fpr": fpr}
        else:
            break  # FPR is monotonic as the threshold lowers
    if best is None:
        best = {"threshold": float("inf"), "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "fpr": 0.0}
    return best


def calculate_threshold_free_metrics(labels, scores, target_fprs=DEFAULT_TARGET_FPRS) -> dict:
    """PR-AUC, ROC-AUC, best F1, and F1/recall at target FPRs."""
    out = {"pr_auc": pr_auc(labels, scores), "roc_auc": roc_auc(labels, scores)}
    bf = best_f1_threshold(labels, scores)
    out["best_f1"] = bf["best_f1"]
    out["best_f1_threshold"] = bf["best_threshold"]
    for t in target_fprs:
        r = f1_at_target_fpr(labels, scores, t)
        pct = int(round(t * 100))
        out[f"f1_at_fpr_{pct}pct"] = r["f1"]
        out[f"recall_at_fpr_{pct}pct"] = r["recall"]
    return out
