"""Classification metrics for binary labels in {0, 1}."""

import numpy as np


def confusion_counts(y_true, y_pred):
    """Return (TP, FP, FN, TN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels and predictions must be vectors of equal length.")
    if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("Metrics expect binary labels and predictions in {0, 1}.")
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, fn, len(y_true) - tp - fp - fn


def f1_score(y_true, y_pred):
    """F1 = 2 TP / (2 TP + FP + FN).
    """
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def precision_score(y_true, y_pred):
    """Precision = TP / (TP + FP)."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if tp + fp > 0 else 0.0


def recall_score(y_true, y_pred):
    """Recall = TP / (TP + FN)."""
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if tp + fn > 0 else 0.0


def accuracy_score(y_true, y_pred):
    """Accuracy = (TP + TN) / N."""
    tp, _, _, tn = confusion_counts(y_true, y_pred)
    return float((tp + tn) / len(np.asarray(y_true)))
