"""F1 score for binary labels in {0, 1}."""

import numpy as np


def f1_score(y_true, y_pred):
    """F1 = 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall.

    Returns 0 when there are no true and no predicted positives.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels and predictions must be vectors of equal length.")
    if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("F1 expects binary labels and predictions in {0, 1}.")
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0
