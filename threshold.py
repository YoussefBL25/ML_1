"""Decision cutoff selection on training-side validation predictions."""

import numpy as np

from preprocess import binary_labels

THRESHOLDS = np.linspace(0.05, 0.95, 91)


def best_threshold(y_true, scores, thresholds=THRESHOLDS):
    """Return (threshold, F1), breaking ties toward the lowest threshold.

    The default grid is shared by linear scores and logistic probabilities.
    Custom finite grids can cover values outside [0, 1] for linear models.
    """
    y_true = binary_labels(y_true)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if scores.shape != y_true.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite and match the label vector.")
    if thresholds.ndim != 1 or len(thresholds) == 0 or not np.all(np.isfinite(thresholds)):
        raise ValueError("Threshold grid must be a nonempty finite vector.")
    thresholds = np.unique(thresholds)
    order = np.argsort(scores)
    positives = np.r_[0, np.cumsum(y_true[order])]
    starts = np.searchsorted(scores[order], thresholds, side="left")
    tp = positives[-1] - positives[starts]
    denom = (len(scores) - starts) + positives[-1]
    f1s = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])
