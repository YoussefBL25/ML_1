import numpy as np

from metrics import f1_score

THRESHOLDS = np.linspace(0.05, 0.95, 91)


def best_threshold(y_true, scores):
    """Return (threshold, F1) maximizing F1 on the given scores. Labels in {0, 1}."""
    best_t, best_f1 = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        f1 = f1_score(y_true, (scores >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1