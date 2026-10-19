"""Stratified k-fold cross-validation with F1 scoring.

A model is described by two functions so that every method goes through the
same folds and the same scoring:
    train_fn(y, X, params) -> w        trains on (y, X) with the given params
    score_fn(X, w) -> scores           continuous scores, thresholded into {0, 1}
"""

import numpy as np

from metrics import f1_score
from threshold import best_threshold


def stratified_k_fold(y, k=5, seed=0):
    """Split indices 0..N-1 into k folds with the same share of each label.

    Args:
        y (np.ndarray): Labels of shape (N,).
        k (int): Number of folds.
        seed (int): Random seed, so every method gets the same folds.

    Returns:
        list[np.ndarray]: k sorted arrays of validation indices.
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    # Shuffle each class separately and deal it evenly over the k folds
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def cross_validate(y, X, train_fn, score_fn, params, k=5, seed=0, tune_threshold=True):
    """Run stratified k-fold CV and return (threshold, mean_f1, std_f1).

    Every sample gets one out-of-fold score. If tune_threshold is True, the
    cutoff maximizing F1 on all out-of-fold scores is chosen with
    threshold.best_threshold, otherwise 0.5 is used. The F1 of each fold is
    then computed at that cutoff.

    Args:
        y (np.ndarray): Labels in {0, 1} of shape (N,).
        X (np.ndarray): Features of shape (N, D).
        train_fn (callable): train_fn(y, X, params) -> w.
        score_fn (callable): score_fn(X, w) -> scores of shape (N,).
        params (dict): Hyperparameters passed to train_fn.
        k (int): Number of folds.
        seed (int): Seed of the fold split.
        tune_threshold (bool): Whether to tune the cutoff.

    Returns:
        tuple[float, float, float]: (threshold, mean F1, std F1 over folds).
    """
    folds = stratified_k_fold(y, k, seed)
    scores = np.empty(len(y))
    for val_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        w = train_fn(y[train_mask], X[train_mask], params)
        scores[val_idx] = score_fn(X[val_idx], w)

    threshold = best_threshold(y, scores)[0] if tune_threshold else 0.5
    f1s = [f1_score(y[idx], (scores[idx] >= threshold).astype(int)) for idx in folds]
    return float(threshold), float(np.mean(f1s)), float(np.std(f1s))
