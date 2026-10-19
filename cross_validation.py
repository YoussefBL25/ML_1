"""Stratified validation with training-only preprocessing and cutoff tuning."""

import numpy as np

from metrics import f1_score
from preprocess import Preprocessor, binary_labels
from threshold import THRESHOLDS, best_threshold


def stratified_k_fold(y, k=5, seed=0):
    """Return deterministic, disjoint folds containing both classes."""
    y = binary_labels(y)
    labels, counts = np.unique(y, return_counts=True)
    if (not isinstance(k, (int, np.integer)) or isinstance(k, bool) or
            k < 2 or len(labels) != 2 or np.min(counts) < k):
        raise ValueError("k must be >= 2, with at least k examples of each class.")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in labels:
        idx = rng.permutation(np.flatnonzero(y == label))
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def validation_data(y, X):
    """Validate raw features and binary labels without imputing or scaling."""
    y = binary_labels(y)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] == 0:
        raise ValueError("Features must have shape (len(y), D) with D > 0.")
    if np.any(np.isinf(X)):
        raise ValueError("Features contain infinity.")
    return y, X


def checked_scores(score_fn, X, w):
    scores = np.asarray(score_fn(X, w), dtype=float)
    if scores.shape != (len(X),) or not np.all(np.isfinite(scores)):
        raise ValueError("Model scores must be finite and one-dimensional.")
    return scores


def out_of_fold_scores(y, X, train_fn, score_fn, params, k=5, seed=0,
                       preprocessor_factory=Preprocessor):
    """Produce selection scores, fitting preprocessing on each training fold."""
    y, X = validation_data(y, X)
    scores = np.empty(len(y))
    for val_idx in stratified_k_fold(y, k, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        processor = preprocessor_factory()
        train_X = processor.fit_transform(X[train_mask])
        val_X = processor.transform(X[val_idx])
        w = train_fn(y[train_mask], train_X, params)
        scores[val_idx] = checked_scores(score_fn, val_X, w)
    return scores


def cross_validate(y, X, train_fn, score_fn, params, k=5, seed=0,
                   tune_threshold=True, preprocessor_factory=Preprocessor,
                   inner_k=3, thresholds=THRESHOLDS):
    """Return (deployment cutoff, mean F1, std F1) for fixed parameters.

    Each outer fold's cutoff is chosen using inner out-of-fold predictions from
    its training partition only. Outer labels are used solely for evaluation.
    The returned deployment cutoff is separately selected from the pooled outer
    scores; reported F1 uses the individual training-selected cutoffs, not that
    pooled cutoff. Hyperparameter selection requires nested_grid_search.
    Supply raw features, not a matrix preprocessed on the full dataset.
    """
    y, X = validation_data(y, X)
    scores, f1s = np.empty(len(y)), []
    for val_idx in stratified_k_fold(y, k, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_y, train_X = y[train_mask], X[train_mask]
        cutoff = 0.5
        if tune_threshold:
            inner_scores = out_of_fold_scores(
                train_y, train_X, train_fn, score_fn, params, inner_k, seed,
                preprocessor_factory)
            cutoff = best_threshold(train_y, inner_scores, thresholds)[0]
        processor = preprocessor_factory()
        tx = processor.fit_transform(train_X)
        w = train_fn(train_y, tx, params)
        scores[val_idx] = checked_scores(score_fn, processor.transform(X[val_idx]), w)
        f1s.append(f1_score(y[val_idx], (scores[val_idx] >= cutoff).astype(int)))
    deployment_cutoff = best_threshold(y, scores, thresholds)[0] if tune_threshold else 0.5
    return float(deployment_cutoff), float(np.mean(f1s)), float(np.std(f1s))
