"""Grid selection and nested evaluation of model/hyperparameter/cutoff choices."""

import numpy as np

from cross_validation import (
    cross_validate, stratified_k_fold, validation_data,
)
from metrics import f1_score
from preprocess import Preprocessor
from threshold import THRESHOLDS, best_threshold


def grid_search(y, X, train_fn, score_fn, gammas, lambdas=(0.0,), max_iters=200,
                k=5, seed=0, tune_threshold=True, preprocessor_factory=Preprocessor,
                inner_k=3):
    """Compatibility grid search; its best row is a selection score, not a test score."""
    results = []
    for gamma in gammas:
        for lambda_ in lambdas:
            params = {"gamma": gamma, "lambda_": lambda_, "max_iters": max_iters}
            cutoff, mean_f1, std_f1 = cross_validate(
                y, X, train_fn, score_fn, params, k, seed, tune_threshold,
                preprocessor_factory, inner_k)
            results.append({"params": params, "threshold": cutoff,
                            "f1": mean_f1, "std": std_f1})
    if not results:
        raise ValueError("Hyperparameter grids must not be empty.")
    return max(results, key=lambda row: row["f1"]), results


def select_candidate(y, X, candidates, k=3, seed=0, thresholds=THRESHOLDS):
    """Select model, preprocessing, hyperparameters, and cutoff using CV.

    Selection F1 is intentionally labeled as such. nested_grid_search evaluates
    the complete selection procedure on separate outer folds. Candidate dicts
    use the same schema as config/model.json.
    """
    from models import score_model, train_model, validate_config

    y, X = validation_data(y, X)
    candidates = [validate_config(c) for c in candidates]
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    scores = np.empty((len(candidates), len(y)))
    folds = stratified_k_fold(y, k, seed)
    for val_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        # Keep at most one transformed fold in memory, reusing its preprocessing
        # for adjacent candidates with the same feature settings.
        previous_settings = None
        for i, candidate in enumerate(candidates):
            settings = candidate["preprocessing"]
            if settings != previous_settings:
                processor = Preprocessor(**settings)
                train_X = processor.fit_transform(X[train_mask])
                val_X = processor.transform(X[val_idx])
                previous_settings = settings
            w, _ = train_model(candidate, y[train_mask], train_X)
            scores[i, val_idx] = score_model(candidate, val_X, w)
    rows = []
    for candidate, candidate_scores in zip(candidates, scores):
        cutoff, selection_f1 = best_threshold(y, candidate_scores, thresholds)
        config = dict(candidate, threshold=cutoff)
        fold_f1 = [f1_score(y[idx], (candidate_scores[idx] >= cutoff).astype(int))
                   for idx in folds]
        rows.append({"config": config, "selection_f1": selection_f1,
                     "selection_fold_std": float(np.std(fold_f1))})
    return max(rows, key=lambda row: row["selection_f1"]), rows


def nested_grid_search(y, X, candidates, k=5, inner_k=3, seed=0,
                       thresholds=THRESHOLDS):
    """Evaluate the entire selection procedure, then select a deployment config.

    Every outer fold independently selects all candidate settings and its cutoff
    using only its own training partition. Final full-data selection scores are
    separate from the outer estimate. Candidates must be fixed before looking
    at outer results; repeatedly adjusting them can still overfit this estimate.
    """
    from models import score_model, train_model

    y, X = validation_data(y, X)
    candidates = list(candidates)
    folds = stratified_k_fold(y, k, seed)
    # Fail before expensive training if the inner partitions are too small.
    for val_idx in folds:
        stratified_k_fold(np.delete(y, val_idx), inner_k, seed)
    outer = []
    for fold, val_idx in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        best, _ = select_candidate(y[train_mask], X[train_mask], candidates,
                                   inner_k, seed, thresholds)
        config = best["config"]
        processor = Preprocessor(**config["preprocessing"])
        train_X = processor.fit_transform(X[train_mask])
        w, _ = train_model(config, y[train_mask], train_X)
        scores = score_model(config, processor.transform(X[val_idx]), w)
        f1 = f1_score(y[val_idx], (scores >= config["threshold"]).astype(int))
        outer.append({"fold": fold, "f1": f1, "config": config})
        print(f"Outer fold {fold + 1}/{k}: F1={f1:.4f}", flush=True)
    best, rows = select_candidate(y, X, candidates, inner_k, seed, thresholds)
    return {"protocol": "nested_stratified_cv", "k": k, "inner_k": inner_k,
            "seed": seed, "thresholds": np.asarray(thresholds).tolist(),
            "outer_mean_f1": float(np.mean([row["f1"] for row in outer])),
            "outer_std_f1": float(np.std([row["f1"] for row in outer])),
            "outer_folds": outer, "best": best, "selection_results": rows}
