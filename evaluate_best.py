"""5-fold CV of the final model, with the threshold tuned on inner folds.

Uses the same stratified folds (seed 0) as tune_models.py, so the per-fold
F1 can be compared with the earlier models. The options remove one part of
the final model at a time, for the ablation of the report: --n-top 0 drops
the interactions, --no-indicators the missing-value indicators, and
--no-one-hot the one-hot encoding. Writes results/best_model_cv.json.
Hyperparameters and the configured categorical column list remain fixed;
only the decision threshold is selected in the inner loop.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from cross_validation import stratified_k_fold, validation_data
from data import DATA_PATH, ROOT, load_data
from interactions import (BEST_CONFIG, load_best_config, log_loss,
                          model_from_config, validate_best_config)
from metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess import Preprocessor
from threshold import THRESHOLDS, best_threshold


def fit_predict(config, y, X_train, X_eval):
    processor = Preprocessor(**config["preprocessing"])
    train = processor.fit_transform(X_train)
    model = model_from_config(config).fit(y, train)
    return model, model.predict_proba(processor.transform(X_eval))


def inner_threshold(config, y, X, k=3, seed=0):
    scores = np.empty(len(y))
    for idx in stratified_k_fold(y, k, seed):
        train = np.ones(len(y), dtype=bool)
        train[idx] = False
        scores[idx] = fit_predict(config, y[train], X[train], X[idx])[1]
    return best_threshold(y, scores, THRESHOLDS)[0]


def evaluate_config(config, y, X, k=5, inner_k=3, seed=0):
    """Evaluate a fixed interaction configuration with training-only thresholds.

    This does not evaluate the search that originally chose n_top, gamma,
    lambda_, or the categorical column list stored in the configuration.
    """
    config = validate_best_config(config)
    if config["model"] != "interaction_logistic_regression":
        raise ValueError("Expected an interaction_logistic_regression configuration.")
    y, X = validation_data(y, X)
    folds = stratified_k_fold(y, k, seed)
    for idx in folds:
        stratified_k_fold(np.delete(y, idx), inner_k, seed)
    rows, start = [], time.time()
    for fold, idx in enumerate(folds):
        train = np.ones(len(y), dtype=bool)
        train[idx] = False
        threshold = inner_threshold(config, y[train], X[train], inner_k, seed)
        model, scores = fit_predict(config, y[train], X[train], X[idx])
        pred = (scores >= threshold).astype(int)
        rows.append({"fold": fold, "threshold": threshold,
                     "f1": f1_score(y[idx], pred),
                     "precision": precision_score(y[idx], pred),
                     "recall": recall_score(y[idx], pred),
                     "accuracy": accuracy_score(y[idx], pred),
                     "train_loss": model.loss_,
                     "val_loss": log_loss(y[idx], scores)})
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])),
                     "std": float(np.std([r[key] for r in rows]))}
               for key in rows[0] if key != "fold"}
    return {"protocol": "stratified_cv_with_inner_threshold_selection",
            "selection_scope": "threshold_only", "k": k, "inner_k": inner_k,
            "seed": seed, "thresholds": THRESHOLDS.tolist(),
            "config": config, "folds": rows, "summary": summary,
            "training_shape": list(X.shape),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "elapsed_s": time.time() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BEST_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-top", type=int, help="override the number of top features")
    parser.add_argument("--no-indicators", action="store_true")
    parser.add_argument("--no-one-hot", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "best_model_cv.json")
    args = parser.parse_args(argv)
    try:
        if args.config.resolve() == args.output.resolve():
            raise ValueError("--config and --output must be different files.")
        config = load_best_config(args.config)
        if args.n_top is not None:
            config["params"]["n_top"] = args.n_top
        if args.no_indicators:
            config["preprocessing"]["add_missing_indicators"] = False
        if args.no_one_hot:
            config["preprocessing"]["categorical_columns"] = []
        config = validate_best_config(config)
        x_train, _, y_train, _, _ = load_data(args.data_dir, use_cache=not args.no_cache)
        out = evaluate_config(config, y_train, x_train, args.folds, args.inner_folds, args.seed)
        out["dataset_directory"] = str(args.data_dir.resolve())
        # Serialize before opening the output, so invalid metrics cannot overwrite it.
        payload = json.dumps(out, indent=1, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for key, value in out["summary"].items():
        print(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f}")


if __name__ == "__main__":
    main()
