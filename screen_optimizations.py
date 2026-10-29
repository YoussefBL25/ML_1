"""Screen convergence and threshold resolution without changing the final model.

Uses a stratified 60/20/20 training/threshold-selection/test split. This is an
exploratory comparison, not the five-fold estimate of a selection procedure.
The initial feature ranking stays fixed at the configured number of iterations;
only the final fit on interaction features is continued to later checkpoints.
"""

import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np

from cross_validation import stratified_k_fold
from data import DATA_PATH, ROOT, load_data
from interactions import BEST_CONFIG, load_best_config, log_loss, model_from_config, sigmoid
from metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess import Preprocessor, binary_labels
from threshold import THRESHOLDS, best_threshold


def metrics(y, scores, threshold):
    prediction = (scores >= threshold).astype(int)
    return {"f1": f1_score(y, prediction),
            "precision": precision_score(y, prediction),
            "recall": recall_score(y, prediction),
            "accuracy": accuracy_score(y, prediction),
            "log_loss": log_loss(y, scores)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--config", type=Path, default=BEST_CONFIG)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "optimization_screen.json")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = load_best_config(args.config)
    X, X_test, y, train_ids, test_ids = load_data(args.data_dir)
    del X_test, train_ids, test_ids
    y = binary_labels(y)
    folds = stratified_k_fold(y, 5, args.seed)
    test, tuning = folds[0], folds[1]
    train = np.concatenate(folds[2:])
    print(f"Split: {len(train)} train / {len(tuning)} threshold / {len(test)} test", flush=True)
    processor = Preprocessor(**config["preprocessing"])
    train_X = processor.fit_transform(X[train]).astype(np.float32)
    tuning_X = processor.transform(X[tuning]).astype(np.float32)
    test_X = processor.transform(X[test]).astype(np.float32)
    del X
    y_train, y_tuning, y_test = y[train], y[tuning], y[test]
    print("Fitting the current model as the paired baseline.", flush=True)
    model = model_from_config(config).fit(y_train, train_X)
    train_Z, tuning_Z, test_Z = (model._products(values) for values in (train_X, tuning_X, test_X))
    del train_X, tuning_X, test_X
    original_steps = config["params"]["max_iters"]
    snapshots = []
    w = model.w_.copy()
    train_labels = y_train.astype(np.float32)
    for iteration in range(original_steps, 3 * original_steps + 1):
        if iteration in (original_steps, 2 * original_steps, 3 * original_steps):
            tuning_scores = sigmoid(tuning_Z @ w)
            test_scores = sigmoid(test_Z @ w)
            snapshots.append((iteration, tuning_scores, test_scores,
                              log_loss(y_train, sigmoid(train_Z @ w))))
            print(f"Completed final-fit checkpoint {iteration}; train loss {snapshots[-1][3]:.6f}",
                  flush=True)
        if iteration < 3 * original_steps:
            with np.errstate(over="raise", invalid="raise"):
                w -= model.gamma * (train_Z.T @ (sigmoid(train_Z @ w) - train_labels) /
                                    len(y_train) + 2 * model.lambda_ * w)
            if not np.all(np.isfinite(w)):
                raise ValueError("Continuation produced nonfinite weights.")
    rows = []
    for iteration, tuning_scores, test_scores, train_loss in snapshots:
        for name, grid in (("original", THRESHOLDS), ("fine", np.linspace(0.10, 0.35, 251))):
            threshold, tuning_f1 = best_threshold(y_tuning, tuning_scores, grid)
            row = {"final_fit_iterations": iteration, "threshold_grid": name,
                   "threshold": threshold, "tuning_f1": tuning_f1,
                   "train_loss": train_loss, "test": metrics(y_test, test_scores, threshold)}
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    # Select using tuning labels only; test values are diagnostic screening scores.
    chosen = max(rows, key=lambda row: row["tuning_f1"])
    baseline = rows[0]
    out = {"protocol": "exploratory_stratified_train_tuning_test_split",
           "seed": args.seed, "split_sizes": {"train": len(train), "tuning": len(tuning),
                                               "test": len(test)},
           "config": config, "ranking_iterations": original_steps,
           "selection_scope": "final_fit_iterations_and_threshold_grid",
           "threshold_grids": {"original": THRESHOLDS.tolist(),
                               "fine": np.linspace(0.10, 0.35, 251).tolist()},
           "rows": rows, "selected_by_tuning": chosen,
           "selected_test_f1_gain_over_baseline": chosen["test"]["f1"] - baseline["test"]["f1"],
           "dataset_directory": str(args.data_dir.resolve()),
           "environment": {"python": platform.python_version(), "numpy": np.__version__},
           "elapsed_s": time.perf_counter() - started}
    payload = json.dumps(out, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
