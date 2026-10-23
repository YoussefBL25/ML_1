"""Nested tuning for all six required models using raw CSV features.

Besides the outer F1, the output stores precision, recall and accuracy of each
outer fold's selected config, and the inner-CV F1 of every grid combination on
the full training set with the number of outer folds that selected it.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import platform
import time

import numpy as np

from cross_validation import stratified_k_fold
from data import DATA_PATH, ROOT, load_data
from metrics import accuracy_score, f1_score, precision_score, recall_score
from models import (LINEAR_MODELS, LOGISTIC_MODELS, MODEL_NAMES, score_model,
                    train_model, validate_config)
from preprocess import Preprocessor, binary_labels
from tuning import nested_grid_search


# Logistic grid, log-spaced: gamma in half decades from too slow to converge
# in 300 iterations up to beyond the step size bound 8 / lambda_max(X^T X / N)
# ~ 0.57; lambda_ in decades from negligible to strong. 300 iterations: the
# loss changes by 0.06% from 300 to 600 iterations at gamma = 0.5.
LOGISTIC_GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0)
LOGISTIC_LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
LOGISTIC_MAX_ITERS = 300


def make_candidates(methods=MODEL_NAMES, preprocessing=None, seed=0):
    """Starter grids, to be reviewed after EDA; no claim of optimality."""
    preprocessing = preprocessing or {}
    candidates = []
    for name in methods:
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        if name == "least_squares":
            grid = [{}]
        elif name == "ridge_regression":
            grid = [{"lambda_": v} for v in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
        else:
            if name == "mean_squared_error_sgd":
                gammas, iterations = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4), 200_000
            elif name == "mean_squared_error_gd":
                gammas, iterations = (0.001, 0.003, 0.01, 0.03, 0.06, 0.09), 200
            else:
                gammas, iterations = LOGISTIC_GAMMAS, LOGISTIC_MAX_ITERS
            lambdas = LOGISTIC_LAMBDAS if name == "reg_logistic_regression" else (None,)
            grid = []
            for gamma, lambda_ in product(gammas, lambdas):
                params = {"gamma": gamma, "max_iters": iterations}
                if lambda_ is not None:
                    params["lambda_"] = lambda_
                grid.append(params)
        candidates.extend(validate_config({"model": name, "params": params,
                                           "seed": seed, "preprocessing": preprocessing})
                          for params in grid)
    return candidates


def outer_metrics(y, X, result):
    """Refit each outer fold's selected config and score its held-out fold."""
    y = binary_labels(y)
    rows = []
    for outer, val_idx in zip(result["outer_folds"],
                              stratified_k_fold(y, result["k"], result["seed"])):
        config = outer["config"]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        processor = Preprocessor(**config["preprocessing"])
        w, _ = train_model(config, y[train_mask], processor.fit_transform(X[train_mask]))
        scores = score_model(config, processor.transform(X[val_idx]), w)
        y_pred = (scores >= config["threshold"]).astype(int)
        row = {name: metric(y[val_idx], y_pred) for name, metric in
               (("f1", f1_score), ("precision", precision_score),
                ("recall", recall_score), ("accuracy", accuracy_score))}
        if not np.isclose(row["f1"], outer["f1"]):
            raise ValueError("Refit outer F1 differs from the nested search.")
        rows.append(dict(row, threshold=config["threshold"]))
    return {name: {"mean": float(np.mean([r[name] for r in rows])),
                   "std": float(np.std([r[name] for r in rows]))} for name in rows[0]}


def tuning_grid(result):
    """Inner-CV F1 of every combination and how many outer folds selected it."""
    chosen = [(o["config"]["model"], o["config"]["params"]) for o in result["outer_folds"]]
    return [{"model": row["config"]["model"], "params": row["config"]["params"],
             "inner_f1": row["selection_f1"], "inner_fold_std": row["selection_fold_std"],
             "threshold": row["config"]["threshold"],
             "outer_folds_selected": chosen.count((row["config"]["model"],
                                                   row["config"]["params"]))}
            for row in result["selection_results"]]


def main(argv=None, default_methods="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--methods", default=default_methods,
                        help="all, linear, logistic, or comma-separated model names")
    parser.add_argument("--preprocessing", type=Path,
                        help="JSON object of Preprocessor settings")
    parser.add_argument("--candidates", type=Path,
                        help="JSON list of full model configs; overrides --methods/--preprocessing")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tuning_nested.json")
    parser.add_argument("--model-output", type=Path,
                        default=ROOT / "results" / "selected_model.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.model_output.resolve():
            raise ValueError("--output and --model-output must be different files.")
        if args.candidates:
            with args.candidates.open(encoding="utf-8") as stream:
                candidates = json.load(stream)
            if not isinstance(candidates, list):
                raise ValueError("Candidates must be a JSON list of model configurations.")
            candidates = [validate_config(c) for c in candidates]
        else:
            preprocessing = {}
            if args.preprocessing:
                with args.preprocessing.open(encoding="utf-8") as stream:
                    preprocessing = json.load(stream)
            groups = {"all": MODEL_NAMES, "linear": LINEAR_MODELS, "logistic": LOGISTIC_MODELS}
            methods = groups.get(args.methods, tuple(args.methods.split(",")))
            candidates = make_candidates(methods, preprocessing, args.seed)
        x_train, _, y, _, _ = load_data(args.data_dir, use_cache=not args.no_cache)
        print(f"Nested CV: {len(candidates)} candidates, {args.folds} outer / "
              f"{args.inner_folds} inner folds; this can require many training runs.", flush=True)
        start = time.perf_counter()
        result = nested_grid_search(y, x_train, candidates, args.folds,
                                    args.inner_folds, args.seed)
        result["outer_metrics"] = outer_metrics(y, x_train, result)
        result["tuning_grid"] = tuning_grid(result)
        result["elapsed_s"] = time.perf_counter() - start
        result["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
        result["dataset_directory"] = str(args.data_dir.resolve())
        result["training_shape"] = list(x_train.shape)
        for path, payload in ((args.output, result),
                              (args.model_output, result["best"]["config"])):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(f"Outer F1: {result['outer_mean_f1']:.4f} +/- {result['outer_std_f1']:.4f}")
        print("Outer " + ", ".join(f"{name} {value['mean']:.4f}"
                                   for name, value in result["outer_metrics"].items()))
        print(f"Selected config: {args.model_output}")
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
