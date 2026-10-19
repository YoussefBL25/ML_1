"""Step 4 (Person 3): time and tune GD, SGD, least squares and ridge regression.

Every method uses the same 5 stratified folds (seed 0) and, when it has a
hyperparameter, the same number of settings (6). The cutoff of each method is
chosen with threshold.py inside cross_validate. Results are written to
results/tuning_linear.json.

Usage: python tune_linear.py
"""

import json
import os
import time

import numpy as np

from data import load_data, baseline_preprocess
from implementations import (
    least_squares,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    ridge_regression,
)
from tuning import grid_search

K, SEED = 5, 0
GD_ITERS = 200
SGD_ITERS = 200_000  # single-sample steps, about 0.75 epoch of a training fold

# Grids chosen from the data: GD is stable for gamma < 2 / lambda_max(X^T X / N)
# (about 0.095 here); single SGD steps can blow up on rows with a huge norm.
GD_GAMMAS = (0.001, 0.003, 0.01, 0.03, 0.06, 0.09)
SGD_GAMMAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
RIDGE_LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def score_linear(X, w):
    """Regression output used as a score, thresholded into {0, 1}."""
    return X @ w


def train_gd(y, X, params):
    w0 = np.zeros(X.shape[1])
    return mean_squared_error_gd(y, X, w0, params["max_iters"], params["gamma"])[0]


def train_sgd(y, X, params):
    np.random.seed(SEED)
    w0 = np.zeros(X.shape[1])
    return mean_squared_error_sgd(y, X, w0, params["max_iters"], params["gamma"])[0]


def train_least_squares(y, X, params):
    return least_squares(y, X)[0]


def train_ridge(y, X, params):
    return ridge_regression(y, X, params["lambda_"])[0]


METHODS = {
    "mean_squared_error_gd": dict(
        train_fn=train_gd, gammas=GD_GAMMAS, lambdas=(0.0,), max_iters=GD_ITERS
    ),
    "mean_squared_error_sgd": dict(
        train_fn=train_sgd, gammas=SGD_GAMMAS, lambdas=(0.0,), max_iters=SGD_ITERS
    ),
    "least_squares": dict(
        train_fn=train_least_squares, gammas=(None,), lambdas=(0.0,), max_iters=None
    ),
    "ridge_regression": dict(
        train_fn=train_ridge, gammas=(None,), lambdas=RIDGE_LAMBDAS, max_iters=None
    ),
}


def main():
    x_train, x_test, y_train, _, _ = load_data()
    tx, _, y = baseline_preprocess(x_train, x_test, y_train)
    print(f"Data: {tx.shape}, positives: {y.mean():.4f}")

    summary = {}
    for name, m in METHODS.items():
        # Time one training run on the full data with the middle setting
        params = {
            "gamma": m["gammas"][len(m["gammas"]) // 2],
            "lambda_": m["lambdas"][len(m["lambdas"]) // 2],
            "max_iters": m["max_iters"],
        }
        start = time.time()
        m["train_fn"](y, tx, params)
        train_time = time.time() - start
        print(f"\n=== {name} (one training run: {train_time:.2f}s) ===")

        best, results = grid_search(
            y,
            tx,
            m["train_fn"],
            score_linear,
            m["gammas"],
            m["lambdas"],
            m["max_iters"],
            k=K,
            seed=SEED,
        )
        summary[name] = {"train_time_s": train_time, "best": best, "all": results}
        print(f"BEST {name}: {best}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "tuning_linear.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary (5-fold CV, F1 at the tuned cutoff):")
    for name, s in summary.items():
        b = s["best"]
        print(
            f"{name:24s} F1={b['f1']:.4f} +/- {b['std']:.4f}  "
            f"cutoff={b['threshold']:.2f}  params={b['params']}"
        )


if __name__ == "__main__":
    main()
