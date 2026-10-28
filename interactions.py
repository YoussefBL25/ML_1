"""Best logistic model: richer preprocessing plus pairwise feature interactions.

The preprocessing adds missing-value indicators and one-hot encodes every
feature with 3 to 10 observed values, on top of the BRFSS codebook rules. The
model is regularized logistic regression trained twice: a first fit ranks the
features by |weight|, then the pairwise products of the top features are
added and the model is refit. Everything is learned from training rows only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import ROOT, load_data
from preprocess import Preprocessor

BEST_CONFIG = ROOT / "config" / "best_model.json"


def low_cardinality_columns(X, settings, max_levels=10):
    """Columns with 3..max_levels distinct values after the codebook rules."""
    cleaned = Preprocessor(**settings)._clean(X)
    excluded = set(settings["drop_columns"]) | set(settings["categorical_columns"])
    columns = []
    for c in range(X.shape[1]):
        if c in excluded:
            continue
        values = cleaned[:, c]
        if 2 < len(np.unique(values[~np.isnan(values)])) <= max_levels:
            columns.append(c)
    return columns


def best_preprocessing(X, base, max_levels=10):
    settings = dict(base)
    settings["add_missing_indicators"] = True
    settings["categorical_columns"] = (list(base["categorical_columns"]) +
                                       low_cardinality_columns(X, base, max_levels))
    return settings


def sigmoid(t):
    # tanh form: no overflow for large |t|, and keeps the float32 dtype.
    return 0.5 * (1 + np.tanh(0.5 * t))


def logistic_gd(y, X, lambda_, max_iters, gamma):
    """Same updates as implementations.reg_logistic_regression, in float32.

    The interaction matrix has about 1,600 columns; float32 halves its memory
    and the cost of each gradient step.
    """
    w = np.zeros(X.shape[1], dtype=X.dtype)
    y = y.astype(X.dtype)
    for _ in range(max_iters):
        w -= gamma * (X.T @ (sigmoid(X @ w) - y) / len(y) + 2 * lambda_ * w)
    return w, log_loss(y, sigmoid(X @ w))


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class InteractionModel:
    """Regularized logistic regression with products of the top features."""

    def __init__(self, gamma=0.3, lambda_=1e-4, max_iters=300, n_top=40):
        self.gamma, self.lambda_ = gamma, lambda_
        self.max_iters, self.n_top = max_iters, n_top

    def _products(self, X):
        P = X[:, self.left_] * X[:, self.right_]
        return np.hstack([X, (P - self.means_) / self.scales_])

    def fit(self, y, X):
        X = np.asarray(X, dtype=np.float32)
        w, loss = logistic_gd(y, X, self.lambda_, self.max_iters, self.gamma)
        if self.n_top < 2:
            self.left_ = self.right_ = np.array([], dtype=int)
            self.means_ = self.scales_ = np.array([], dtype=np.float32)
            self.w_, self.loss_ = w, loss
            return self
        # Column 0 is the bias; rank the other features by |weight|.
        top = 1 + np.argsort(-np.abs(w[1:]))[:self.n_top]
        i, j = np.triu_indices(len(top), 1)
        self.left_, self.right_ = top[i], top[j]
        P = X[:, self.left_] * X[:, self.right_]
        self.means_ = P.mean(axis=0)
        self.scales_ = P.std(axis=0) + 1e-6
        del P
        self.w_, self.loss_ = logistic_gd(y, self._products(X), self.lambda_,
                                          self.max_iters, self.gamma)
        return self

    def predict_proba(self, X):
        return sigmoid(self._products(np.asarray(X, dtype=np.float32)) @ self.w_)


def load_best_config(path=BEST_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def model_from_config(config):
    p = config["params"]
    return InteractionModel(p["gamma"], p["lambda_"], p["max_iters"], p["n_top"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write config/best_model.json.")
    parser.add_argument("--base", type=Path, default=ROOT / "config" / "preprocessing_brfss.json")
    # 0.21 is the threshold chosen by evaluate_best.inner_threshold on the full
    # training set, i.e. by the same 3-fold selection as in each outer fold.
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--output", type=Path, default=BEST_CONFIG)
    args = parser.parse_args(argv)
    x_train = load_data()[0]
    with args.base.open(encoding="utf-8") as stream:
        base = json.load(stream)
    config = {"model": "interaction_logistic_regression",
              "params": {"gamma": 0.3, "lambda_": 1e-4, "max_iters": 300, "n_top": 40},
              "threshold": args.threshold,
              "preprocessing": best_preprocessing(x_train, base)}
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=1)
        stream.write("\n")
    print(f"{len(config['preprocessing']['categorical_columns'])} categorical columns; "
          f"saved {args.output}")


if __name__ == "__main__":
    main()
