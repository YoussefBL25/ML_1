"""Best logistic model: richer preprocessing plus pairwise feature interactions.

The preprocessing adds missing-value indicators and one-hot encodes every
feature with 3 to 10 observed values, on top of the BRFSS codebook rules. The
model is regularized logistic regression trained twice: a first fit ranks the
features by |weight|, then the pairwise products of the top features are
added and the model is refit. Imputation, scaling, category levels, and
interaction ranking are fitted on training rows. The configured categorical
column list is generated separately from the full training feature schema.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import DATA_PATH, ROOT, load_data
from models import validate_config
from preprocess import Preprocessor, binary_labels

BEST_CONFIG = ROOT / "config" / "best_model.json"


def low_cardinality_columns(X, settings, max_levels=10):
    """Columns with 3..max_levels distinct values after the codebook rules."""
    cleaned = Preprocessor(**settings)._clean(X)
    excluded = set(settings.get("drop_columns", [])) | set(settings.get("categorical_columns", []))
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
    settings["categorical_columns"] = (list(base.get("categorical_columns", [])) +
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
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for _ in range(max_iters):
                w -= gamma * (X.T @ (sigmoid(X @ w) - y) / len(y) + 2 * lambda_ * w)
    except FloatingPointError as exc:
        raise ValueError("Interaction model diverged; reduce gamma or rescale features.") from exc
    if not np.all(np.isfinite(w)):
        raise ValueError("Interaction model produced nonfinite weights.")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = X @ w
    if not np.all(np.isfinite(scores)):
        raise ValueError("Interaction model produced nonfinite training scores.")
    return w, log_loss(y, sigmoid(scores))


def log_loss(y, p):
    """Finite binary log loss, including saturated float32 probabilities."""
    y = binary_labels(y)
    p = np.asarray(p, dtype=np.float64)
    if (p.shape != y.shape or not np.all(np.isfinite(p)) or
            np.any((p < 0) | (p > 1))):
        raise ValueError("Probabilities must be finite, in [0, 1], and match labels.")
    # Cast before clipping: in float32, 1 - 1e-12 rounds back to 1.
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


class InteractionModel:
    """Regularized logistic regression with products of the top features."""

    def __init__(self, gamma=0.3, lambda_=1e-4, max_iters=300, n_top=40):
        validate_best_config({"model": "interaction_logistic_regression",
                              "params": {"gamma": gamma, "lambda_": lambda_,
                                         "max_iters": max_iters, "n_top": n_top}})
        self.gamma, self.lambda_ = gamma, lambda_
        self.max_iters, self.n_top = max_iters, n_top

    def _products(self, X):
        with np.errstate(over="ignore", invalid="ignore"):
            P = X[:, self.left_] * X[:, self.right_]
            result = np.hstack([X, (P - self.means_) / self.scales_])
        if not np.all(np.isfinite(result)):
            raise ValueError("Interaction features contain nonfinite values.")
        return result

    def fit(self, y, X):
        y = binary_labels(y)
        with np.errstate(over="ignore", invalid="ignore"):
            X = np.asarray(X, dtype=np.float32)
        if (X.ndim != 2 or X.shape[1] == 0 or len(X) != len(y) or
                not np.all(np.isfinite(X)) or not np.all(X[:, 0] == 1)):
            raise ValueError("Training features must be finite, match labels, and start with a bias column.")
        self.n_features_in_ = X.shape[1]
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
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                P = X[:, self.left_] * X[:, self.right_]
                self.means_ = P.mean(axis=0)
                self.scales_ = P.std(axis=0) + 1e-6
        except FloatingPointError as exc:
            raise ValueError("Interaction features are too large to scale safely.") from exc
        del P
        self.w_, self.loss_ = logistic_gd(y, self._products(X), self.lambda_,
                                          self.max_iters, self.gamma)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "w_"):
            raise ValueError("Call fit before predict_proba.")
        with np.errstate(over="ignore", invalid="ignore"):
            X = np.asarray(X, dtype=np.float32)
        if (X.ndim != 2 or X.shape[1] != self.n_features_in_ or
                not np.all(np.isfinite(X)) or not np.all(X[:, 0] == 1)):
            raise ValueError("Prediction features must be finite and match the fitted schema and bias.")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self._products(X) @ self.w_
        if not np.all(np.isfinite(scores)):
            raise ValueError("Interaction model produced nonfinite scores.")
        return sigmoid(scores)


def validate_best_config(config):
    """Validate both the final model and the six baseline configurations."""
    if not isinstance(config, dict) or config.get("model") != "interaction_logistic_regression":
        return validate_config(config)
    params = config.get("params")
    if not isinstance(params, dict) or "n_top" not in params:
        raise ValueError("Interaction params must include n_top.")
    n_top = params["n_top"]
    if isinstance(n_top, bool) or not isinstance(n_top, int) or n_top < 0:
        raise ValueError("n_top must be a nonnegative integer.")
    base = dict(config, model="reg_logistic_regression",
                params={k: v for k, v in params.items() if k != "n_top"})
    validated = validate_config(base)
    if not validated["preprocessing"].get("add_bias", True):
        raise ValueError("The interaction model requires add_bias=True.")
    return dict(validated, model="interaction_logistic_regression",
                params=dict(validated["params"], n_top=n_top))


def load_best_config(path=BEST_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_best_config(json.load(stream))


def model_from_config(config):
    config = validate_best_config(config)
    if config["model"] != "interaction_logistic_regression":
        raise ValueError("Expected an interaction_logistic_regression configuration.")
    p = config["params"]
    return InteractionModel(p["gamma"], p["lambda_"], p["max_iters"], p["n_top"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write config/best_model.json.")
    parser.add_argument("--base", type=Path, default=ROOT / "config" / "preprocessing_brfss.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--no-cache", action="store_true")
    # 0.21 is the threshold chosen by evaluate_best.inner_threshold on the full
    # training set, i.e. by the same 3-fold selection as in each outer fold.
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--output", type=Path, default=BEST_CONFIG)
    args = parser.parse_args(argv)
    try:
        if args.base.resolve() == args.output.resolve():
            raise ValueError("--base and --output must be different files.")
        x_train = load_data(args.data_dir, use_cache=not args.no_cache)[0]
        with args.base.open(encoding="utf-8") as stream:
            base = json.load(stream)
        config = validate_best_config({
            "model": "interaction_logistic_regression",
            "params": {"gamma": 0.3, "lambda_": 1e-4, "max_iters": 300, "n_top": 40},
            "threshold": args.threshold,
            "preprocessing": best_preprocessing(x_train, base)})
        payload = json.dumps(config, indent=1, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"{len(config['preprocessing']['categorical_columns'])} categorical columns; "
          f"saved {args.output}")


if __name__ == "__main__":
    main()
