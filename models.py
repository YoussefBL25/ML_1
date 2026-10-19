"""Shared model adapters and configuration for experiments and submission."""

import json
from pathlib import Path

import numpy as np

from implementations import (
    least_squares, logistic_regression, mean_squared_error_gd,
    mean_squared_error_sgd, reg_logistic_regression, ridge_regression, sigmoid,
)
from preprocess import Preprocessor, binary_labels

LINEAR_MODELS = ("mean_squared_error_gd", "mean_squared_error_sgd",
                 "least_squares", "ridge_regression")
LOGISTIC_MODELS = ("logistic_regression", "reg_logistic_regression")
MODEL_NAMES = LINEAR_MODELS + LOGISTIC_MODELS
FUNCTIONS = {name: globals()[name] for name in MODEL_NAMES}


def validate_config(config):
    """Validate a JSON-compatible model specification; reject unknown options."""
    if not isinstance(config, dict):
        raise ValueError("A model configuration must be a JSON object.")
    unknown = set(config) - {"model", "params", "seed", "threshold", "preprocessing", "status"}
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    name = config.get("model")
    if name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}.")
    if not isinstance(config.get("params", {}), dict):
        raise ValueError("params must be a JSON object.")
    params = dict(config.get("params", {}))
    expected = set()
    if name in ("ridge_regression", "reg_logistic_regression"):
        expected.add("lambda_")
    if name not in ("least_squares", "ridge_regression"):
        expected.update(("gamma", "max_iters"))
    if set(params) != expected:
        raise ValueError(f"{name} requires exactly these params: {sorted(expected)}")
    for key in ("gamma", "lambda_"):
        if key in params:
            value = params[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not np.isfinite(value) or value < 0 or (key == "gamma" and value == 0)):
                raise ValueError(f"{key} must be finite and {'positive' if key == 'gamma' else 'nonnegative'}.")
    if "max_iters" in params:
        value = params["max_iters"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_iters must be a positive integer.")
    seed = config.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    cutoff = config.get("threshold", 0.5)
    if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or
            not np.isfinite(cutoff)):
        raise ValueError("threshold must be a finite number.")
    if not isinstance(config.get("preprocessing", {}), dict):
        raise ValueError("preprocessing must be a JSON object.")
    preprocessing = dict(config.get("preprocessing", {}))
    try:
        Preprocessor(**preprocessing)
    except TypeError as exc:
        raise ValueError(f"Invalid preprocessing configuration: {exc}") from exc
    return {"model": name, "params": params, "seed": seed,
            "threshold": float(cutoff), "preprocessing": preprocessing,
            "status": config.get("status", "candidate")}


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))


def train_model(config, y, X):
    """Train any of the six methods with deterministic SGD and finite checks."""
    config = validate_config(config)
    y = binary_labels(y)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or not np.all(np.isfinite(X)):
        raise ValueError("Training features must be finite and match labels.")
    name, params = config["model"], config["params"]
    fn = FUNCTIONS[name]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if name == "least_squares":
                w, loss = fn(y, X)
            elif name == "ridge_regression":
                w, loss = fn(y, X, params["lambda_"])
            else:
                initial = np.zeros(X.shape[1])
                if name == "reg_logistic_regression":
                    w, loss = fn(y, X, params["lambda_"], initial,
                                 params["max_iters"], params["gamma"])
                elif name == "mean_squared_error_sgd":
                    state = np.random.get_state()
                    try:
                        np.random.seed(config["seed"])
                        w, loss = fn(y, X, initial, params["max_iters"], params["gamma"])
                    finally:
                        np.random.set_state(state)
                else:
                    w, loss = fn(y, X, initial, params["max_iters"], params["gamma"])
        except FloatingPointError as exc:
            raise ValueError(f"{name} diverged; reduce gamma or rescale features.") from exc
    if not np.all(np.isfinite(w)) or not np.isfinite(loss):
        raise ValueError(f"{name} produced nonfinite weights/loss.")
    return w, float(loss)


def score_model(config, X, w):
    """Use sigmoid probabilities for logistic methods and raw linear scores otherwise."""
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.asarray(X) @ w
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced nonfinite scores.")
    return sigmoid(scores) if config["model"] in LOGISTIC_MODELS else scores
