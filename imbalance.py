"""Class-weighted logistic regression at the tuned hyperparameters.

The loss of positive examples is multiplied by 1, sqrt(r) or r, with
r = negatives / positives on each training part; weight 1 is exactly
reg_logistic_regression. All weights share the folds of
cross_validation.cross_validate, with cutoffs chosen on inner folds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cross_validation import cross_validate
from data import DATA_PATH, ROOT, load_data
from implementations import sigmoid
from preprocess import Preprocessor, binary_labels


def weighted_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, pos_weight):
    sample_weight = np.where(y == 1, pos_weight, 1.0)
    sample_weight /= sample_weight.mean()
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        grad = tx.T @ (sample_weight * (sigmoid(tx @ w) - y)) / len(y) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w


def positive_weight(name, y):
    ratio = np.sum(y == 0) / np.sum(y == 1)
    return {"1": 1.0, "sqrt": np.sqrt(ratio), "balanced": ratio}[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path,
                        default=ROOT / "results" / "tuning_reg_logistic_regression.json",
                        help="tune_models.py output whose selected config is reused")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lambda_", type=float)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "imbalance.json")
    args = parser.parse_args(argv)
    with args.tuning.open(encoding="utf-8") as stream:
        selected = json.load(stream)["best"]["config"]
    params = dict(selected["params"])
    params.update({k: v for k, v in (("gamma", args.gamma), ("lambda_", args.lambda_)) if v is not None})
    x_train, _, y, _, _ = load_data(args.data_dir)
    y = binary_labels(y)

    def train_fn(y_part, tx, weight):
        return weighted_reg_logistic_regression(
            y_part, tx, params["lambda_"], np.zeros(tx.shape[1]), params["max_iters"],
            params["gamma"], positive_weight(weight, y_part))

    rows = []
    for weight in ("1", "sqrt", "balanced"):
        cutoff, mean_f1, std_f1 = cross_validate(
            y, x_train, train_fn, lambda tx, w: sigmoid(tx @ w), weight, args.folds,
            args.seed, preprocessor_factory=lambda: Preprocessor(**selected["preprocessing"]),
            inner_k=args.inner_folds)
        rows.append({"pos_weight": weight, "f1": mean_f1, "std": std_f1,
                     "deployment_threshold": cutoff})
        print(f"pos_weight={weight}: F1 {mean_f1:.4f} +/- {std_f1:.4f}, threshold {cutoff:.2f}",
              flush=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump({"params": params, "folds": args.folds, "inner_folds": args.inner_folds,
                   "seed": args.seed, "rows": rows}, stream, indent=2)
        stream.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
