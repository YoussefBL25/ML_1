"""5-fold CV of the final model, with the threshold tuned on inner folds.

Uses the same stratified folds (seed 0) as tune_models.py, so the per-fold
F1 can be compared with the earlier models. The options remove one part of
the final model at a time, for the ablation of the report: --n-top 0 drops
the interactions, --no-indicators the missing-value indicators, and
--no-one-hot the one-hot encoding. Writes results/best_model_cv.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cross_validation import stratified_k_fold
from data import ROOT, load_data
from interactions import load_best_config, log_loss, model_from_config
from metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess import Preprocessor, binary_labels
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-top", type=int, help="override the number of top features")
    parser.add_argument("--no-indicators", action="store_true")
    parser.add_argument("--no-one-hot", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "best_model_cv.json")
    args = parser.parse_args(argv)
    config = load_best_config()
    if args.n_top is not None:
        config["params"]["n_top"] = args.n_top
    if args.no_indicators:
        config["preprocessing"]["add_missing_indicators"] = False
    if args.no_one_hot:
        config["preprocessing"]["categorical_columns"] = []
    x_train, _, y_train, _, _ = load_data()
    y = binary_labels(y_train)
    rows, start = [], time.time()
    for fold, idx in enumerate(stratified_k_fold(y, 5, 0)):
        train = np.ones(len(y), dtype=bool)
        train[idx] = False
        threshold = inner_threshold(config, y[train], x_train[train])
        model, scores = fit_predict(config, y[train], x_train[train], x_train[idx])
        pred = (scores >= threshold).astype(int)
        rows.append({"fold": fold, "threshold": threshold,
                     "f1": f1_score(y[idx], pred),
                     "precision": precision_score(y[idx], pred),
                     "recall": recall_score(y[idx], pred),
                     "accuracy": accuracy_score(y[idx], pred),
                     "train_loss": model.loss_,
                     "val_loss": log_loss(y[idx], scores)})
        print(json.dumps(rows[-1]), flush=True)
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])),
                     "std": float(np.std([r[key] for r in rows]))}
               for key in rows[0] if key != "fold"}
    out = {"config": config, "folds": rows, "summary": summary,
           "elapsed_s": time.time() - start}
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=1)
    for key, value in summary.items():
        print(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f}")


if __name__ == "__main__":
    main()
