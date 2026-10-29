"""Compare NumPy boosted trees and logistic blends on disjoint data portions.

Fixed 60/20/20 stratified split: train models on 60%, select tree count, blend
weight and threshold on 20%, and score on the remaining 20%. Scores are an
exploratory holdout comparison, not the nested five-fold estimate in the report.
Writes the best NEW configuration selected by tuning, even if baseline wins;
the JSON results explicitly record whether it improved the independent test.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np

from boosting import HistogramBinner, HistogramGradientBoosting, validate_boosted_config
from cross_validation import stratified_k_fold
from data import DATA_PATH, ROOT, load_data
from interactions import BEST_CONFIG, load_best_config, model_from_config
from preprocess import Preprocessor, binary_labels
from screen_optimizations import metrics
from threshold import THRESHOLDS, best_threshold


def save_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--logistic-config", type=Path, default=BEST_CONFIG)
    parser.add_argument("--candidate-config", type=Path,
                        help="Evaluate one fixed candidate on a new split; do not search parameters")
    parser.add_argument("--preprocessing", type=Path, default=ROOT / "config/preprocessing_brfss.json")
    parser.add_argument("--depths", default="3,4")
    parser.add_argument("--checkpoints", default="100,200,300")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "results/boosting_screen.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "config/boosted_candidate.json")
    args = parser.parse_args(argv)
    depths = [int(value) for value in args.depths.split(",")]
    checkpoints = sorted(set(int(value) for value in args.checkpoints.split(",")))
    if not depths or not checkpoints or min(checkpoints) < 1:
        parser.error("Provide at least one depth and positive checkpoint.")
    if args.output.resolve() == args.model_output.resolve():
        parser.error("Result and model output paths must differ.")
    start = time.perf_counter()
    logistic_config = load_best_config(args.logistic_config)
    with args.preprocessing.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    fixed = None
    if args.candidate_config:
        with args.candidate_config.open(encoding="utf-8") as stream:
            fixed = validate_boosted_config(json.load(stream))
        fixed_model = HistogramGradientBoosting(**fixed["params"])
        depths = [fixed_model.max_depth]
        checkpoints = [fixed_model.n_estimators]
        settings = fixed["preprocessing"]
        if fixed["tree_weight"] < 1:
            logistic_config = fixed["logistic_config"]
    X, X_test, y, train_ids, test_ids = load_data(args.data_dir)
    del X_test, train_ids, test_ids
    y = binary_labels(y)
    folds = stratified_k_fold(y, 5, args.seed)
    test, tuning, train = folds[0], folds[1], np.concatenate(folds[2:])
    yt, yv, ye = y[train], y[tuning], y[test]
    print(f"Split: {len(train)} training / {len(tuning)} tuning / {len(test)} test", flush=True)
    print("Training paired interaction-logistic baseline.", flush=True)
    processor = Preprocessor(**logistic_config["preprocessing"])
    features = processor.fit_transform(X[train])
    logistic = model_from_config(logistic_config).fit(yt, features)
    del features
    logistic_v = logistic.predict_proba(processor.transform(X[tuning]))
    logistic_e = logistic.predict_proba(processor.transform(X[test]))
    del logistic, processor
    gc.collect()
    cutoff, f1 = best_threshold(yv, logistic_v, THRESHOLDS)
    baseline = {"model": "interaction_logistic_regression", "threshold": cutoff,
                "tuning_f1": f1, "test": metrics(ye, logistic_e, cutoff)}
    binner = HistogramBinner(settings, max_bins=fixed["params"].get("max_bins", 32) if fixed else 32,
                             seed=fixed["params"].get("seed", 0) if fixed else 0)
    train_bins = binner.fit_transform(X[train])
    tuning_bins = binner.transform(X[tuning])
    test_bins = binner.transform(X[test])
    del X
    candidates = []
    for depth in depths:
        params = {"n_estimators": max(checkpoints), "max_depth": depth,
                  "learning_rate": .08, "max_bins": 32, "min_samples_leaf": 150,
                  "l2": 10., "subsample": .7, "feature_fraction": .8, "seed": 0}
        if fixed:
            params = dict(fixed["params"])
        print(f"Training boosted trees, depth {depth}.", flush=True)
        model = HistogramGradientBoosting(**params).fit(yt, train_bins, progress=True)
        v_scores = dict(model.staged_predict_proba(tuning_bins, checkpoints))
        e_scores = dict(model.staged_predict_proba(test_bins, checkpoints))
        for count in checkpoints:
            for weight in ([fixed["tree_weight"]] if fixed else (.5, 1.)):
                scores_v = weight * v_scores[count] + (1 - weight) * logistic_v
                scores_e = weight * e_scores[count] + (1 - weight) * logistic_e
                cutoff, selection_f1 = best_threshold(yv, scores_v, THRESHOLDS)
                config = validate_boosted_config({
                    "model": "histogram_boosted_blend", "params": dict(params, n_estimators=count),
                    "preprocessing": settings, "tree_weight": weight, "threshold": cutoff,
                    "logistic_config": logistic_config if weight < 1 else None,
                    "status": "candidate_selected_on_tuning_holdout_not_aicrowd_verified"})
                candidates.append({"config": config, "tuning_f1": selection_f1,
                                   "test": metrics(ye, scores_e, cutoff)})
        del model
    # All selection is based on the tuning portion, never on the test scores.
    selected = max(candidates, key=lambda row: row["tuning_f1"])
    result = {"protocol": "exploratory_stratified_train_tuning_test_split", "seed": args.seed,
              "selection_scope": "threshold_only" if fixed else "tree_parameters_blend_and_threshold",
              "split_sizes": {"train": len(train), "tuning": len(tuning), "test": len(test)},
              "baseline": baseline, "candidates": candidates, "selected_by_tuning": selected,
              "test_f1_gain": selected["test"]["f1"] - baseline["test"]["f1"],
              "tuning_prefers_new_model": selected["tuning_f1"] > baseline["tuning_f1"],
              "thresholds": THRESHOLDS.tolist(), "dataset_directory": str(args.data_dir.resolve()),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "elapsed_s": time.perf_counter() - start}
    save_json(args.output, result)
    save_json(args.model_output, selected["config"])
    print("Baseline:", json.dumps(baseline, allow_nan=False), flush=True)
    print("Selected:", json.dumps({k: v for k, v in selected.items() if k != "config"}), flush=True)
    print("Selected parameters:", selected["config"]["params"], flush=True)
    print("Selected tree weight:", selected["config"]["tree_weight"], flush=True)
    print(f"Saved {args.output} and {args.model_output}", flush=True)


if __name__ == "__main__":
    main()
