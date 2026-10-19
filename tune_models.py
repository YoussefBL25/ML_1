"""Nested tuning for all six required models using raw CSV features."""

import argparse
from itertools import product
import json
from pathlib import Path
import platform
import time

import numpy as np

from data import DATA_PATH, ROOT, load_data
from models import LINEAR_MODELS, MODEL_NAMES, validate_config
from tuning import nested_grid_search


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
            gammas = ((1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
                      if name == "mean_squared_error_sgd"
                      else (0.001, 0.003, 0.01, 0.03, 0.06, 0.09))
            iterations = 200_000 if name == "mean_squared_error_sgd" else 200
            lambdas = (1e-4, 1e-2) if name == "reg_logistic_regression" else (None,)
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


def main(argv=None, default_methods="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--methods", default=default_methods,
                        help="all, linear, or comma-separated model names")
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
            methods = (MODEL_NAMES if args.methods == "all" else LINEAR_MODELS
                       if args.methods == "linear" else tuple(args.methods.split(",")))
            candidates = make_candidates(methods, preprocessing, args.seed)
        x_train, _, y, _, _ = load_data(args.data_dir, use_cache=not args.no_cache)
        print(f"Nested CV: {len(candidates)} candidates, {args.folds} outer / "
              f"{args.inner_folds} inner folds; this can require many training runs.", flush=True)
        start = time.perf_counter()
        result = nested_grid_search(y, x_train, candidates, args.folds,
                                    args.inner_folds, args.seed)
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
        print(f"Selected config: {args.model_output}")
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
