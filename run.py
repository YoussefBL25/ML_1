"""Train a configured model and generate a reproducible submission CSV."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import DATA_PATH, ROOT, load_data
from boosting import BoostedPipeline, validate_boosted_config
from helpers import create_csv_submission
from interactions import BEST_CONFIG, model_from_config, validate_best_config
from models import score_model, train_model
from preprocess import Preprocessor, binary_labels


def validate_submission_config(config):
    if isinstance(config, dict) and config.get("model") == "histogram_boosted_blend":
        return validate_boosted_config(config)
    return validate_best_config(config)


def generate_submission(config, data_dir=DATA_PATH, output=ROOT / "submission.csv",
                        use_cache=True, workers=1):
    config = validate_submission_config(config)
    x_train, x_test, y_train, _, test_ids = load_data(data_dir, use_cache=use_cache)
    if config["model"] == "histogram_boosted_blend":
        model = BoostedPipeline(config).fit(binary_labels(y_train), x_train, progress=True,
                                            workers=workers)
        loss, scores = model.loss_, model.predict_proba(x_test)
    else:
        processor = Preprocessor(**config["preprocessing"])
        train_X = processor.fit_transform(x_train)
        test_X = processor.transform(x_test)
        if config["model"] == "interaction_logistic_regression":
            model = model_from_config(config).fit(binary_labels(y_train), train_X)
            loss, scores = model.loss_, model.predict_proba(test_X)
        else:
            w, loss = train_model(config, binary_labels(y_train), train_X)
            scores = score_model(config, test_X, w)
    scores = np.asarray(scores)
    if scores.shape != (len(test_ids),) or not np.all(np.isfinite(scores)) or not np.isfinite(loss):
        raise ValueError("Model scores and training loss must be finite and match test rows.")
    predictions = (scores >= config["threshold"]).astype(int) * 2 - 1
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, output)
    print(f"Training loss: {loss:.6f}")
    print(f"Wrote {output}: {len(predictions)} rows, "
          f"{np.mean(predictions == 1):.2%} positive")
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BEST_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processes for the tree seeds of a boosted configuration")
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8") as stream:
            config = validate_submission_config(json.load(stream))
        generate_submission(config, args.data_dir, args.output, not args.no_cache, args.workers)
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
