"""Train a configured model and generate a reproducible submission CSV."""

import argparse
from pathlib import Path

import numpy as np

from data import DATA_PATH, ROOT, load_data
from helpers import create_csv_submission
from interactions import BEST_CONFIG, load_best_config, model_from_config
from models import load_config, score_model, train_model
from preprocess import Preprocessor, binary_labels


def generate_submission(config, data_dir=DATA_PATH, output=ROOT / "submission.csv",
                        use_cache=True):
    x_train, x_test, y_train, _, test_ids = load_data(data_dir, use_cache=use_cache)
    processor = Preprocessor(**config["preprocessing"])
    train_X = processor.fit_transform(x_train)
    test_X = processor.transform(x_test)
    if config["model"] == "interaction_logistic_regression":
        model = model_from_config(config).fit(binary_labels(y_train), train_X)
        loss, scores = model.loss_, model.predict_proba(test_X)
    else:
        w, loss = train_model(config, binary_labels(y_train), train_X)
        scores = score_model(config, test_X, w)
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
    args = parser.parse_args(argv)
    try:
        config = load_best_config(args.config)
        if config.get("model") != "interaction_logistic_regression":
            config = load_config(args.config)
        generate_submission(config, args.data_dir, args.output, not args.no_cache)
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
