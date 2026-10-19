"""Validated CSV loading with a source-aware NumPy cache."""

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from helpers import load_csv_data
from preprocess import Preprocessor, binary_labels

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset_to_release" / "dataset"
KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids")
FILES = ("x_train.csv", "y_train.csv", "x_test.csv")


def _validate(arrays):
    x_train, x_test, y_train, train_ids, test_ids = arrays
    if (x_train.ndim != 2 or x_test.ndim != 2 or
            x_train.shape[1] == 0 or x_train.shape[1] != x_test.shape[1] or
            len(x_train) == 0 or len(x_test) == 0):
        raise ValueError("Training/test features must be nonempty with matching columns.")
    binary_labels(y_train)
    if len(y_train) != len(x_train) or len(train_ids) != len(x_train) or len(test_ids) != len(x_test):
        raise ValueError("Feature, label, and Id row counts do not match.")
    if np.any(np.isinf(x_train)) or np.any(np.isinf(x_test)):
        raise ValueError("Features contain infinity; missing values must be NaN.")
    for ids in (train_ids, test_ids):
        if (ids.ndim != 1 or not np.all(np.isfinite(ids)) or
                not np.all(ids == np.floor(ids)) or len(np.unique(ids)) != len(ids)):
            raise ValueError("Each CSV must have unique finite integer Ids.")
    return tuple(arrays)


def load_data(data_path=DATA_PATH, cache_file=None, use_cache=True):
    """Return (x_train, x_test, y_train, train_ids, test_ids).

    Paths default to the project directory, independent of the working directory.
    Cache metadata includes source paths, byte sizes, and nanosecond mtimes.
    Legacy caches without metadata are rebuilt; CSVs must always be present.
    """
    data_path = Path(data_path).resolve()
    sources = [data_path / name for name in FILES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Dataset files are missing: " + ", ".join(missing) +
                                ". Add the three course CSVs or pass --data-dir.")
    signature = json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sources])
    cache_file = Path(cache_file) if cache_file else data_path / "cache.npz"
    if use_cache and cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                if str(cached["sources"]) == signature:
                    return _validate([cached[key] for key in KEYS])
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
    arrays = _validate(load_csv_data(data_path))
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, sources=signature, **dict(zip(KEYS, arrays)))
    return arrays


def baseline_preprocess(x_train, x_test, y_train):
    """Compatibility wrapper; CV must fit a new Preprocessor per fold."""
    processor = Preprocessor()
    return (processor.fit_transform(x_train), processor.transform(x_test),
            binary_labels(y_train))
