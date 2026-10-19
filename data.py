"""Data loading (with a .npz cache) and a mock preprocessing.

baseline_preprocess is a mock, NOT the team's preprocessing: preprocessing and
label conversion belong to preprocess.py (Person 1). It was written only so that
tune_linear.py and run.py could run before preprocess.py exists, and is to be
deleted once it does.
"""

import os

import numpy as np

from helpers import load_csv_data

DATA_PATH = os.path.join("data", "dataset_to_release", "dataset")
CACHE_FILE = os.path.join("data", "cache.npz")


def load_data(data_path=DATA_PATH, cache_file=CACHE_FILE):
    """Load the CSVs with helpers.load_csv_data, caching the arrays as .npz.

    Parsing the CSVs takes minutes; the cache brings it down to seconds.

    Returns:
        tuple: (x_train, x_test, y_train, train_ids, test_ids), y_train in {-1, 1}.
    """
    if os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    np.savez(
        cache_file,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return x_train, x_test, y_train, train_ids, test_ids


def baseline_preprocess(x_train, x_test, y_train):
    """Minimal preprocessing: median imputation, standardization, bias column.

    All statistics come from x_train only. Columns that are entirely NaN or
    constant on x_train are dropped. Special codes (7, 9, 77, ...) are not handled.

    Returns:
        tuple: (tx_train, tx_test, y) with y in {0, 1}.
    """
    # Drop columns with no information on the training set
    keep = ~np.all(np.isnan(x_train), axis=0)
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    # Replace NaN by the training median of the column
    median = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), median, x_train)
    x_test = np.where(np.isnan(x_test), median, x_test)

    # Standardize with training statistics, dropping constant columns
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    keep = std > 0
    x_train = (x_train[:, keep] - mean[keep]) / std[keep]
    x_test = (x_test[:, keep] - mean[keep]) / std[keep]

    tx_train = np.c_[np.ones(len(x_train)), x_train]
    tx_test = np.c_[np.ones(len(x_test)), x_test]
    y = (y_train == 1).astype(float)
    return tx_train, tx_test, y
