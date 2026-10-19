"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=float,
        usecols=(0, 1),
        ndmin=2,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1, ndmin=2
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1, ndmin=2
    )

    if not np.array_equal(y_train[:, 0], x_train[:, 0]):
        raise ValueError("y_train Ids must match x_train Ids in the same row order.")
    for ids in (x_train[:, 0], x_test[:, 0]):
        if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
            raise ValueError("CSV Ids must be finite integers.")
    y_train = y_train[:, 1]
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    ids, y_pred = np.asarray(ids), np.asarray(y_pred)
    if ids.ndim != 1 or y_pred.ndim != 1 or len(ids) != len(y_pred) or len(ids) == 0:
        raise ValueError("Ids and predictions must be nonempty vectors of equal length.")
    if (not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)) or
            len(np.unique(ids)) != len(ids)):
        raise ValueError("Submission Ids must be unique finite integers.")
    if not np.all(np.isin(y_pred, [-1, 1])):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
