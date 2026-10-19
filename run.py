"""Main pipeline for EPFL CS-433 Project 1: train the final model and write submission.csv.

Current model: ridge regression (lambda_ = 1e-6) on the mock preprocessing of data.py
(stand-in until preprocess.py from Person 1 exists),
with cutoff 0.22 on X @ w. Both values come from 5-fold stratified CV in
tune_linear.py (F1 = 0.412 +/- 0.005).
"""

from data import load_data, baseline_preprocess
from helpers import create_csv_submission
from implementations import ridge_regression

LAMBDA = 1e-6
THRESHOLD = 0.22
OUTPUT_FILE = "submission.csv"


def main():
    """Train on the full training set, predict the test set and write the submission."""
    x_train, x_test, y_train, _, test_ids = load_data()
    tx_train, tx_test, y = baseline_preprocess(x_train, x_test, y_train)

    w, loss = ridge_regression(y, tx_train, LAMBDA)
    print(f"Training MSE loss (half): {loss:.5f}")

    # Scores above the cutoff are predicted positive, mapped back to {-1, 1}
    y_pred = (tx_test @ w >= THRESHOLD).astype(int) * 2 - 1
    create_csv_submission(test_ids, y_pred, OUTPUT_FILE)
    print(
        f"Wrote {OUTPUT_FILE}: {len(y_pred)} rows, {(y_pred == 1).mean():.2%} positive"
    )


if __name__ == "__main__":
    main()
