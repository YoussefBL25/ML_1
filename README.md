# ML_1 — EPFL CS-433 Project 1 (heart disease prediction)

Binary classification of MICHD (coronary heart disease) from the BRFSS survey,
using only NumPy (plus matplotlib for plots).

## Setup

Put the three dataset files in `data/dataset_to_release/dataset/` without
renaming them:

```
data/dataset_to_release/dataset/x_train.csv
data/dataset_to_release/dataset/y_train.csv
data/dataset_to_release/dataset/x_test.csv
```

The first load parses the CSVs (about 2.5 minutes) and caches them in
`data/cache.npz`; later loads take a few seconds. Delete the cache if the CSVs change.

## Reproducing the submission

```
python run.py
```

This trains the final model on the full training set and writes
`submission.csv` (ids and predictions in {-1, 1}) for upload to AIcrowd.

## Missing deliverables (from the project description)

| Required | Status |
| --- | --- |
| `implementations.py` with the 6 methods | Done |
| `run.py` reproducing exactly the best AIcrowd submission | Exists, but no submission has been made yet; it must be updated to the final best model |
| `README.md` documenting data preparation, feature generation and cross-validation | Partial: the preprocessing described below is a mock until `preprocess.py` is ready |
| Report: 2-page PDF written in LaTeX (references may go on a third page) | **Missing** |
| Link to the public GitHub repository, submitted with the report | **Missing** (to give at submission) |

## Files

| File | Content |
| --- | --- |
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic), each returning `(w, loss)`. |
| `helpers.py` | Course-provided CSV loading and submission writing. |
| `data.py` | Cached data loading and a mock preprocessing (`baseline_preprocess`) used until `preprocess.py` exists. |
| `metrics.py` | F1 score for labels in {0, 1}. |
| `cross_validation.py` | Stratified k-fold split and `cross_validate` (out-of-fold scores, tuned cutoff, F1 per fold). |
| `threshold.py` | Picks the cutoff that maximizes F1 on validation scores. |
| `tuning.py` | Grid search over `gamma` and `lambda_` using `cross_validate`. |
| `tune_linear.py` | Timing and tuning of GD, SGD, least squares and ridge; writes `results/tuning_linear.json`. |
| `run.py` | Trains the final model and writes `submission.csv`. |

## Method

- **Labels**: converted from {-1, 1} to {0, 1} for training, back to {-1, 1} for the submission.
- **Preprocessing (mock)**: `baseline_preprocess` in `data.py` is a mock written only so
  that `tune_linear.py` and `run.py` could run before the team's `preprocess.py` (Person 1)
  exists. It drops all-NaN and constant columns, replaces NaN by the training median,
  standardizes with training mean and std, and adds a bias column; special codes
  (7, 9, 77, 99, ...) are not handled. It will be deleted and replaced by `preprocess.py`,
  and the results below will be recomputed with it.
- **Evaluation**: 5-fold stratified cross-validation, same folds (seed 0) for every method.
  Each model outputs a continuous score; the cutoff maximizing F1 on the pooled
  out-of-fold scores is chosen (grid 0.05 to 0.95, step 0.01), and the F1 is reported as
  mean ± std over folds at that cutoff. Because the cutoff is picked on the same
  out-of-fold scores, the F1 is slightly optimistic.
- **Tuning**: 6 settings per method with a hyperparameter, same folds for all.

## Results so far (5-fold CV, mock preprocessing)

| Method | Best setting | Cutoff | F1 (mean ± std) | One full training run |
| --- | --- | --- | --- | --- |
| GD (200 iters) | gamma = 0.09 | 0.22 | 0.4097 ± 0.0053 | 16.4 s |
| SGD (200k single-sample steps) | gamma = 1e-5 | 0.19 | 0.4028 ± 0.0049 | 5.3 s |
| Least squares | — | 0.22 | 0.4120 ± 0.0050 | 1.0 s |
| Ridge regression | lambda_ = 1e-6 | 0.22 | 0.4120 ± 0.0050 | 0.7 s |

Grids: GD gamma in {0.001, 0.003, 0.01, 0.03, 0.06, 0.09} (GD is stable for
gamma < 2 / lambda_max(X^T X / N) ≈ 0.095), SGD gamma in {1e-6, ..., 3e-4},
ridge lambda_ in {1e-6, ..., 1e-1}. `run.py` uses ridge regression, which ties
least squares but stays well-conditioned (X^T X is singular because of collinear features).
