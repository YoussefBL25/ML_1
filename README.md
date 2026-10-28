# ML_1 — EPFL CS-433 Project 1

Binary classification of reported coronary heart disease (MICHD) from BRFSS
survey features. Training and preprocessing use only NumPy and Python's standard
library. The dataset is not included.

## Current status

The preprocessing, validation, tuning, and submission infrastructure can be
tested without the medical dataset. The six required methods are implemented.
Real-data feature rules, experiments, final model selection, and the report's
findings remain pending.

`config/model.json` is explicitly a **provisional legacy baseline**: ridge with
`lambda_=1e-6` and cutoff `0.22`. It has not been selected using the new
validation protocol or verified against a best AIcrowd submission.

## Setup and checks without the dataset

Use Python 3.10+ and install the runtime dependency:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --help
python tune_models.py --help
```

The tests generate tiny synthetic CSVs in temporary directories. They check
analytic updates and losses for all six methods, gradients, singular least
squares, extreme logistic scores, preprocessing statistics and schemas,
validation isolation, metrics, cache refresh, CSV validation, and repeatable
tuning-to-submission execution. They do not measure medical-data performance.
No external testing library is required for this local suite.

Verification completed with Python 3.12 and NumPy 2.3.5: all 30 local synthetic
tests passed. The official course public tests were also run from a separate
temporary directory: 26 passed, with the GitHub URL format check and optional
Black formatting check excluded for local verification. Those tests do not
establish final submission performance or validate the published repository URL.
To repeat the course checks using a separate checkout of
[the course tests](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests),
install their test dependencies in a development environment and run:

```bash
python -m pytest /path/to/ML_course/projects/project1/grading_tests \
  --github_link /absolute/path/to/ML_1 \
  -k 'not github_link_format and not black_format'
```

Run the URL check with the final public commit URL before submitting. The LaTeX
draft has not been compiled in this environment because LaTeX is not installed.

## Dataset loading

Place the three course files here without renaming them:

```text
data/dataset_to_release/dataset/x_train.csv
data/dataset_to_release/dataset/y_train.csv
data/dataset_to_release/dataset/x_test.csv
```

Default paths are relative to the project directory, so scripts also work when
launched from another working directory. `--data-dir /path/to/dataset` overrides
the input directory. The loader checks feature dimensions, row counts, unique
integer Ids, label values, and alignment of training feature/label Ids.

The first load parses CSVs and creates `cache.npz` inside that dataset directory.
Later loads reuse it only when source paths, sizes, and modification times match.
Legacy caches are rebuilt. Use `--no-cache` to bypass caching, including when
files have been replaced while preserving their sizes and timestamps. Missing
files produce an actionable error before training or submission output.

## Preprocessing and feature generation

`preprocess.Preprocessor` provides `fit`, `transform`, and `fit_transform`.
Every validation training partition gets a fresh instance. The submission
pipeline fits once on all training rows and transforms test rows with that state.
Labels convert from `{-1, 1}` to `{0, 1}` for training.

The default transformation drops all-missing numeric columns, imputes numeric
NaNs with training medians, drops constant generated features, standardizes with
training means/standard deviations, and adds a bias column. It replaces the
previous mock implementation; `data.baseline_preprocess` remains a compatibility
wrapper and must not be called on all data before cross-validation.

Options in `config/preprocessing.json` are:

| Option | Meaning |
| --- | --- |
| `missing_codes` | Per-column lists of survey codes to replace with NaN; empty by default. |
| `zero_codes` | Per-column lists of survey codes meaning "none"/"never", replaced with 0. |
| `drop_columns` | Raw column indices removed before any other step. |
| `categorical_columns` | Raw column indices to encode categorically; empty by default. |
| `add_missing_indicators` | Add indicators for columns containing missing values during fit. |
| `degree` | `1` for original numeric values; `2` also adds their imputed squares, without interactions. |
| `standardize` | Scale all retained generated features, including indicator/encoded features. |
| `add_bias` | Prepend a column of ones. |

Indices are zero-based and exclude the CSV Id column. For example,
`"missing_codes": {"0": [7, 9], "2": [77, 99]}` only changes columns 0 and 2.
This illustrates the configuration format, **not verified BRFSS rules**. Confirm
each mapping using the course feature documentation before enabling it. Codes
such as 7 or 99 can be valid values in other columns.

Categorical levels are learned from training data, with the first observed
sorted level as reference (one fewer dummy column). Missingness gets its own
level only if present during fit. Unseen levels, including previously unseen
missingness, map to all-zero dummy values before scaling. Numeric all-missing
columns remain dropped even if validation/test values exist. These conventions
keep validation/test distributions from changing the fitted feature schema.

### BRFSS preprocessing

`config/preprocessing_brfss.json` applies the BRFSS 2015 codebook
(https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf):
"don't know"/"refused" codes become missing, "none"/"never" codes become 0,
and identifiers, dates, survey weights, raw duplicates of calculated variables
and high-cardinality nominal codes are dropped (64 columns). Seven nominal
variables are one-hot encoded. Regenerate it with:

```bash
python make_preprocessing_config.py
```

The rules and per-column exceptions are listed in that script; only training
feature maxima are read, never labels.

## Models and reproducibility

`models.py` exposes all six required implementations to both tuning and
submission. Linear models output `X @ w`; logistic methods output sigmoid
probabilities. SGD uses the configuration's seed and restores the caller's
random-number state. All methods return the last weights and unpenalized loss.
MSE includes the required factor `1/2`; regularizers use `lambda_ * ||w||^2`,
including the bias. Least squares uses normal equations with a pseudoinverse
fallback for singular systems. Logistic loss uses stable `logaddexp` evaluation.

Model JSON files contain `model`, `params`, `seed`, `threshold`, `preprocessing`,
and an optional `status` label. Unknown keys, invalid parameters, and nonfinite
model outputs are rejected. Retain the exact selected JSON, code commit,
Python/NumPy versions, input files, and uploaded CSV for each final experiment.
Dependency ranges alone do not guarantee identical numerical results across
different environments.

## Validation and tuning once data is available

Run all six methods, only the four linear methods, or a smaller model grid:

```bash
python tune_models.py --preprocessing config/preprocessing.json
python tune_linear.py
python tune_models.py --methods ridge_regression,logistic_regression
```

The default protocol uses five stratified outer folds, three inner folds, and
seed 0. Each outer training partition independently selects the model,
hyperparameters, preprocessing variant, and threshold using inner out-of-fold
scores. Its validation partition is used only to compute F1 for the chosen
configuration. Report `outer_mean_f1` and `outer_std_f1`. The same splits are used
for every candidate, and every inner/outer fit learns preprocessing only on its
own training rows. Each class must have enough examples for both split levels.

After outer evaluation, a separate full-training CV search produces the
deployment configuration. Its `selection_f1` and `selection_fold_std` are
selection statistics, not unbiased test estimates. Candidate sets must be fixed
before interpreting outer scores; repeatedly changing grids using those scores
can overfit the evaluation. `cross_validate` provides nested cutoff selection
for a fixed model; `nested_grid_search` evaluates the entire selection procedure.
The compatibility `grid_search` reports selection results only.

The default threshold grid is `0.05` through `0.95`, step `0.01`; ties choose the
lowest threshold. The Python validation APIs also accept custom finite threshold
grids, including thresholds outside `[0, 1]` for linear scores.

Hyperparameter grids are defined in `tune_models.make_candidates`. Linear
methods use starter grids (200 GD iterations, 200,000 single-sample SGD steps,
six learning rates per iterative method, six ridge penalties). The logistic
grid is log-spaced: gamma in {0.01, 0.03, 0.1, 0.3, 1}, lambda_ in
{1e-6, ..., 1e-1} for regularized logistic regression, and 300 iterations
(see "Logistic experiments" below). Divergence stops the run with an error; review
scaling and learning rates before rerunning. Nested searches can be expensive.

To compare feature variants or use smaller grids, supply a list of full model
configurations. The example contains baseline ridge, missing indicators, numeric
squares, logistic, and regularized logistic candidates:

```bash
python tune_models.py --candidates config/candidates.example.json \
  --folds 5 --inner-folds 3 --seed 0
```

`--candidates` overrides `--methods` and `--preprocessing`; candidate seeds are
stored in the JSON, while `--seed` controls folds. For ablations, run restricted
candidate lists on identical splits and record their separate outer estimates.

The default outputs are `results/tuning_nested.json` (protocol, outer scores,
outer precision/recall/accuracy, the tuning grid with the inner-CV F1 of every
combination and how many outer folds selected it, runtime versions, and elapsed time) and
`results/selected_model.json` (the configuration accepted by `run.py`). Override
them with `--output` and `--model-output` to keep each experiment separately.

### Logistic experiments (report Tables I-II, Fig. 1)

Each method is tuned separately with the same folds, once with the BRFSS
preprocessing and once with the generic default preprocessing (ablation):

```bash
for m in logistic_regression reg_logistic_regression; do
  python tune_models.py --methods $m --preprocessing config/preprocessing_brfss.json \
    --output results/tuning_$m.json --model-output results/selected_$m.json
  python tune_models.py --methods $m --preprocessing config/preprocessing.json \
    --output results/tuning_${m}_default.json --model-output /tmp/selected_${m}_default.json
done
python imbalance.py   # results/imbalance.json
python plots.py       # report/figures/tuning.pdf
```

The logistic grid is log-spaced over each hyperparameter: gamma in half
decades from too slow to converge in 300 iterations up to beyond the step size
bound 8 / lambda_max(X^T X / N) ~ 0.57, and lambda_ in decades from negligible
(1e-6) to strong (1e-1). `imbalance.py` reuses the tuned
regularized config and compares positive-class loss weights 1, sqrt(r) and r
(r = negatives / positives) on the same folds; `plots.py` plots
the cross-validated grid scores. The regularized logistic grid (30
combinations) takes about one hour on 16 cores.

### Final model (best AIcrowd submission)

The submitted model is regularized logistic regression with
`gamma=0.3`, `lambda_=1e-4`, 300 iterations and threshold `0.21`, defined in
`config/best_model.json` and implemented in `interactions.py`:

1. Preprocessing: the BRFSS codebook rules above, plus a missing-value
   indicator for every feature with missing values, and one-hot encoding of
   every feature with 3 to 10 observed values (86 categorical features).
2. Interactions: a first fit ranks the features by |weight|; the pairwise
   products of the 40 top features (780 products) are standardized and added,
   and the model is refit. The ranking uses training rows only.

```bash
python interactions.py   # writes config/best_model.json from the training features
python evaluate_best.py  # 5-fold CV, threshold tuned on 3 inner folds (~15 min)
python evaluate_best.py --n-top 0 --output results/best_model_no_interactions_cv.json
```

`evaluate_best.py` uses the same outer folds as `tune_models.py` and reports
F1, precision, recall and accuracy per fold in `results/best_model_cv.json`
(F1 0.441 +/- 0.006). The threshold 0.21 is the one selected by the same
3-fold procedure on the full training set. The submission scored F1 0.450 and
accuracy 0.878 on AIcrowd.

## Generating submissions

```bash
# Reproduces the best AIcrowd submission (about 1.5 min).
python run.py

# Any other configuration, e.g. one selected by tune_models.py.
python run.py --config results/selected_model.json --output submission.csv
```

Training uses all training rows, then predictions are mapped back to `{-1, 1}`.
Output contains exactly `Id,Prediction` with test Ids in their original order.
Length mismatches, duplicate/noninteger Ids, and invalid labels are rejected
before opening the output file. Repeated runs with identical data, configuration,
and environment are covered by synthetic tests.

`python run.py` reads `config/best_model.json` by default and reproduces the
CSV of the best AIcrowd submission exactly. No AIcrowd submission is performed
by these scripts.

## Historical results

`results/tuning_linear.json` preserves the earlier mock-preprocessing results:
GD F1 0.4097, SGD 0.4028, least squares 0.4120, ridge 0.4120. Those experiments
preprocessed the full training set before CV and selected/evaluated the cutoff
on the same pooled predictions. They require recomputation and are not results
from the current validation pipeline. New tuning outputs use different filenames
to preserve this historical record.

## Report and remaining deliverables

`report/main.tex` is a draft structure with explicit placeholders, not a completed
scientific report. With a LaTeX installation, build it using:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory report report/main.tex
```

Keep the completed report within two pages, without an appendix; references may
occupy a third page. Report sources and the final `report/main.pdf` are allowed
in Git; auxiliary build files are ignored.

Remaining work requiring data or measured findings:

- Inspect the actual feature schema and distributions; verify per-column survey
  codes and categorical/ordinal/continuous treatments.
- Run the real-data tests, model grids, feature experiments, and ablations;
  measure results, runtime, and errors, and finalize the configuration.
- Generate and upload predictions to AIcrowd; verify exact best-CSV reproduction.
- Fill in report authors, methods/settings, measured results, conclusions,
  references, and the public repository URL; compile and inspect the final PDF.
- Submit the report with a public repository link pinned to the final commit.

## Files

| File | Purpose |
| --- | --- |
| `implementations.py` | Six required algorithms and loss/gradient helpers. |
| `helpers.py`, `data.py` | Validated CSV I/O and source-aware cache. |
| `preprocess.py` | Training-only fit/transform preparation and label conversion. |
| `metrics.py`, `threshold.py` | F1, precision, recall, accuracy; F1-optimal cutoff search. |
| `cross_validation.py`, `tuning.py` | Stratified splits, fold preprocessing, nested evaluation and selection. |
| `models.py`, `config/` | Shared model adapters, provisional config, and experiment examples. |
| `tune_models.py`, `tune_linear.py` | Nested experiment command-line entry points. |
| `make_preprocessing_config.py` | BRFSS codebook rules -> `config/preprocessing_brfss.json`. |
| `imbalance.py` | Class-weighted logistic regression ablation. |
| `plots.py` | Report figures from `results/` (no training). |
| `interactions.py` | Final model: richer preprocessing and pairwise interactions; writes `config/best_model.json`. |
| `evaluate_best.py` | 5-fold CV of the final model, with and without interactions. |
| `run.py` | Full-data training and submission generation; defaults to the final model. |
| `tests/` | Synthetic algorithm, isolation, I/O, and end-to-end checks. |
| `report/main.tex` | LaTeX report (IEEE template of the course); figures in `report/figures/`. |
| `results/` | JSON outputs of the experiments above. |
