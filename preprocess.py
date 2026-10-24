"""Training-only feature preparation using NumPy.

Column indices exclude the CSV Id column. Survey codes are configured per
column; no codes are guessed globally.
"""

import numpy as np


def binary_labels(y):
    """Convert {-1, 1} or {0, 1} labels to a one-dimensional float array."""
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    if np.all(np.isin(y, [0, 1])):
        return y.astype(float)
    if np.all(np.isin(y, [-1, 1])):
        return (y == 1).astype(float)
    raise ValueError("Labels must contain only {-1, 1} or {0, 1}.")


class Preprocessor:
    """Median imputation, optional encoding/features, scaling, and a bias.

    Numeric all-missing columns are removed. Categorical columns use training
    levels with the first level as reference; unseen levels map to all zeros.
    A categorical missing level is learned only if missingness occurs in fit.
    Constant generated features are dropped. All generated features (including
    indicators) are standardized when standardize=True. degree=2 adds numeric
    squares, without interactions. fit never reads validation/test statistics.
    """

    def __init__(self, missing_codes=None, categorical_columns=(),
                 add_missing_indicators=False, degree=1, standardize=True,
                 add_bias=True, zero_codes=None, drop_columns=()):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree not in (1, 2):
            raise ValueError("degree must be 1 or 2.")
        for name, value in (("standardize", standardize), ("add_bias", add_bias),
                            ("add_missing_indicators", add_missing_indicators)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean.")
        self.missing_codes = self._codes(missing_codes, "missing_codes")
        self.zero_codes = self._codes(zero_codes, "zero_codes")
        for name, columns in (("categorical_columns", categorical_columns),
                              ("drop_columns", drop_columns)):
            if not isinstance(columns, (list, tuple)):
                raise ValueError(f"{name} must be a list of column indices.")
        self.categorical_columns = tuple(self._column(c) for c in categorical_columns)
        self.drop_columns = tuple(self._column(c) for c in drop_columns)
        for name, columns in (("categorical_columns", self.categorical_columns),
                              ("drop_columns", self.drop_columns)):
            if len(set(columns)) != len(columns):
                raise ValueError(f"{name} contains duplicates.")
        if set(self.categorical_columns) & set(self.drop_columns):
            raise ValueError("A column cannot be both categorical and dropped.")
        self.add_missing_indicators = add_missing_indicators
        self.degree = degree
        self.standardize = standardize
        self.add_bias = add_bias

    @staticmethod
    def _column(value):
        if isinstance(value, bool):
            raise ValueError("Column indices must be nonnegative integers.")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Column indices must be nonnegative integers.") from exc
        if index < 0 or str(index) != str(value):
            raise ValueError("Column indices must be nonnegative integers.")
        return index

    @classmethod
    def _codes(cls, mapping, name):
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"{name} must map column indices to code lists.")
        result = {}
        for column, codes in (mapping or {}).items():
            values = np.asarray(codes, dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be lists of finite numbers.")
            result[cls._column(column)] = values
        return result

    def _clean(self, X):
        X = np.array(X, dtype=float, copy=True)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("Features must be a two-dimensional array with columns.")
        if np.any(np.isinf(X)):
            raise ValueError("Features contain infinity; use NaN for missing values.")
        for codes_by_column, value in ((self.missing_codes, np.nan), (self.zero_codes, 0.0)):
            for column, codes in codes_by_column.items():
                if column >= X.shape[1]:
                    raise ValueError(f"Code column {column} is out of range.")
                X[np.isin(X[:, column], codes), column] = value
        return X

    def _encode(self, X):
        numeric = X[:, self.numeric_columns_]
        numeric = np.where(np.isnan(numeric), self.medians_, numeric)
        parts = [numeric]
        if self.degree == 2:
            with np.errstate(over="ignore", invalid="ignore"):
                parts.append(numeric ** 2)
        for column, levels, missing in self.categories_:
            parts.extend((X[:, column] == level)[:, None] for level in levels[1:])
            if missing:
                parts.append(np.isnan(X[:, column])[:, None])
        if self.add_missing_indicators:
            parts.append(np.isnan(X[:, self.indicator_columns_]).astype(float))
        encoded = np.column_stack(parts)
        if not np.all(np.isfinite(encoded)):
            raise ValueError("Feature generation produced nonfinite values.")
        return encoded

    def fit(self, X):
        """Learn schema, medians, levels, retained features, and scaling."""
        X = self._clean(X)
        if len(X) == 0:
            raise ValueError("Cannot fit preprocessing on an empty training set.")
        self.n_features_in_ = X.shape[1]
        if any(c >= X.shape[1] for c in self.categorical_columns + self.drop_columns):
            raise ValueError("A categorical or dropped column is out of range.")
        excluded = set(self.categorical_columns) | set(self.drop_columns)
        self.numeric_columns_ = [c for c in range(X.shape[1])
                                 if c not in excluded and not np.all(np.isnan(X[:, c]))]
        self.medians_ = np.nanmedian(X[:, self.numeric_columns_], axis=0)
        self.categories_ = [
            (c, np.unique(X[~np.isnan(X[:, c]), c]), bool(np.any(np.isnan(X[:, c]))))
            for c in self.categorical_columns
        ]
        self.indicator_columns_ = np.array(
            [c for c in np.flatnonzero(np.any(np.isnan(X), axis=0))
             if c not in self.drop_columns], dtype=int)
        encoded = self._encode(X)
        self.means_ = encoded.mean(axis=0)
        self.scales_ = encoded.std(axis=0)
        if not np.all(np.isfinite(self.means_)) or not np.all(np.isfinite(self.scales_)):
            raise ValueError("Features are too large to scale safely.")
        self.keep_ = self.scales_ > 0
        if not np.any(self.keep_) and not self.add_bias:
            raise ValueError("No nonconstant features remain; enable add_bias.")
        return self

    def transform(self, X):
        """Apply the fitted training transformation without changing its state."""
        if not hasattr(self, "keep_"):
            raise ValueError("Call fit before transform.")
        X = self._clean(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from the fitted training schema.")
        encoded = self._encode(X)[:, self.keep_]
        if self.standardize:
            encoded = ((encoded - self.means_[self.keep_]) /
                       self.scales_[self.keep_])
        if not np.all(np.isfinite(encoded)):
            raise ValueError("Scaling produced nonfinite values.")
        if self.add_bias:
            encoded = np.column_stack((np.ones(len(X)), encoded))
        return encoded

    def fit_transform(self, X):
        return self.fit(X).transform(X)
