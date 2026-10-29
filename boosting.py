"""Histogram gradient boosting for binary classification, using only NumPy.

Trees optimize logistic loss with regularized Newton leaf values. Quantile bins,
nominal levels, and missing-value routing are learned from training rows only.
The six required course implementations are independent of this module.
"""

import multiprocessing

import numpy as np

from interactions import log_loss, model_from_config, sigmoid, validate_best_config
from preprocess import Preprocessor, binary_labels


class HistogramBinner:
    """Training quantiles for numeric values; one-hot bins for nominal values.

    Bin zero always means missing. All other bins are ordered, starting at one.
    Nominal levels not seen in training match no one-hot indicator.
    With code_indicator_min > 0, each configured missing code (such as "don't
    know" or "refused") seen at least that often in training also gets its own
    raw-value indicator, so a blank ("not asked") answer stays distinguishable.
    """

    def __init__(self, preprocessing=None, max_bins=32, seed=0, code_indicator_min=0):
        if isinstance(max_bins, bool) or not isinstance(max_bins, int) or not 4 <= max_bins <= 255:
            raise ValueError("max_bins must be an integer from 4 through 255.")
        if (isinstance(code_indicator_min, bool) or not isinstance(code_indicator_min, int) or
                code_indicator_min < 0):
            raise ValueError("code_indicator_min must be a nonnegative integer.")
        self.processor = Preprocessor(**(preprocessing or {}))
        self.max_bins, self.seed = max_bins, seed
        self.code_indicator_min = code_indicator_min

    def fit(self, X):
        raw = np.asarray(X, dtype=float)
        X = self.processor._clean(X)
        if len(X) == 0:
            raise ValueError("Cannot fit bins on empty features.")
        self.n_features_in_ = X.shape[1]
        dropped = set(self.processor.drop_columns)
        categorical = set(self.processor.categorical_columns)
        if any(c >= X.shape[1] for c in dropped | categorical):
            raise ValueError("A categorical or dropped column is out of range.")
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(len(X), min(len(X), 50_000), replace=False)
        self.columns_ = []
        for column in range(X.shape[1]):
            if column in dropped:
                continue
            values = X[:, column]
            observed = values[~np.isnan(values)]
            if len(observed) == 0:
                continue
            if column in categorical:
                for level in np.unique(observed):
                    self.columns_.append((column, float(level), None))
            else:
                unique = np.unique(observed)
                if len(unique) <= self.max_bins - 1:
                    # Using observed endpoints avoids overflowing midpoint sums.
                    edges = unique[1:]
                else:
                    sampled = values[sample]
                    sampled = sampled[~np.isnan(sampled)]
                    if len(sampled) == 0:
                        sampled = observed
                    edges = np.unique(np.quantile(
                        sampled, np.arange(1, self.max_bins - 1) / (self.max_bins - 1)))
                self.columns_.append((column, None, edges))
        if not self.columns_:
            raise ValueError("No observed tree features remain.")
        self.code_columns_ = []
        if self.code_indicator_min:
            for column, codes in sorted(self.processor.missing_codes.items()):
                if column not in dropped:
                    self.code_columns_ += [(column, float(code)) for code in codes
                                           if np.sum(raw[:, column] == code) >= self.code_indicator_min]
        return self

    def transform(self, X):
        if not hasattr(self, "columns_"):
            raise ValueError("Call fit before transform.")
        raw = np.asarray(X, dtype=float)
        X = self.processor._clean(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted bins.")
        bins = np.empty((len(X), len(self.columns_) + len(self.code_columns_)),
                        dtype=np.uint8, order="F")
        for j, (column, level, edges) in enumerate(self.columns_):
            values = X[:, column]
            if level is None:
                bins[:, j] = 1 + np.searchsorted(edges, values, side="right")
            else:
                bins[:, j] = 1 + (values == level)
            bins[np.isnan(values), j] = 0
        for j, (column, code) in enumerate(self.code_columns_, len(self.columns_)):
            values = raw[:, column]
            bins[:, j] = np.where(np.isnan(values), 0, 1 + (values == code))
        return bins

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class HistogramGradientBoosting:
    """Shallow histogram trees with row/feature sampling and L2 leaf shrinkage."""

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.08,
                 max_bins=32, min_samples_leaf=100, l2=10., subsample=0.7,
                 feature_fraction=0.8, seed=0):
        for name, value, low, high in (("n_estimators", n_estimators, 1, 5000),
                                       ("max_depth", max_depth, 1, 8),
                                       ("max_bins", max_bins, 4, 255),
                                       ("min_samples_leaf", min_samples_leaf, 1, 1_000_000)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}].")
        for name, value, lower, upper in (("learning_rate", learning_rate, 0, 1),
                                          ("subsample", subsample, 0, 1),
                                          ("feature_fraction", feature_fraction, 0, 1)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not np.isfinite(value) or not lower < value <= upper):
                raise ValueError(f"{name} must be finite and in (0, 1].")
        if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.learning_rate, self.max_bins = learning_rate, max_bins
        self.min_samples_leaf, self.l2 = min_samples_leaf, l2
        self.subsample, self.feature_fraction, self.seed = subsample, feature_fraction, seed

    def _bins(self, X):
        X = np.asarray(X)
        if (X.ndim != 2 or X.shape[1] == 0 or X.dtype != np.uint8 or
                (X.size and X.max() >= self.max_bins)):
            raise ValueError("Tree inputs must be uint8 bins smaller than max_bins.")
        return X

    def _grow(self, X, gradient, hessian, rows, features):
        """Grow one tree level by level, with one histogram pass per feature and level.

        Each node's sums accumulate its rows in the same order as a depth-first
        build, so the splits and leaf values match it exactly.
        """
        B = self.max_bins
        bins = np.empty((len(rows), len(features)), dtype=np.uint8, order="F")
        for j, feature in enumerate(features):
            bins[:, j] = X[rows, feature]
        g, h = gradient[rows], hessian[rows]
        root = [None]
        # Open nodes: (parent container, slot, positions into rows, ascending).
        level = [(root, 0, np.arange(len(rows)))]
        for depth in range(self.max_depth + 1):
            splittable = []
            for parent, slot, positions in level:
                G, H = g[positions].sum(), h[positions].sum()
                parent[slot] = float(np.clip(G / (H + self.l2), -5, 5))
                if depth < self.max_depth and len(positions) >= 2 * self.min_samples_leaf:
                    splittable.append((parent, slot, positions, G, H))
            if not splittable:
                break
            m = len(splittable)
            offsets = np.full(len(rows), m * B, dtype=np.intp)  # Other rows: ignored slot.
            for k, (_, _, positions, _, _) in enumerate(splittable):
                offsets[positions] = k * B
            n = np.array([len(s[2]) for s in splittable])[:, None, None]
            G = np.array([s[3] for s in splittable])[:, None, None]
            H = np.array([s[4] for s in splittable])[:, None, None]
            parent_score = G * G / (H + self.l2)
            best_gain = np.full(m, 1e-10)
            best = np.full((m, 2), -1, dtype=np.intp)  # (local feature, flat cut index)
            for j in range(len(features)):
                codes = offsets + bins[:, j]
                size = (m + 1) * B
                counts = np.bincount(codes, minlength=size).reshape(m + 1, B)[:m]
                gradients = np.bincount(codes, weights=g, minlength=size).reshape(m + 1, B)[:m]
                hessians = np.bincount(codes, weights=h, minlength=size).reshape(m + 1, B)[:m]
                # Candidate cuts 0..max_bins-2 include a missing-versus-observed split.
                left_count = np.concatenate((np.zeros((m, 1), dtype=counts.dtype),
                                             np.cumsum(counts[:, 1:-1], axis=1)), axis=1)
                left_g = np.concatenate((np.zeros((m, 1)), np.cumsum(gradients[:, 1:-1], axis=1)), axis=1)
                left_h = np.concatenate((np.zeros((m, 1)), np.cumsum(hessians[:, 1:-1], axis=1)), axis=1)
                C = np.stack((left_count, left_count + counts[:, :1]), axis=1)
                GL = np.stack((left_g, left_g + gradients[:, :1]), axis=1)
                HL = np.stack((left_h, left_h + hessians[:, :1]), axis=1)
                valid = ((C >= self.min_samples_leaf) &
                         (n - C >= self.min_samples_leaf) &
                         (HL > 1e-9) & (H - HL > 1e-9))
                # Every split must leave the largest bin intact on one side.
                valid[n[:, 0, 0] - counts.max(axis=1) < self.min_samples_leaf] = False
                Gb, Hb = np.broadcast_to(G, C.shape), np.broadcast_to(H, C.shape)
                gains = np.full(C.shape, -np.inf)
                gains[valid] = (GL[valid] ** 2 / (HL[valid] + self.l2) +
                                (Gb[valid] - GL[valid]) ** 2 / (Hb[valid] - HL[valid] + self.l2) -
                                np.broadcast_to(parent_score, C.shape)[valid])
                flat = gains.reshape(m, -1)
                index = np.argmax(flat, axis=1)
                gain = flat[np.arange(m), index]
                better = gain > best_gain
                best_gain[better] = gain[better]
                best[better] = np.stack((np.full(m, j), index), axis=1)[better]
            level = []
            for (parent, slot, positions, _, _), (j, index) in zip(splittable, best):
                if j < 0:
                    continue
                missing_left, cut = divmod(int(index), B - 1)
                values = bins[positions, j]
                goes_left = np.where(values == 0, bool(missing_left), values <= cut)
                node = [int(features[j]), cut, bool(missing_left), None, None]
                parent[slot] = node
                level += [(node, 3, positions[goes_left]), (node, 4, positions[~goes_left])]

        def freeze(node):
            return node if isinstance(node, float) else (*node[:3], freeze(node[3]), freeze(node[4]))

        return freeze(root[0])

    @staticmethod
    def _predict_tree(tree, X):
        result = np.empty(len(X), dtype=float)

        def visit(node, rows):
            if isinstance(node, float):
                result[rows] = node
                return
            feature, cut, missing_left, left, right = node
            values = X[rows, feature]
            mask = np.where(values == 0, missing_left, values <= cut)
            visit(left, rows[mask])
            visit(right, rows[~mask])

        visit(tree, np.arange(len(X)))
        return result

    def fit(self, y, X, progress=False):
        y, X = binary_labels(y), self._bins(X)
        if len(y) != len(X):
            raise ValueError("Features and labels must have equal row counts.")
        self.n_features_in_ = X.shape[1]
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.intercept_ = float(np.log(prior / (1 - prior)))
        scores = np.full(len(y), self.intercept_)
        self.trees_ = []
        rng = np.random.default_rng(self.seed)
        count = max(1, int(len(X) * self.subsample))
        feature_count = max(1, int(X.shape[1] * self.feature_fraction))
        for iteration in range(self.n_estimators):
            probability = sigmoid(scores)
            gradient = y - probability
            hessian = np.maximum(probability * (1 - probability), 1e-6)
            rows = np.sort(rng.choice(len(X), count, replace=False))
            features = np.sort(rng.choice(X.shape[1], feature_count, replace=False))
            tree = self._grow(X, gradient, hessian, rows, features)
            self.trees_.append(tree)
            scores += self.learning_rate * self._predict_tree(tree, X)
            if progress and (iteration + 1) % 25 == 0:
                print(f"Tree {iteration + 1}/{self.n_estimators}: training loss "
                      f"{log_loss(y, sigmoid(scores)):.6f}", flush=True)
        self.loss_ = log_loss(y, sigmoid(scores))
        return self

    def staged_predict_proba(self, X, checkpoints):
        if not hasattr(self, "trees_"):
            raise ValueError("Call fit before prediction.")
        X = self._bins(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted trees.")
        checkpoints = set(checkpoints)
        if not checkpoints or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or
                                   not 1 <= c <= len(self.trees_) for c in checkpoints):
            raise ValueError("Checkpoints must be within the fitted tree count.")
        scores = np.full(len(X), self.intercept_)
        for i, tree in enumerate(self.trees_, 1):
            scores += self.learning_rate * self._predict_tree(tree, X)
            if i in checkpoints:
                yield i, sigmoid(scores)

    def predict_proba(self, X):
        if not hasattr(self, "trees_"):
            raise ValueError("Call fit before prediction.")
        return next(self.staged_predict_proba(X, [len(self.trees_)]))[1]


def validate_boosted_config(config):
    if not isinstance(config, dict) or config.get("model") != "histogram_boosted_blend":
        raise ValueError("Expected a histogram_boosted_blend configuration.")
    unknown = set(config) - {"model", "params", "preprocessing", "threshold", "tree_weight",
                             "logistic_config", "status", "seeds", "binning"}
    if unknown:
        raise ValueError(f"Unknown boosted configuration keys: {sorted(unknown)}")
    for key in ("params", "preprocessing", "binning"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a JSON object.")
    params = dict(config.get("params", {}))
    binning = dict(config.get("binning", {}))
    try:
        model = HistogramGradientBoosting(**params)
        settings = dict(config.get("preprocessing", {}))
        HistogramBinner(settings, model.max_bins, model.seed, **binning)
    except TypeError as exc:
        raise ValueError(f"Invalid boosted configuration: {exc}") from exc
    seeds = config.get("seeds", [model.seed])
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds) or
            any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("seeds must be a list of 1 to 10 distinct integers in [0, 2**32).")
    cutoff, weight = config.get("threshold", .21), config.get("tree_weight", 1.)
    for name, value in (("threshold", cutoff), ("tree_weight", weight)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and in [0, 1].")
    logistic = config.get("logistic_config")
    if weight < 1:
        logistic = validate_best_config(logistic)
        if logistic["model"] != "interaction_logistic_regression":
            raise ValueError("The blend requires an interaction logistic configuration.")
    return dict(config, params=params, preprocessing=settings, binning=binning, seeds=list(seeds),
                threshold=float(cutoff), tree_weight=float(weight), logistic_config=logistic)


def tree_member(config, seed):
    """Unfitted (binner, trees) for one seed; the binner's quantile sample uses it too."""
    model = HistogramGradientBoosting(**dict(config["params"], seed=seed))
    binner = HistogramBinner(config["preprocessing"], model.max_bins, seed, **config["binning"])
    return binner, model


_MEMBER_JOBS = []  # Read by forked workers, so the bins are shared, not pickled.


def _fit_member(index):
    model, y, bins, progress = _MEMBER_JOBS[index]
    return model.fit(y, bins, progress=progress)


def fit_members(jobs, workers=1):
    """Fit (model, y, bins, progress) jobs, in parallel worker processes if workers > 1."""
    if workers <= 1 or len(jobs) == 1:
        return [model.fit(y, bins, progress=progress) for model, y, bins, progress in jobs]
    _MEMBER_JOBS[:] = jobs
    try:
        with multiprocessing.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            return pool.map(_fit_member, range(len(jobs)), chunksize=1)
    finally:
        _MEMBER_JOBS.clear()


class BoostedPipeline:
    """Raw-feature boosted trees, averaged over seeds, with an optional logistic blend."""

    def __init__(self, config):
        self.config = validate_boosted_config(config)

    def fit(self, y, X, progress=False, workers=1):
        y = binary_labels(y)
        members = [tree_member(self.config, seed) for seed in self.config["seeds"]]
        self.binners_ = [binner for binner, _ in members]
        bins = [binner.fit_transform(X) for binner in self.binners_]
        self.models_ = fit_members([(model, y, b, progress) for (_, model), b in zip(members, bins)],
                                   workers)
        scores = np.mean([m.predict_proba(b) for m, b in zip(self.models_, bins)], axis=0)
        del bins
        weight = self.config["tree_weight"]
        if weight < 1:
            config = self.config["logistic_config"]
            self.processor_ = Preprocessor(**config["preprocessing"])
            features = self.processor_.fit_transform(X)
            self.logistic_ = model_from_config(config).fit(y, features)
            scores = weight * scores + (1 - weight) * self.logistic_.predict_proba(features)
        self.loss_ = log_loss(y, scores)
        return self

    def tree_proba(self, X):
        if not hasattr(self, "models_"):
            raise ValueError("Call fit before prediction.")
        return np.mean([m.predict_proba(b.transform(X))
                        for m, b in zip(self.models_, self.binners_)], axis=0)

    def predict_proba(self, X):
        scores = self.tree_proba(X)
        weight = self.config["tree_weight"]
        if weight < 1:
            scores = weight * scores + (1 - weight) * self.logistic_.predict_proba(
                self.processor_.transform(X))
        return scores
