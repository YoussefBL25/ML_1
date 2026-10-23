"""Checks that held-out rows cannot influence preprocessing or selection."""

import unittest
from unittest.mock import patch

import numpy as np

from cross_validation import cross_validate, out_of_fold_scores, stratified_k_fold
from metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess import Preprocessor
from threshold import best_threshold
from tuning import nested_grid_search, select_candidate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(48, 3))
        self.y = np.tile([0., 1.], 24)
        self.X[:, 0] += self.y * 2
        self.candidates = [
            {"model": "ridge_regression", "params": {"lambda_": penalty}}
            for penalty in (0.001, 0.1)
        ]

    def test_folds_cover_all_rows_and_preserve_classes(self):
        folds = stratified_k_fold(self.y, 4, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(48))
        for fold, repeated in zip(folds, stratified_k_fold(self.y, 4, 0)):
            np.testing.assert_array_equal(fold, repeated)
            self.assertEqual(np.sum(self.y[fold]), 6)
        for k in (1, 25, 2.5):
            with self.assertRaises(ValueError):
                stratified_k_fold(self.y, k)
        with self.assertRaises(ValueError):
            stratified_k_fold(np.zeros(10), 2)

    def test_each_fold_fits_only_its_training_features(self):
        fits = []

        class SpyProcessor(Preprocessor):
            def fit(self, X):
                fits.append(X.copy())
                return super().fit(X)

        def train(y, X, params):
            return np.zeros(X.shape[1])

        scores = out_of_fold_scores(self.y, self.X, train, lambda X, w: X @ w,
                                    {}, k=4, preprocessor_factory=SpyProcessor)
        self.assertEqual(len(fits), 4)
        for actual, val_idx in zip(fits, stratified_k_fold(self.y, 4)):
            np.testing.assert_array_equal(actual, np.delete(self.X, val_idx, axis=0))
        np.testing.assert_array_equal(scores, np.zeros(48))

    def test_threshold_tuning_excludes_outer_labels(self):
        tuned_labels = []

        def track(y, scores, thresholds):
            tuned_labels.append(y.copy())
            return best_threshold(y, scores, thresholds)

        from models import score_model, train_model
        config = self.candidates[0]
        with patch("cross_validation.best_threshold", side_effect=track):
            result = cross_validate(
                self.y, self.X, lambda y, X, p: train_model(config, y, X)[0],
                lambda X, w: score_model(config, X, w), {}, k=4, inner_k=3)
        self.assertEqual(len(tuned_labels), 5)  # Four training choices, one deployment choice.
        for labels, idx in zip(tuned_labels[:-1], stratified_k_fold(self.y, 4)):
            np.testing.assert_array_equal(labels, np.delete(self.y, idx))
        self.assertTrue(0 <= result[1] <= 1)

    def test_model_selection_excludes_outer_rows(self):
        seen_features = []

        def select(y, X, candidates, k, seed, thresholds):
            seen_features.append(X.copy())
            return select_candidate(y, X, candidates, k, seed, thresholds)

        with patch("tuning.select_candidate", side_effect=select):
            result = nested_grid_search(self.y, self.X, self.candidates, k=3, inner_k=2)
        self.assertEqual(len(seen_features), 4)
        for actual, idx in zip(seen_features[:-1], stratified_k_fold(self.y, 3)):
            np.testing.assert_array_equal(actual, np.delete(self.X, idx, axis=0))
        np.testing.assert_array_equal(seen_features[-1], self.X)
        self.assertEqual(result["protocol"], "nested_stratified_cv")
        self.assertEqual(len(result["outer_folds"]), 3)
        self.assertIn("selection_f1", result["best"])
        self.assertTrue(0 <= result["outer_mean_f1"] <= 1)

    def test_threshold_search_matches_brute_force(self):
        scores = np.array([-2., 0.1, 0.1, 0.5, 0.5, 3.])
        y = np.array([0., 1., 0., 1., 1., 0.])
        grid = np.array([-3., 0.1, 0.3, 0.5, 4.])
        expected_f1s = [f1_score(y, (scores >= t).astype(int)) for t in grid]
        threshold, f1 = best_threshold(y, scores, grid[::-1])
        self.assertEqual(threshold, grid[np.argmax(expected_f1s)])
        self.assertEqual(f1, max(expected_f1s))
        self.assertEqual(f1_score([0, 0], [0, 0]), 0.)
        for scores in ([np.nan] * 6, [0., 1.]):
            with self.assertRaises(ValueError):
                best_threshold(y, scores)
        with self.assertRaises(ValueError):
            f1_score([0, 1], [1])
        with self.assertRaises(ValueError):
            f1_score([0, 2], [1, 1])

    def test_metrics_from_confusion_counts(self):
        # TP = 2, FP = 1, FN = 2, TN = 2
        y_true, y_pred = [1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0]
        self.assertAlmostEqual(precision_score(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(recall_score(y_true, y_pred), 2 / 4)
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 4 / 7)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 4 / 7)
        for metric in (precision_score, recall_score):
            self.assertEqual(metric([0, 0], [0, 0]), 0.)
            with self.assertRaises(ValueError):
                metric([0, 2], [1, 1])

    def test_bad_model_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scores"):
            out_of_fold_scores(self.y, self.X, lambda y, X, p: np.zeros(4),
                               lambda X, w: np.full(len(X), np.nan), {}, k=3)


if __name__ == "__main__":
    unittest.main()
