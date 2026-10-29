"""Numerical and validation checks for the final interaction model."""

import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cross_validation import stratified_k_fold
from evaluate_best import evaluate_config, inner_threshold, main as evaluate_main
from interactions import InteractionModel, log_loss, validate_best_config
from preprocess import Preprocessor


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": "interaction_logistic_regression",
            "params": {"gamma": 0.1, "lambda_": 0.001, "max_iters": 20, "n_top": 3},
            "threshold": 0.5,
            "preprocessing": {"add_missing_indicators": True},
        }
        self.y = np.tile([0., 1.], 18)
        self.X = np.random.default_rng(12).normal(size=(36, 4))
        self.X[:, 0] += self.y
        self.X[0, 2] = np.nan

    def test_saturated_float32_probabilities_have_finite_loss(self):
        with np.errstate(all="raise"):
            self.assertLess(log_loss([0, 1], np.array([0, 1], dtype=np.float32)), 1e-10)
            self.assertGreater(log_loss([1, 0], np.array([0, 1], dtype=np.float32)), 20)
        for probabilities in ([np.nan, 0.5], [np.inf, 0], [-0.1, 0.5], [0.5, 1.1], [0.5]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                log_loss([0, 1], probabilities)

    def test_invalid_interaction_configs_are_rejected(self):
        variants = []
        for key, value in (("n_top", -1), ("n_top", True), ("n_top", 2.5),
                           ("gamma", float("nan")), ("max_iters", 0), ("typo", 1)):
            config = copy.deepcopy(self.config)
            config["params"][key] = value
            variants.append(config)
        variants.extend([
            dict(self.config, threshold=float("nan")),
            dict(self.config, preprocessing={"add_bias": False}),
            dict(self.config, preprocessing={"standarize": True}),
        ])
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_best_config(config)

    def test_fit_and_predictions_are_repeatable_and_keep_training_scaling(self):
        X = Preprocessor().fit_transform(self.X)
        first = InteractionModel(**self.config["params"]).fit(self.y, X)
        second = InteractionModel(**self.config["params"]).fit(self.y, X)
        means, scales = first.means_.copy(), first.scales_.copy()
        np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
        self.assertTrue(np.isfinite(first.loss_))
        changed = X.copy()
        changed[:, 1:] *= 10
        self.assertTrue(np.all(np.isfinite(first.predict_proba(changed))))
        np.testing.assert_array_equal(first.means_, means)
        np.testing.assert_array_equal(first.scales_, scales)

    def test_invalid_features_and_divergence_are_rejected(self):
        model = InteractionModel()
        with self.assertRaisesRegex(ValueError, "fit"):
            model.predict_proba([[1, 2]])
        for X in ([[1, np.nan], [1, 2]], [[0, 2], [0, 3]], [[1, np.inf], [1, 2]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                model.fit([0, 1], X)
        with self.assertRaisesRegex(ValueError, "diverged"):
            InteractionModel(gamma=1e38, max_iters=3).fit([0, 1], [[1, -10], [1, 10]])

    def test_evaluation_selects_thresholds_without_outer_rows(self):
        seen = []

        def select(config, y, X, k, seed):
            seen.append((y.copy(), X.copy()))
            return inner_threshold(config, y, X, k, seed)

        with patch("evaluate_best.inner_threshold", side_effect=select), redirect_stdout(io.StringIO()):
            result = evaluate_config(self.config, self.y, self.X, k=3, inner_k=2, seed=7)
        for (y, X), idx in zip(seen, stratified_k_fold(self.y, 3, 7)):
            np.testing.assert_array_equal(y, np.delete(self.y, idx))
            np.testing.assert_array_equal(X, np.delete(self.X, idx, axis=0))
        self.assertEqual(result["selection_scope"], "threshold_only")
        self.assertEqual(result["config"]["params"], self.config["params"])
        self.assertEqual(result["seed"], 7)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_evaluation_cannot_overwrite_saved_results(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps(self.config), encoding="utf-8")
            output = Path(directory) / "evaluation.json"
            output.write_text("existing results", encoding="utf-8")
            with patch("evaluate_best.load_data", return_value=(self.X, None, self.y, None, None)), \
                    patch("evaluate_best.evaluate_config", return_value={"loss": float("nan")}), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate_main(["--config", str(config), "--output", str(output)])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_text(), "existing results")


if __name__ == "__main__":
    unittest.main()
