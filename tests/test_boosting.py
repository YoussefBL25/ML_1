"""Analytic tree updates, missing routing, nonlinear fitting and submission."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from boosting import (HistogramBinner, HistogramGradientBoosting,
                      validate_boosted_config)
from data import ROOT
from run import generate_submission


class BoostingTests(unittest.TestCase):
    def test_newton_stump_matches_analytic_update(self):
        X = np.array([[1], [1], [2], [2]], dtype=np.uint8)
        model = HistogramGradientBoosting(n_estimators=1, max_depth=1, learning_rate=.5,
                                          l2=0, subsample=1, feature_fraction=1,
                                          min_samples_leaf=1).fit([0, 0, 1, 1], X)
        expected = 1 / (1 + np.exp(-np.array([-1, -1, 1, 1])))
        np.testing.assert_allclose(model.predict_proba(X), expected)

    def test_missing_values_can_be_the_informative_split(self):
        X = np.array([[0], [0], [1], [1]], dtype=np.uint8)
        model = HistogramGradientBoosting(n_estimators=3, max_depth=1, learning_rate=.5,
                                          l2=0, subsample=1, min_samples_leaf=1).fit([1, 1, 0, 0], X)
        np.testing.assert_array_equal(model.predict_proba(X) >= .5, [True, True, False, False])

    def test_depth_two_learns_nonlinear_rule_and_repeats(self):
        X = np.tile([[1, 1], [1, 2], [2, 1], [2, 2]], (20, 1)).astype(np.uint8)
        y = np.tile([0, 0, 0, 1], 20)
        settings = dict(n_estimators=20, max_depth=2, learning_rate=.3,
                        l2=.1, min_samples_leaf=2, seed=7)
        first = HistogramGradientBoosting(**settings).fit(y, X)
        second = HistogramGradientBoosting(**settings).fit(y, X)
        np.testing.assert_array_equal(first.predict_proba(X) >= .5, y)
        np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
        staged = dict(first.staged_predict_proba(X, [1, 20]))
        np.testing.assert_array_equal(staged[20], first.predict_proba(X))
        self.assertTrue(np.isfinite(first.loss_))

    def test_bins_and_nominal_levels_use_only_training_values(self):
        X = np.array([[0, 10], [1, 20], [2, np.nan], [3, 10]], dtype=float)
        binner = HistogramBinner({"categorical_columns": [1]}, max_bins=8).fit(X)
        original_edges = binner.columns_[0][2].copy()
        bins = binner.transform([[1000, 30], [np.nan, np.nan]])
        self.assertTrue(np.all(bins[0, 1:] == 1))  # Unseen nominal value matches no level.
        self.assertTrue(np.all(bins[1] == 0))
        np.testing.assert_array_equal(original_edges, binner.columns_[0][2])
        self.assertLess(bins.max(), 8)

    def test_invalid_parameters_and_unfitted_predictions_fail(self):
        for params in ({"n_estimators": 0}, {"max_depth": True}, {"max_bins": 256},
                       {"l2": -1}, {"subsample": 0}, {"learning_rate": float("nan")}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                HistogramGradientBoosting(**params)
        with self.assertRaisesRegex(ValueError, "fit"):
            HistogramGradientBoosting().predict_proba(np.ones((2, 1), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            HistogramGradientBoosting().fit([0, 1], [[np.nan], [1.]])
        with self.assertRaises(ValueError):
            HistogramBinner().fit([[np.inf]])

    def test_boosted_pipeline_and_submission_keep_ids_and_repeat(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 3))
        y = np.tile([-1, 1], 30)
        X[:, 0] += y
        config = {"model": "histogram_boosted_blend",
                  "params": {"n_estimators": 5, "max_depth": 2, "max_bins": 8,
                             "min_samples_leaf": 2}, "preprocessing": {}, "threshold": .4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savetxt(path / "x_train.csv", np.c_[np.arange(60), X], delimiter=",",
                       header="Id,a,b,c", comments="")
            np.savetxt(path / "y_train.csv", np.c_[np.arange(60), y], delimiter=",",
                       header="Id,Prediction", comments="")
            np.savetxt(path / "x_test.csv", np.c_[[105, 101, 103], X[:3]], delimiter=",",
                       header="Id,a,b,c", comments="")
            for weight in (1., .5):
                candidate = dict(config, tree_weight=weight,
                                 logistic_config={"model": "interaction_logistic_regression",
                                     "params": {"gamma": .1, "lambda_": .001,
                                                "max_iters": 5, "n_top": 3}})
                output = path / "submission.csv"
                first = generate_submission(candidate, path, output)
                contents = output.read_bytes()
                np.testing.assert_array_equal(first, generate_submission(candidate, path, output))
                self.assertEqual(contents, output.read_bytes())
                self.assertEqual([line.split(',')[0] for line in output.read_text().splitlines()[1:]],
                                 ["105", "101", "103"])
            (path / "logistic.json").write_text(json.dumps(candidate["logistic_config"]))
            (path / "preprocessing.json").write_text("{}")
            result = subprocess.run([
                sys.executable, str(ROOT / "tune_boosting.py"), "--data-dir", str(path),
                "--logistic-config", str(path / "logistic.json"), "--preprocessing",
                str(path / "preprocessing.json"), "--depths", "2", "--checkpoints", "2,4",
                "--output", str(path / "screen.json"), "--model-output", str(path / "model.json"),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((path / "screen.json").read_text())
            self.assertEqual(saved["selected_by_tuning"]["tuning_f1"],
                             max(row["tuning_f1"] for row in saved["candidates"]))
            json.dumps(saved, allow_nan=False)
            selected = validate_boosted_config(json.loads((path / "model.json").read_text()))
            confirmed = subprocess.run([
                sys.executable, str(ROOT / "tune_boosting.py"), "--data-dir", str(path),
                "--logistic-config", str(path / "logistic.json"),
                "--candidate-config", str(path / "model.json"), "--seed", "2027",
                "--output", str(path / "confirm.json"),
                "--model-output", str(path / "confirm_model.json"),
            ], capture_output=True, text=True)
            self.assertEqual(confirmed.returncode, 0, confirmed.stderr)
            fixed = json.loads((path / "confirm.json").read_text())
            self.assertEqual(fixed["selection_scope"], "threshold_only")
            self.assertEqual(len(fixed["candidates"]), 1)
            self.assertEqual(fixed["selected_by_tuning"]["config"]["params"], selected["params"])

    def test_invalid_blend_configuration_is_rejected(self):
        for config in ({"model": "histogram_boosted_blend", "tree_weight": -.1},
                       {"model": "histogram_boosted_blend", "tree_weight": .5},
                       {"model": "histogram_boosted_blend", "threshold": float("nan")},
                       {"model": "histogram_boosted_blend", "params": {"typo": 1}},
                       {"model": "histogram_boosted_blend", "params": []},
                       {"model": "histogram_boosted_blend", "typo": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_boosted_config(config)


if __name__ == "__main__":
    unittest.main()
