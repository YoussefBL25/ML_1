"""End-to-end tests using tiny generated CSVs in temporary directories."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data import ROOT, load_data
from helpers import create_csv_submission
from models import load_config, validate_config
from run import generate_submission


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(13)
        self.y = np.tile([-1, 1], 18)
        self.X = rng.normal(size=(36, 3))
        self.X[:, 0] += (self.y == 1) * 1.5
        self.X[0, 1] = np.nan
        self.ids = np.arange(100, 136)
        self.test_ids = np.array([202, 201, 205])
        np.savetxt(self.data / "x_train.csv", np.c_[self.ids, self.X], delimiter=",",
                   header="Id,a,b,c", comments="")
        np.savetxt(self.data / "y_train.csv", np.c_[self.ids, self.y], delimiter=",",
                   header="Id,Prediction", comments="", fmt="%d")
        np.savetxt(self.data / "x_test.csv", np.c_[self.test_ids, self.X[3:6]], delimiter=",",
                   header="Id,a,b,c", comments="")

    def test_cache_reloads_changed_source_and_recovers_legacy_cache(self):
        first = load_data(self.data)
        cache = self.data / "cache.npz"
        self.assertTrue(cache.exists())
        for actual, expected in zip(load_data(self.data), first):
            np.testing.assert_array_equal(actual, expected)
        np.savetxt(self.data / "y_train.csv", np.c_[self.ids, -self.y], delimiter=",",
                   header="Id,Prediction", comments="", fmt="%d")
        np.testing.assert_array_equal(load_data(self.data)[2], -self.y)
        np.savez(cache, obsolete=np.ones(2))
        np.testing.assert_array_equal(load_data(self.data)[2], -self.y)
        cache.write_bytes(b"invalid cache")
        np.testing.assert_array_equal(load_data(self.data)[2], -self.y)

    def test_reordered_labels_are_rejected(self):
        np.savetxt(self.data / "y_train.csv", np.c_[self.ids[::-1], self.y], delimiter=",",
                   header="Id,Prediction", comments="", fmt="%d")
        with self.assertRaisesRegex(ValueError, "Ids must match"):
            load_data(self.data, use_cache=False)

    def test_single_test_row_preserves_matrix_shape(self):
        np.savetxt(self.data / "x_test.csv", [[200, 1., 2., 3.]], delimiter=",",
                   header="Id,a,b,c", comments="")
        arrays = load_data(self.data, use_cache=False)
        self.assertEqual(arrays[1].shape, (1, 3))

    def test_submission_is_repeatable_and_keeps_original_id_order(self):
        config = load_config(ROOT / "config" / "model.json")
        path = self.root / "outputs" / "submission.csv"
        generate_submission(config, self.data, path)
        first = path.read_bytes()
        generate_submission(config, self.data, path)
        self.assertEqual(first, path.read_bytes())
        with path.open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, ["Id", "Prediction"])
        self.assertEqual([int(row["Id"]) for row in rows], self.test_ids.tolist())
        self.assertTrue(all(int(row["Prediction"]) in (-1, 1) for row in rows))

    def test_invalid_submission_cannot_truncate_existing_output(self):
        output = self.root / "existing.csv"
        output.write_text("existing content", encoding="utf-8")
        for ids, predictions in (([1, 2], [1]), ([1, 1], [1, -1]),
                                 ([1, 2], [1, 0]), ([1.5], [1])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                create_csv_submission(ids, predictions, output)
            self.assertEqual(output.read_text(), "existing content")

    def test_interaction_submission_and_evaluation_cli(self):
        base = self.root / "base.json"
        base.write_text("{}", encoding="utf-8")
        generated = self.root / "nested" / "best.json"
        prepared = subprocess.run([
            sys.executable, str(ROOT / "interactions.py"), "--data-dir", str(self.data),
            "--base", str(base), "--no-cache", "--output", str(generated),
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(json.loads(generated.read_text())["model"], "interaction_logistic_regression")
        config = {"model": "interaction_logistic_regression",
                  "params": {"gamma": 0.1, "lambda_": 0.001, "max_iters": 10, "n_top": 3},
                  "threshold": 0.5, "preprocessing": {"add_missing_indicators": True}}
        config_path = self.root / "interaction.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = self.root / "interaction.csv"
        command = [sys.executable, str(ROOT / "run.py"), "--data-dir", str(self.data),
                   "--config", str(config_path), "--output", str(output)]
        first = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        contents = output.read_bytes()
        second = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(contents, output.read_bytes())
        with output.open() as stream:
            self.assertEqual([int(row["Id"]) for row in csv.DictReader(stream)], self.test_ids.tolist())
        evaluation = self.root / "nested" / "evaluation.json"
        result = subprocess.run([
            sys.executable, str(ROOT / "evaluate_best.py"), "--data-dir", str(self.data),
            "--config", str(config_path), "--folds", "3", "--inner-folds", "2",
            "--seed", "4", "--n-top", "0", "--no-cache", "--output", str(evaluation),
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(evaluation.read_text())
        self.assertEqual(saved["selection_scope"], "threshold_only")
        self.assertEqual(saved["config"]["params"]["n_top"], 0)
        self.assertEqual(saved["dataset_directory"], str(self.data.resolve()))
        json.dumps(saved, allow_nan=False)

    def test_nonfinite_model_outputs_cannot_overwrite_submission(self):
        config = {"model": "interaction_logistic_regression",
                  "params": {"gamma": 0.1, "lambda_": 0.001, "max_iters": 10, "n_top": 3}}
        output = self.root / "existing.csv"
        output.write_text("existing content", encoding="utf-8")
        with patch("run.model_from_config") as factory:
            model = factory.return_value.fit.return_value
            model.loss_ = 0.5
            model.predict_proba.return_value = np.full(3, np.nan)
            with self.assertRaisesRegex(ValueError, "finite"):
                generate_submission(config, self.data, output)
        self.assertEqual(output.read_text(), "existing content")

    def test_missing_data_has_actionable_error(self):
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--data-dir",
                                 str(self.root / "absent")], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Dataset files are missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_nested_tuning_config_runs_from_another_directory(self):
        candidates = [
            {"model": "ridge_regression", "params": {"lambda_": 0.01}},
            {"model": "logistic_regression", "params": {"gamma": 0.03, "max_iters": 10},
             "preprocessing": {"add_missing_indicators": True}},
        ]
        candidate_path = self.root / "candidates.json"
        candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
        selected = self.root / "selected.json"
        results = self.root / "nested.json"
        tuned = subprocess.run([
            sys.executable, str(ROOT / "tune_models.py"), "--data-dir", str(self.data),
            "--candidates", str(candidate_path), "--folds", "3", "--inner-folds", "2",
            "--output", str(results), "--model-output", str(selected),
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(tuned.returncode, 0, tuned.stderr)
        self.assertEqual(json.loads(results.read_text())["protocol"], "nested_stratified_cv")
        validate_config(json.loads(selected.read_text()))
        submission = self.root / "submission.csv"
        command = [sys.executable, str(ROOT / "run.py"), "--data-dir", str(self.data),
                   "--config", str(selected), "--output", str(submission)]
        first = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        contents = submission.read_bytes()
        second = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(contents, submission.read_bytes())

    def test_config_rejects_typos_and_invalid_parameters(self):
        for config in (
            {"model": "unknown"},
            {"model": "ridge_regression", "params": {"lambda_": -1}},
            {"model": "ridge_regression", "params": {"lambda_": 0.1}, "typo": True},
            {"model": "least_squares", "preprocessing": {"standarize": True}},
            {"model": "logistic_regression", "params": {"gamma": 1., "max_iters": 0}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
