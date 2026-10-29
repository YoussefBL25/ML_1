"""Level-wise trees, code indicators, seed ensembles and the staged search driver."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from boosting import BoostedPipeline, HistogramBinner, HistogramGradientBoosting
from data import ROOT
import search_boosting as sb


def reference_grow(model, X, gradient, hessian, rows, features, depth=0):
    """The earlier depth-first tree builder, kept as the specification."""
    g, h = gradient[rows], hessian[rows]
    G, H = g.sum(), h.sum()
    leaf = float(np.clip(G / (H + model.l2), -5, 5))
    if depth == model.max_depth or len(rows) < 2 * model.min_samples_leaf:
        return leaf
    best_gain, best = 1e-10, None
    parent_score = G * G / (H + model.l2)
    for feature in features:
        values = X[rows, feature]
        counts = np.bincount(values, minlength=model.max_bins)
        if len(rows) - counts.max() < model.min_samples_leaf:
            continue
        gradients = np.bincount(values, weights=g, minlength=model.max_bins)
        hessians = np.bincount(values, weights=h, minlength=model.max_bins)
        left_count = np.concatenate(([0], np.cumsum(counts[1:-1])))
        left_g = np.concatenate(([0.], np.cumsum(gradients[1:-1])))
        left_h = np.concatenate(([0.], np.cumsum(hessians[1:-1])))
        C = np.stack((left_count, left_count + counts[0]))
        GL = np.stack((left_g, left_g + gradients[0]))
        HL = np.stack((left_h, left_h + hessians[0]))
        valid = ((C >= model.min_samples_leaf) & (len(rows) - C >= model.min_samples_leaf) &
                 (HL > 1e-9) & (H - HL > 1e-9))
        gains = np.full(C.shape, -np.inf)
        gains[valid] = (GL[valid] ** 2 / (HL[valid] + model.l2) +
                        (G - GL[valid]) ** 2 / (H - HL[valid] + model.l2) - parent_score)
        route, cut = np.unravel_index(np.argmax(gains), gains.shape)
        if gains[route, cut] > best_gain:
            best_gain = gains[route, cut]
            best = (int(feature), int(cut), bool(route))
    if best is None:
        return leaf
    feature, cut, missing_left = best
    values = X[rows, feature]
    goes_left = np.where(values == 0, missing_left, values <= cut)
    return (feature, cut, missing_left,
            reference_grow(model, X, gradient, hessian, rows[goes_left], features, depth + 1),
            reference_grow(model, X, gradient, hessian, rows[~goes_left], features, depth + 1))


class LevelWiseTreeTests(unittest.TestCase):
    def test_level_wise_trees_match_depth_first_reference(self):
        rng = np.random.default_rng(3)
        for bins, depth, leaf, l2 in ((8, 1, 1, 0.), (16, 3, 5, 1.), (32, 4, 20, 10.), (6, 5, 2, .5)):
            with self.subTest(bins=bins, depth=depth):
                X = np.asfortranarray(rng.integers(0, bins, (600, 7), dtype=np.uint8))
                X[:, 2] = np.where(rng.random(600) < .5, 0, X[:, 2])  # Mostly missing.
                X[:, 5] = 1  # Constant: never split.
                gradient, hessian = rng.normal(size=600), rng.uniform(.05, .25, 600)
                rows = np.sort(rng.choice(600, 420, replace=False))
                features = np.array([0, 2, 3, 5, 6])
                model = HistogramGradientBoosting(max_depth=depth, max_bins=bins,
                                                  min_samples_leaf=leaf, l2=l2)
                self.assertEqual(model._grow(X, gradient, hessian, rows, features),
                                 reference_grow(model, X, gradient, hessian, rows, features))


class CodeIndicatorTests(unittest.TestCase):
    def test_dont_know_refused_and_blank_answers_stay_distinct(self):
        X = np.array([[1], [2], [7], [9], [np.nan], [7], [1], [9]], dtype=float)
        settings = {"missing_codes": {"0": [7, 9]}}
        plain = HistogramBinner(settings, max_bins=8).fit(X)
        coded = HistogramBinner(settings, max_bins=8, code_indicator_min=2).fit(X)
        self.assertEqual(coded.code_columns_, [(0, 7.0), (0, 9.0)])
        bins = coded.transform(X)
        np.testing.assert_array_equal(bins[:, 0], plain.transform(X)[:, 0])
        np.testing.assert_array_equal(bins[:, 1], [1, 1, 2, 1, 0, 2, 1, 1])  # "don't know"
        np.testing.assert_array_equal(bins[:, 2], [1, 1, 1, 2, 0, 1, 1, 2])  # "refused"
        rare = HistogramBinner(settings, max_bins=8, code_indicator_min=3).fit(X)
        self.assertEqual(rare.code_columns_, [])
        with self.assertRaises(ValueError):
            HistogramBinner(settings, code_indicator_min=-1)


class SeedEnsembleTests(unittest.TestCase):
    def test_seed_ensemble_averages_members_and_parallel_fit_repeats(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] * X[:, 1] + rng.normal(size=300) > 0).astype(float)
        config = {"model": "histogram_boosted_blend", "seeds": [0, 5, 9],
                  "params": {"n_estimators": 8, "max_depth": 2, "max_bins": 8,
                             "min_samples_leaf": 5}, "threshold": .5}
        serial = BoostedPipeline(config).fit(y, X)
        parallel = BoostedPipeline(config).fit(y, X, workers=3)
        np.testing.assert_array_equal(serial.predict_proba(X), parallel.predict_proba(X))
        members = [BoostedPipeline(dict(config, seeds=[s])).fit(y, X).predict_proba(X)
                   for s in config["seeds"]]
        np.testing.assert_allclose(serial.predict_proba(X), np.mean(members, axis=0))
        for seeds in ([], [1, 1], [True], "0"):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                BoostedPipeline(dict(config, seeds=seeds))


class SearchDriverTests(unittest.TestCase):
    def test_names_grids_selection_and_frontier(self):
        name, spec = sb.parse_tree("trees_lr0.05_d5_n900_l2-30_pp-raw.codes")
        self.assertEqual(name, "trees_d5_lr0.05_n900_l2-30_pp-raw.codes")
        self.assertEqual((spec["d"], spec["leaf"], spec["l2"]), (5, 150, 30.0))
        self.assertEqual(sb.parse_tree("trees_d4_s0")[0], "trees")
        self.assertEqual(sb.checkpoints(900), [300, 600, 900])
        for bad in ("forest_d3", "trees_x1", "trees_pp-other"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sb.parse_tree(bad)
        rng = np.random.default_rng(5)
        y = (rng.random(400) < .2).astype(float)
        scores = np.round(rng.random(400) * .4, 3)  # Ties at grid values.
        cm = sb.confusion_grid(y, scores)
        for t, (tp, fp, fn, tn) in zip(sb.GRID, cm):
            predicted = scores >= t
            self.assertEqual((tp, fp, fn, tn), (np.sum(predicted & (y == 1)), np.sum(predicted & (y == 0)),
                                                np.sum(~predicted & (y == 1)), np.sum(~predicted & (y == 0))))
        grids = [(300, .5, None, cm)]
        free = sb.select(grids)
        self.assertEqual(free["f1"], sb.f1s(cm).max())
        floor = int((cm[:, 0] + cm[:, 3]).max())
        constrained = sb.select(grids, floor)
        i = list(sb.GRID).index(constrained["threshold"])
        self.assertGreaterEqual(cm[i, 0] + cm[i, 3], floor)
        self.assertIsNone(sb.select(grids, len(y) + 1))
        rows = [{"f1": .4, "accuracy": .9}, {"f1": .5, "accuracy": .8}, {"f1": .45, "accuracy": .85},
                {"f1": .44, "accuracy": .84}, {"f1": .5, "accuracy": .7}]
        self.assertEqual(sb.pareto(rows), [rows[1], rows[2], rows[0]])

    def test_fold_rows_are_disjoint_and_cover_the_data(self):
        y = np.tile([0., 0., 0., 1.], 50)
        for fold in range(5):
            train, val, test = sb.fold_rows(y, fold)
            self.assertEqual(len(np.unique(np.r_[train, val, test])), len(y))
            self.assertEqual(len(train) + len(val) + len(test), len(y))
            np.testing.assert_array_equal(test, sb.stratified_k_fold(y, 5, sb.OUTER_SEED)[fold])

    def test_train_search_confirm_final_from_the_command_line(self):
        rng = np.random.default_rng(6)
        n = 1500
        X = rng.normal(size=(n, 3))
        X[rng.random((n, 3)) < .1] = np.nan
        X = np.c_[X, rng.choice([1, 2, 7, 9], n)]
        y = np.where(np.nan_to_num(X[:, 0]) + (X[:, 3] == 7) + rng.normal(size=n) > 1.2, 1, -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            header = "Id,a,b,c,d"
            np.savetxt(path / "x_train.csv", np.c_[np.arange(n), X], delimiter=",", header=header,
                       comments="", fmt="%g")
            np.savetxt(path / "y_train.csv", np.c_[np.arange(n), y], delimiter=",",
                       header="Id,Prediction", comments="", fmt="%g")
            np.savetxt(path / "x_test.csv", np.c_[[9, 4, 7], X[:3]], delimiter=",", header=header,
                       comments="", fmt="%g")
            settings = {"missing_codes": {"3": [7, 9]}, "categorical_columns": [3]}
            (path / "pre.json").write_text(json.dumps(settings))
            (path / "logistic.json").write_text(json.dumps({
                "model": "interaction_logistic_regression", "threshold": .2,
                "params": {"gamma": .3, "lambda_": .001, "max_iters": 30, "n_top": 3},
                "preprocessing": dict(settings, add_missing_indicators=True)}))
            common = [sys.executable, str(ROOT / "search_boosting.py"), "--data-dir", str(path),
                      "--preprocessing", str(path / "pre.json"), "--logistic-config",
                      str(path / "logistic.json"), "--cache-dir", str(path / "cache")]
            models = "logistic,trees,trees_leaf20,trees_leaf20_n600_lr0.05,trees_leaf20_s1,trees_leaf20_pp-codes"

            def run(*arguments):
                result = subprocess.run(common + list(arguments), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout

            run("train", "--folds", "0,1,2,3,4", "--models", models, "--workers", "2")
            self.assertIn("cached", run("train", "--folds", "0", "--models", models))
            run("search", "--models", "trees,trees_leaf20_n600_lr0.05,trees_leaf20_pp-codes",
                "--output", str(path / "search.json"))
            search = json.loads((path / "search.json").read_text())
            self.assertEqual(search["fold"], 0)
            self.assertNotIn("test", json.dumps(search["candidates"]))
            self.assertTrue(search["pareto_frontier"])
            run("confirm", "--candidates", "trees_leaf20_n600_lr0.05@600,trees_leaf20+trees_leaf20_s1@300",
                "--output", str(path / "confirm.json"))
            confirm = json.loads((path / "confirm.json").read_text())
            self.assertEqual(len(confirm["per_fold"]), 5)
            self.assertIn("trees_leaf20+trees_leaf20_s1@300:both", confirm["summary"])
            run("final", "--candidate", "trees_leaf20+trees_leaf20_s1@300", "--rule", "f1",
                "--workers", "2", "--config-output", str(path / "final.json"),
                "--output", str(path / "submission.csv"), "--metadata-output", str(path / "meta.json"))
            config = json.loads((path / "final.json").read_text())
            self.assertEqual(config["seeds"], [0, 1])
            self.assertEqual(config["params"]["n_estimators"], 300)
            lines = (path / "submission.csv").read_text().splitlines()
            self.assertEqual([line.split(",")[0] for line in lines[1:]], ["9", "4", "7"])
            self.assertTrue(all(line.split(",")[1] in ("-1", "1") for line in lines[1:]))
            meta = json.loads((path / "meta.json").read_text())
            self.assertEqual(meta["rows"], 3)


if __name__ == "__main__":
    unittest.main()
