"""Analytic checks of the six required algorithms; no course data needed."""

import unittest

import numpy as np

from implementations import (
    compute_logistic_gradient, compute_logistic_loss, compute_mse_gradient,
    compute_mse_loss, least_squares, logistic_regression, mean_squared_error_gd,
    mean_squared_error_sgd, reg_logistic_regression, ridge_regression,
)
from models import MODEL_NAMES, score_model, train_model
from tune_models import make_candidates


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
        self.y = np.array([0., 0., 1., 1.])

    def test_linear_optimum_and_unpenalized_ridge_loss(self):
        X = np.array([[1., -1.], [1., 1.]])
        y = np.array([0., 2.])
        np.testing.assert_allclose(least_squares(y, X)[0], [1., 1.])
        w, loss = ridge_regression(y, X, 0.5)
        np.testing.assert_allclose(w, [0.5, 0.5])
        self.assertAlmostEqual(loss, 0.25)

    def test_singular_least_squares_still_fits_predictions(self):
        X = np.array([[1., 1.], [2., 2.], [3., 3.]])
        w, loss = least_squares(np.array([2., 4., 6.]), X)
        np.testing.assert_allclose(X @ w, [2., 4., 6.])
        self.assertLess(loss, 1e-20)

    def test_gd_one_step_and_input_unchanged(self):
        initial = np.array([0.2, -0.1])
        saved = initial.copy()
        w, loss = mean_squared_error_gd(self.y, self.X, initial, 1, 0.1)
        residual = self.X @ saved - self.y
        np.testing.assert_allclose(w, saved - 0.1 * self.X.T @ residual / 4)
        np.testing.assert_array_equal(initial, saved)
        self.assertAlmostEqual(loss, np.mean((self.y - self.X @ w) ** 2) / 2)

    def test_sgd_uses_exactly_one_sample(self):
        # Identical rows make the sampled index irrelevant and the update analytic.
        w, loss = mean_squared_error_sgd(np.ones(5), np.ones((5, 1)), np.zeros(1), 1, 0.2)
        np.testing.assert_allclose(w, [0.2])
        self.assertAlmostEqual(loss, 0.32)

    def test_logistic_updates_and_unpenalized_loss(self):
        initial = np.array([0.2, -0.1])
        p = 1 / (1 + np.exp(-(self.X @ initial)))
        gradient = self.X.T @ (p - self.y) / len(self.y)
        for penalty in (0., 0.3):
            if penalty:
                w, loss = reg_logistic_regression(self.y, self.X, penalty, initial, 1, 0.1)
            else:
                w, loss = logistic_regression(self.y, self.X, initial, 1, 0.1)
            np.testing.assert_allclose(w, initial - 0.1 * (gradient + 2 * penalty * initial))
            probabilities = 1 / (1 + np.exp(-(self.X @ w)))
            expected = -np.mean(self.y * np.log(probabilities) +
                                (1 - self.y) * np.log1p(-probabilities))
            self.assertAlmostEqual(loss, expected)

    def test_gradients_match_finite_differences(self):
        w = np.array([0.2, -0.1])
        for loss_fn, gradient_fn in ((compute_mse_loss, compute_mse_gradient),
                                     (compute_logistic_loss, compute_logistic_gradient)):
            numerical = []
            for delta in np.eye(2) * 1e-6:
                numerical.append((loss_fn(self.y, self.X, w + delta) -
                                  loss_fn(self.y, self.X, w - delta)) / 2e-6)
            np.testing.assert_allclose(gradient_fn(self.y, self.X, w), numerical, atol=1e-8)

    def test_extreme_logistic_loss_is_not_clipped(self):
        loss = compute_logistic_loss(np.array([1., 0.]), np.array([[-1000.], [1000.]]), np.ones(1))
        self.assertAlmostEqual(loss, 1000.)

    def test_all_adapters_and_probability_scoring(self):
        candidates = make_candidates()
        self.assertEqual({c["model"] for c in candidates}, set(MODEL_NAMES))
        for name in MODEL_NAMES:
            config = next(c for c in candidates if c["model"] == name)
            if "max_iters" in config["params"]:
                config["params"]["max_iters"] = 10
            w, loss = train_model(config, self.y, self.X)
            self.assertEqual(w.shape, (2,))
            self.assertTrue(np.isfinite(loss))
            scores = score_model(config, self.X, w)
            self.assertEqual(scores.shape, self.y.shape)
            if "logistic" in name:
                self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_seeded_sgd_repeats_without_mutating_global_rng(self):
        config = {"model": "mean_squared_error_sgd", "seed": 4,
                  "params": {"max_iters": 100, "gamma": 0.01}}
        state = np.random.get_state()
        first = train_model(config, self.y, self.X)[0]
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_array_equal(first, train_model(config, self.y, self.X)[0])


if __name__ == "__main__":
    unittest.main()
