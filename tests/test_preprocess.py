"""Synthetic schema, missingness, encoding, and training-statistic checks."""

import unittest

import numpy as np

from preprocess import Preprocessor, binary_labels


class PreprocessorTests(unittest.TestCase):
    def test_training_medians_and_scaling_are_reused(self):
        X = np.array([[0., np.nan, 5.], [2., np.nan, 5.], [np.nan, np.nan, 5.]])
        original = X.copy()
        processor = Preprocessor().fit(X)
        np.testing.assert_allclose(processor.transform(X)[:, 1], [-np.sqrt(1.5), np.sqrt(1.5), 0])
        before = processor.means_.copy()
        transformed = processor.transform(np.array([[101., 10., 9.], [np.nan, 1., 2.]]))
        self.assertEqual(transformed.shape, (2, 2))
        self.assertGreater(transformed[0, 1], 100)
        self.assertEqual(transformed[1, 1], 0)
        np.testing.assert_array_equal(processor.means_, before)
        np.testing.assert_array_equal(X, original)

    def test_missing_codes_are_column_specific(self):
        X = np.array([[7., 7.], [1., 1.], [3., 3.]])
        processor = Preprocessor(missing_codes={"0": [7]}, standardize=False)
        result = processor.fit_transform(X)
        np.testing.assert_allclose(result[0], [1., 2., 7.])

    def test_zero_codes_and_dropped_columns(self):
        X = np.array([[88., 5., np.nan], [3., 6., 1.], [99., 7., np.nan]])
        processor = Preprocessor(missing_codes={"0": [99]}, zero_codes={"0": [88]},
                                 drop_columns=[2], add_missing_indicators=True,
                                 standardize=False)
        result = processor.fit_transform(X)
        np.testing.assert_allclose(result, [[1, 0, 5, 0], [1, 3, 6, 0], [1, 1.5, 7, 1]])

    def test_category_reference_missing_and_unseen_levels(self):
        processor = Preprocessor(categorical_columns=[0], standardize=False)
        result = processor.fit_transform(np.array([[1.], [2.], [np.nan], [1.]]))
        np.testing.assert_array_equal(result, [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(processor.transform([[3.], [np.nan]]), [[1, 0, 0], [1, 0, 1]])

    def test_numeric_squares_and_missing_indicators(self):
        processor = Preprocessor(degree=2, add_missing_indicators=True, standardize=False)
        result = processor.fit_transform(np.array([[1.], [3.], [np.nan]]))
        np.testing.assert_array_equal(result, [[1, 1, 1, 0], [1, 3, 9, 0], [1, 2, 4, 1]])

    def test_all_missing_or_constant_features_leave_bias(self):
        X = np.array([[np.nan, 4.], [np.nan, 4.]])
        np.testing.assert_array_equal(Preprocessor().fit_transform(X), np.ones((2, 1)))
        np.testing.assert_array_equal(
            Preprocessor(categorical_columns=[0, 1]).fit_transform(X), np.ones((2, 1)))
        with self.assertRaises(ValueError):
            Preprocessor(add_bias=False).fit(X)

    def test_invalid_schema_and_unfitted_transform(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            Preprocessor().transform([[1.]])
        for settings in ({"degree": 3}, {"degree": True}, {"missing_codes": {"-1": [9]}},
                         {"categorical_columns": [0, 0]}, {"standardize": "yes"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Preprocessor(**settings)
        processor = Preprocessor().fit([[1.], [2.]])
        for values in ([[np.inf]], [[1., 2.]]):
            with self.assertRaises(ValueError):
                processor.transform(values)

    def test_label_validation(self):
        np.testing.assert_array_equal(binary_labels([-1, 1, -1]), [0., 1., 0.])
        np.testing.assert_array_equal(binary_labels([0, 1]), [0., 1.])
        for labels in ([], [0, 2], [np.nan], [[1], [0]]):
            with self.assertRaises(ValueError):
                binary_labels(labels)


if __name__ == "__main__":
    unittest.main()
