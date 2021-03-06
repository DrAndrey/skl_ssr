# -*- coding: utf-8 -*-

"""

"""

import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils.validation import NotFittedError

from skl_ssr.subset_regression import TTLinearRegression, SubsetRegression

RANDOM_STATE = 1


class TestTTLinearRegression(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-10

    def create_regression_dataset(self, bias=0.0):
        return make_regression(n_samples=1000, n_features=10, n_informative=5, bias=bias, coef=True,
                               random_state=RANDOM_STATE, n_targets=2)

    def test_linear_regression_with_intercept(self):
        bias = 1.0
        regression = TTLinearRegression()
        x, y, real_coef = self.create_regression_dataset(bias=bias)

        regression.fit(x, y)

        bias_error = (regression.coef_[0] - bias) / bias
        self.assertTrue((bias_error < self.tol).all())

        coef_error = abs(regression.coef_[1:] - real_coef) / (real_coef + 1)
        self.assertTrue((coef_error < self.tol).all())

    def test_linear_regression_without_intercept(self):
        regression = TTLinearRegression(intercept=False)
        x, y, real_coef = self.create_regression_dataset()

        regression.fit(x, y)

        self.assertTrue((abs(regression.coef_[0]) < self.tol).all())

    def test_predict(self):
        regression = TTLinearRegression()
        x, y, real_coef = self.create_regression_dataset()

        regression.fit(x, y)

        pred_error = abs(regression.predict(x) - y) / (y + 1)
        self.assertTrue((pred_error < self.tol).all())

    def test_predict_before_fit(self):
        regression = TTLinearRegression()
        x, y, real_coef = self.create_regression_dataset()

        with self.assertRaises(NotFittedError):
            regression.predict(x)

    def test_t_test(self):
        bias = 1.0
        regression = TTLinearRegression()
        x, y, real_coef = self.create_regression_dataset(bias)

        regression.fit(x, y)
        mask = regression.make_t_test(x, y)
        self.assertFalse(mask.all())


class TestSubsetRegression(unittest.TestCase):

    def create_regression_dataset(self, bias=0.0):
        return make_regression(n_samples=1000, n_features=10, n_informative=5, bias=bias, coef=True,
                               random_state=RANDOM_STATE, n_targets=2)

    def test_subset_regression(self):
        bias = 1.0
        regression = SubsetRegression()
        x, y, real_coef = self.create_regression_dataset(bias=bias)

        regression.fit(x, y)
        self.assertEqual(np.unique(real_coef.nonzero()[0]).shape[0], len(regression.best_subset))

if __name__ == '__main__':
    unittest.main()

