# -*- coding: utf-8 -*-

"""

"""

# сделать поддержку разных типов (особенно разряженные матрицы) - как в sklearn.linear_model.LinearRegression

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import NotFittedError


class SubsetRegression(BaseEstimator, RegressorMixin):

    def __init__(self, intercept=True):
        self.intercept = intercept

    def _add_intercept(self, x):
        if self.intercept:
            return np.hstack((np.ones(x.shape[0]).reshape((-1, 1)), x))
        return x

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        x = self._add_intercept(x)
        self.coef_, residuals, rank, s = np.linalg.lstsq(x, y)

    def predict(self, x):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Subset regression is not fitted yet")

        x = np.array(x)
        x = self._add_intercept(x)
        return np.dot(x, self.coef_)


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
