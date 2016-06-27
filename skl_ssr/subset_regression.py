# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import NotFittedError


class TTLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, intercept=True):
        self.intercept = intercept

    def _add_intercept(self, x):
        if self.intercept:
            return np.hstack((np.ones(x.shape[0]).reshape((-1, 1)), x))
        return x

    def _calc_dependent_variable_variance(self, x, y):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Subset regression is not fitted yet")
        pred_y = self.predict(x)
        dvv = ((y - pred_y) ** 2).sum(axis=0) / (x.shape[0] - x.shape[1])
        return dvv.reshape(1, -1)

    def _calc_coef_z_score(self, x, y, alpha):
        dvv = self._calc_dependent_variable_variance(x, y)

        x = self._add_intercept(x)
        xx_1 = np.linalg.pinv(np.dot(x.T, x))
        coef_variances = np.dot(xx_1.reshape(-1, 1), dvv).reshape((x.shape[1], x.shape[1], dvv.shape[1]))
        cv_diag = coef_variances.diagonal(0, 0, 1) ** 0.5

        z_score = self.coef_ / cv_diag.T
        lb = self.coef_ + stats.t.ppf(alpha, x.shape[0]-x.shape[1]) * cv_diag.T
        rb = self.coef_ - stats.t.ppf(alpha, x.shape[0]-x.shape[1]) * cv_diag.T
        conf_interval = np.array([lb, rb])
        return z_score, conf_interval

    def make_t_test(self, x, y, alpha=0.05):
        z_score, conf_interval = self._calc_coef_z_score(x, y, alpha)
        t = stats.t.cdf(z_score, x.shape[0]-x.shape[1])
        return (t < alpha / 2) | (t > 1 - alpha / 2)

    def fit(self, x, y):
        x = self._add_intercept(x)
        self.coef_, residuals, rank, s = np.linalg.lstsq(x, y)

    def predict(self, x):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Subset regression is not fitted yet")

        x = np.array(x)
        x = self._add_intercept(x)
        return np.dot(x, self.coef_)


if __name__ == '__main__':
    pass