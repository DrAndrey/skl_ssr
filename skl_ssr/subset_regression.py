# -*- coding: utf-8 -*-

"""

"""

import copy

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import NotFittedError
from sklearn.metrics import mean_squared_error


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


class SubsetRegression(BaseEstimator, RegressorMixin):

    def __init__(self, model_type=LinearRegression, model_params=None, scorer=None, tol=None, verbose=False):
        self.scorer = scorer
        self.tol = tol
        self.verbose = verbose

        if model_params is None:
            model_params = {}
        self.model = model_type(**model_params)

    def _calc_lr_error(self, x, y):
        self.model.fit(x, y)
        y_pred = self.model.predict(x)
        if self.scorer:
            return self.scorer(y, y_pred)
        return mean_squared_error(y, y_pred)

    def _forward_step(self, x, y, best_subset, considered_vars):
        best_var = None
        best_err = None
        if best_subset:
            best_err = self._calc_lr_error(x[:, list(best_subset)], y)

        for var in considered_vars:
            best_subset.add(var)
            new_err = self._calc_lr_error(x[:, list(best_subset)], y)
            if best_err is None or new_err < best_err:
                best_err = new_err
                best_var = var
            best_subset.remove(var)

        is_converged = True
        if best_var is not None:
            best_subset.add(best_var)
            is_converged = False
        return is_converged

    def _backward_step(self, x, y, best_subset):
        worst_var = None
        best_err = self._calc_lr_error(x[:, list(best_subset)], y)

        need_backward_step = True
        while len(best_subset) > 1 and need_backward_step:
            for var in copy.copy(best_subset):
                best_subset.remove(var)
                new_err = self._calc_lr_error(x[:, list(best_subset)], y)
                if new_err <= best_err:
                    best_err = new_err
                    worst_var = var
                best_subset.add(var)
            if worst_var is not None:
                best_subset.remove(worst_var)
                worst_var = None
            else:
                need_backward_step = False

    def fit(self, x, y):
        is_converged = False
        best_subset = set()
        vars_to_model = set(range(x.shape[1]))
        while not is_converged:
            considered_vars = vars_to_model - best_subset
            is_converged = self._forward_step(x, y, best_subset, considered_vars)
            if not is_converged:
                self._backward_step(x, y, best_subset)

            err = self._calc_lr_error(x[:, list(best_subset)], y)
            if self.tol and err < self.tol:
                is_converged = True
            if self.verbose:
                msg = "Subset step. Number of features - {0}. Error value - {1}".format(len(best_subset), err)
                print(msg)
        self.best_subset = best_subset
        self.model.fit(x[:, list(best_subset)], y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y, sample_weight=None):
        return self._calc_lr_error(x, y)

if __name__ == '__main__':
    pass