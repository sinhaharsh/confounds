import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted)
from sklearn.utils.class_weight import compute_sample_weight, \
    compute_class_weight
from confounds.base import BaseDeconfound
from confounds.utils import _get_variable_type
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import statsmodels as sm


class Reweight(BaseDeconfound):
    """
    Deconfounding estimator class that reweights the input features based on
    the correlation with the conf
    """

    def __init__(self, model='linear'):
        """Constructor"""

        super().__init__(name='Reweight')

        self.model = model
        self.weights = None

    def fit(self,
            X,  # variable names chosen to correspond to sklearn when possible
            y=None,  # y is the confound variables here, not the target!
            ):
        """
        Fits the residualizing model (estimates the contributions of confounding
        variables (y) to the given [training] feature set X.  Variable names X,
        y had to be used to pass sklearn conventions. y here refers to the
        confound variables, and NOT the target. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

        return self._fit(X, y)  # which itself must return self

    def _fit(self, X, C):
        """Actual fit method"""

        X = check_array(X, ensure_2d=False)
        C = check_array(C, ensure_2d=False)

        # turning it into 2D, in case if its just a column
        if X.ndim == 1:
            X = X[:, np.newaxis]

        try:
            check_consistent_length(X, C)
        except:
            raise ValueError('C and X '
                             'must have the same number of rows/samplets!')


        self.adjusted_model = LinearRegression()
        self.adjusted_model.fit(C, X)

        self.dummy_model = LinearRegression()
        ones = np.ones_like(C)
        self.dummy_model.fit(ones, X)
        return self

    def transform(self, X=None, C=None):
        """
        Transforms the given feature set by residualizing the [test] features
        by subtracting the contributions of their confounding variables.

        Variable names X, y had to be used to pass scikit-learn conventions. y here
        refers to the confound variables for the [test] to be transformed,
        and NOT their target values. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

        return self._transform(X, C)

    def _transform(self, X, C):
        """Actual deconfounding of the test features"""

        adjusted_pred = self.adjusted_model.predict(C)
        adjusted_resid_std = np.std(adjusted_pred.ravel() - X.ravel())
        adjusted_density = stats.norm(loc=adjusted_pred,
                                      scale=adjusted_resid_std)
        adjusted_densities = adjusted_density.pdf(X)

        ones = np.ones_like(C)
        dummy_pred = self.dummy_model.predict(ones)
        dummy_residuals = np.std(dummy_pred.ravel() - X.ravel())
        dummy_density = stats.norm(loc=dummy_pred, scale=dummy_residuals)
        dummy_densities = dummy_density.pdf(X)
        self.weights = np.divide(adjusted_densities.ravel(),
                                 dummy_densities.ravel())

        return self.weights

    def get_weights(self):
        """Returns the weights of the reweighting model"""
        # if np.shape(test_features)[0] > 10**4:
        #     warnings.warn('Warning: The number of test samples is very large. '
        #           'This may take a long time to compute.')
        return self.weights
