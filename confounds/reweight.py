import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted)
from sklearn.utils.class_weight import compute_sample_weight, \
    compute_class_weight
from confounds.base import BaseDeconfound
from confounds.utils import _get_variable_type


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

    def _fit(self, X, y):
        """Actual fit method"""

        X = check_array(X, ensure_2d=True)
        y = check_array(y, ensure_2d=False)

        # turning it into 2D, in case if its just a column
        if X.ndim == 1:
            X = X[:, np.newaxis]

        try:
            check_consistent_length(X, y)
        except:
            raise ValueError('y and X '
                             'must have the same number of rows/samplets!')

        # Just testing for binary case now
        model = LogisticRegression()
        clf = model.fit(X, y)
        pred_proba = clf.predict_proba(X)
        sample_proba = pred_proba[:, 1]

        # n = np.shape(confounds_)[0]
        # _, weights = np.unique(confounds_, return_counts=True)
        norm_weights = 1./sample_proba
        norm_weights = norm_weights / np.sum(norm_weights)
        self.weights = norm_weights
        return norm_weights

    def transform(self, X=None, y=None):
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

        return self._transform()

    def _transform(self):
        """Actual deconfounding of the test features"""

        # check_is_fitted(self, ('model_', 'n_features_'))
        # test_features = check_array(test_features, accept_sparse=True)
        #
        # if test_features.shape[1] != self.n_features_:
        #     raise ValueError('number of features must be {}. Given {}'
        #                      ''.format(self.n_features_,
        #                                test_features.shape[1]))
        #
        # if test_confounds is None:  # during estimator checks
        #     return test_features  # do nothing
        #
        # test_confounds = check_array(test_confounds, ensure_2d=False)
        # if test_confounds.ndim == 1:
        #     test_confounds = test_confounds[:, np.newaxis]
        # check_consistent_length(test_features, test_confounds)

        weights = self.get_weights()
        return weights

    def get_weights(self):
        """Returns the weights of the reweighting model"""
        # if np.shape(test_features)[0] > 10**4:
        #     warnings.warn('Warning: The number of test samples is very large. '
        #           'This may take a long time to compute.')
        return self.weights
