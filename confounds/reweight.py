import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted)

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

    def _fit(self, in_features, confounds=None,
             features_type=None,
             confounds_type=None):
        """Actual fit method"""

        in_features = check_array(in_features)
        confounds = check_array(confounds, ensure_2d=False)

        # turning it into 2D, in case if its just a column
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]

        try:
            check_consistent_length(in_features, confounds)
        except:
            raise ValueError('X (features) and y (confounds) '
                             'must have the same number of rows/samplets!')

        self.n_features_ = in_features.shape[1]
        dep_type = _get_variable_type(in_features)
        indep_type = _get_variable_type(confounds)
        kde = sm.nonparametric.KDEMultivariateConditional(endog=in_features,
                                                          exog=confounds,
                                                          dep_type=dep_type,
                                                          indep_type=indep_type,
                                                          bw='normal_reference')

        self.model_ = kde
        return self

    def transform(self, X, y=None):
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

        return self._transform(X, y)

    def _transform(self, test_features, test_confounds):
        """Actual deconfounding of the test features"""

        check_is_fitted(self, ('model_', 'n_features_'))
        test_features = check_array(test_features, accept_sparse=True)

        if test_features.shape[1] != self.n_features_:
            raise ValueError('number of features must be {}. Given {}'
                             ''.format(self.n_features_,
                                       test_features.shape[1]))

        if test_confounds is None:  # during estimator checks
            return test_features  # do nothing

        test_confounds = check_array(test_confounds, ensure_2d=False)
        if test_confounds.ndim == 1:
            test_confounds = test_confounds[:, np.newaxis]
        check_consistent_length(test_features, test_confounds)

        weights = self.get_weights(test_features, test_confounds)
        rescaled_weights = weights / np.sum(weights)
        return rescaled_weights

    def get_weights(self, test_features, test_confounds):
        """Returns the weights of the reweighting model"""
        if np.shape(test_features)[0] > 10**4:
            warnings.warn('Warning: The number of test samples is very large. '
                  'This may take a long time to compute.')
        return self.model_.pdf(endog_predict=test_features, exog_predict=test_confounds)