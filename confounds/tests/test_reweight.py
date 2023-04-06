from confounds import Reweight
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('tkagg')


def test_reweight_simple():
    """Test Reweight on a simple example"""
    X = np.random.randn(100, 10)
    C = np.random.randn(100, 5)
    y = np.random.randn(100)
    deconfounder = Reweight()
    deconfounder.fit(X, C)
    sample_weights = deconfounder.transform(X, C)
    assert sample_weights.shape == (100,)
    assert np.all(sample_weights >= 0)
    assert np.all(sample_weights <= 1)
    assert np.allclose(sample_weights.sum(), 1)


def test_reweight_with_model():
    """Test Reweight with a model"""
    X = np.random.randn(100, 10)
    C = np.random.randn(100, 5)
    beta = np.random.randn(10)
    y = X.dot(beta) + np.random.randn(100)

    deconfounder = Reweight()
    deconfounder.fit(X, C)
    sample_weights = deconfounder.transform(X, C)
    weighted_model = LinearRegression()
    weighted_model.fit(X, y, sample_weight=sample_weights)
    weighted_y_pred = weighted_model.predict(X)
    weighted_score = mean_squared_error(y, weighted_y_pred)

    vanilla_model = LinearRegression()
    vanilla_model.fit(X, y, sample_weight=sample_weights)
    vanilla_y_pred = vanilla_model.predict(X)
    vanilla_score = mean_squared_error(y, vanilla_y_pred)
    print(weighted_score, vanilla_score)


def test_reweight_with_pure_causality():
    """
    Test Reweight with pure causality
    C -> X -> y
    """


def test_reweight_with_confounding():
    """
    Test Reweight with confounding
    X <- C -> y
    """


def test_reweight_with_pure_confounding():
    """
    Test Reweight with pure confounding
    X <- C -> y
    """
    num_samples = 100
    num_features = 1
    num_confounds = 2
    num_targets = 1
    skewness = -5
    # C = np.random.randint(low=1, high=3, size=(num_samples, num_confounds))
    C = np.ceil(4*skewnorm.rvs(a=skewness, size=(num_samples, num_confounds))+4)
    # matrix multiply to get correlated X
    X = np.matmul(C, 2*np.ones((num_confounds, num_features))) + 0.1*np.random.randn(num_samples, num_features)
    y = np.matmul(C, np.ones((num_confounds, num_targets)) - 5) + 0.1*np.random.randn(num_samples, num_targets)

    deconfounder = Reweight()
    deconfounder.fit(X, C)
    sample_weights = deconfounder.transform(X, C)
    weighted_model = SVR()
    weighted_model.fit(X, y, sample_weight=sample_weights)
    weighted_y_pred = weighted_model.predict(X)
    weighted_score = mean_squared_error(y, weighted_y_pred)

    vanilla_model = SVR()
    vanilla_model.fit(X, y)
    vanilla_y_pred = vanilla_model.predict(X)
    vanilla_score = mean_squared_error(y, vanilla_y_pred)
    print(weighted_score, vanilla_score)
