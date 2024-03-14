from confounds import Reweight
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import matplotlib

from confounds.tests.simulated_data import generate_data

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

    X, y, C = generate_data()
    Y = y.reshape(-1, 1)


    deconfounder = Reweight()
    deconfounder.fit(X, C)
    sample_weights = deconfounder.transform(X, C)

    weighted_model = SVC()
    weighted_model.fit(X, Y, sample_weight=sample_weights)
    weighted_y_pred = weighted_model.predict(X)
    weighted_score = mean_squared_error(Y, weighted_y_pred)

    vanilla_model = SVC()
    vanilla_model.fit(X, Y)
    vanilla_y_pred = vanilla_model.predict(X)
    vanilla_score = mean_squared_error(Y, vanilla_y_pred)
    print(weighted_score, vanilla_score)
