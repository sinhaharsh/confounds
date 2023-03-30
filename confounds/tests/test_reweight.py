from confounds import Reweight
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
