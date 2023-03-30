from confounds.utils import _get_variable_type, _is_categorical
import numpy as np


def test_get_variable_type():
    X = np.random.randn(100, 10)
    assert _get_variable_type(X) == 'c' * 10
    X = np.random.randn(100, 10).astype(int)
    assert _get_variable_type(X) == 'c' * 10


def test_is_categorical():
    X = np.random.randn(100, 10)
    assert not _is_categorical(X)
    colors = np.array(['red', 'green', 'blue', 'yellow'])
    assert _is_categorical(colors)
    digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert not _is_categorical(digits)
    large_list_integers = np.arange(1000)
    assert not _is_categorical(large_list_integers)
    large_list_floats = np.arange(1000).astype(float)
    assert not _is_categorical(large_list_floats)
    large_list_strings = np.arange(1000).astype(str)
    assert _is_categorical(large_list_strings)


