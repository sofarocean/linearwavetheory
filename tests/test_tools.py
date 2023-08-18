from src.linearwavetheory._array_shape_preprocessing import (
    atleast_1d,
    atleast_2d,
    _to_2d_array,
)
from src.linearwavetheory._numba_settings import numba_default
from numba import jit
import numpy as np


@jit(**numba_default)
def _numba_test_atleast1d(x):
    """
    Note we need to wrap to ensure it gets compiled to numba
    """
    return atleast_1d(x)


@jit(**numba_default)
def _numba_test_atleast2d(x):
    """
    Note we need to wrap to ensure it gets compiled to numba
    """
    return atleast_2d(x)


def test_atleast_1d():
    """
    Test if the numba implementation behaves the same as the numpy implementation.
    """
    assert _numba_test_atleast1d(1) == np.atleast_1d(1)
    assert np.all(_numba_test_atleast1d(np.array([1, 2])) == np.atleast_1d([1, 2]))
    assert np.all(_numba_test_atleast1d(np.array((1, 2))) == np.atleast_1d([1, 2]))


def test_atleast_2d():
    """
    Test if the numba implementation behaves the same as the numpy implementation.
    """
    assert _numba_test_atleast2d(1) == np.atleast_2d(1)
    assert np.all(_numba_test_atleast2d(np.array([1, 2])) == np.atleast_2d([1, 2]))
    assert np.all(_numba_test_atleast2d(np.array((1, 2))) == np.atleast_2d([1, 2]))
    assert np.all(_numba_test_atleast2d((1.0, 2.0)) == np.atleast_2d([1, 2]))


def test_to_2d_array():
    """
    Test if promotion to 2d vector arrays follows the rules.
    """
    try:
        # Scalars are not allowed
        _to_2d_array(1)
        assert False
    except ValueError:
        pass

    try:
        # 1D arrays must be of length 2
        _to_2d_array((1, 2, 3))
        assert False
    except ValueError:
        pass

    assert _to_2d_array((1.0, 0.0)).shape == (1, 2)
    assert _to_2d_array((1.0, 0.0), 10).shape == (10, 2)
    assert _to_2d_array((1.0, -1.0), 10)[8, 1] == -1.0

    y = np.zeros((11, 2))
    try:
        # Mismatch in 2D array
        _to_2d_array(y, 10)
        assert False
    except ValueError:
        pass

    assert np.all(_to_2d_array(y, 11) == y)
