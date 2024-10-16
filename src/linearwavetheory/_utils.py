import numpy as np
from numba import jit
from linearwavetheory._numba_settings import numba_default


@jit(**numba_default)
def _direction_bin(direction, wrap=2 * np.pi, kind="midpoint"):
    _tmp = np.zeros(len(direction) + 2)
    _tmp[0] = direction[-1]
    _tmp[1:-1] = direction
    _tmp[-1] = direction[0]
    _angle_diff = (np.diff(_tmp) + wrap / 2) % wrap - wrap / 2

    if kind == "midpoint":
        bins = (_angle_diff[:-1] + _angle_diff[1:]) / 2
    elif kind == "forward":
        bins = _angle_diff[1:]
    elif kind == "backward":
        bins = _angle_diff[:-1]
    else:
        raise Exception('kind must be either "midpoint", "forward" or "backward"')
    return bins


@jit(**numba_default)
def _frequency_bin(frequency):
    _tmp = np.zeros(len(frequency) + 2)
    _tmp[0] = frequency[0] - (frequency[1] - frequency[0])
    if _tmp[0] < 0:
        _tmp[0] = 0

    _tmp[1:-1] = frequency
    _tmp[-1] = frequency[-1] + (frequency[-1] - frequency[-2])
    _freq_diff = np.diff(_tmp)
    return (_freq_diff[:-1] + _freq_diff[1:]) / 2
