import numpy as np
from .utils import get_wave_regime


def ch(dimensionless_depth, dimensionless_height, n, **kwargs):
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        _ch = 1
    elif wave_regime == "deep":
        _ch = np.exp(n * dimensionless_height)
    else:
        _ch = np.cosh(n * (dimensionless_depth + dimensionless_height)) / np.cosh(
            n * dimensionless_depth
        )
    return _ch


def sh(dimensionless_depth, dimensionless_height, n, **kwargs):
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        _sh = dimensionless_depth + dimensionless_height
    elif wave_regime == "deep":
        _sh = np.exp(n * dimensionless_height)
    else:
        _sh = np.sinh(n * (dimensionless_depth + dimensionless_height)) / np.cosh(
            n * dimensionless_depth
        )
    return _sh
