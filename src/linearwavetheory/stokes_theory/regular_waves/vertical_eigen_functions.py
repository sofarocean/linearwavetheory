import numpy as np
from .utils import get_wave_regime


def ch(dimensionless_depth, dimensionless_height, n, **kwargs):
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        _ch = 1
    elif wave_regime == "deep":
        _ch = np.exp(n * dimensionless_height)
    else:
        if np.isscalar(dimensionless_depth):
            if n * dimensionless_depth > 10:
                _ch = np.exp(n * dimensionless_height)
            else:
                _ch = np.cosh(
                    n * (dimensionless_depth + dimensionless_height)
                ) / np.cosh(n * dimensionless_depth)
        else:
            dimensionless_depth = dimensionless_depth + np.zeros_like(
                dimensionless_height
            )
            dimensionless_height = dimensionless_height + np.zeros_like(
                dimensionless_depth
            )

            mask = np.isinf(dimensionless_depth)
            _ch = np.zeros_like(dimensionless_depth)
            _ch[mask] = np.exp(n * dimensionless_height[mask])
            _ch[~mask] = np.cosh(
                n * (dimensionless_depth[~mask] + dimensionless_height[~mask])
            ) / np.cosh(n * dimensionless_depth[~mask])
    return _ch


def sh(dimensionless_depth, dimensionless_height, n, **kwargs):
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        _sh = dimensionless_depth + dimensionless_height
    elif wave_regime == "deep":
        _sh = np.exp(n * dimensionless_height)
    else:
        if np.isscalar(dimensionless_depth):
            if n * dimensionless_depth > 10:
                _sh = np.exp(n * dimensionless_height)
            else:
                _sh = np.sinh(
                    n * (dimensionless_depth + dimensionless_height)
                ) / np.cosh(n * dimensionless_depth)
        else:
            dimensionless_depth = dimensionless_depth + np.zeros_like(
                dimensionless_height
            )
            dimensionless_height = dimensionless_height + np.zeros_like(
                dimensionless_depth
            )

            mask = np.isinf(dimensionless_depth)
            _sh = np.zeros_like(dimensionless_depth)
            _sh[mask] = np.exp(n * dimensionless_height[mask])
            _sh[~mask] = np.sinh(
                n * (dimensionless_depth[~mask] + dimensionless_height[~mask])
            ) / np.cosh(n * dimensionless_depth[~mask])
    return _sh
