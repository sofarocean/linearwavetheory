from linearwavetheory.stokes_theory._perturbation_theory_coeficients import (
    _second_order_surface_elevation,
    _second_order_lagrangian_surface_elevation,
)
import numpy as np


def stokes_sum_ampitude(k, d):
    w = np.tanh(k * d)
    return k * (3 - w**2) / 4 / w**3


def stokes_setdown(wavenumber, depth):
    return -wavenumber / np.sinh(2 * wavenumber * depth) / 2


def stokes_wave(amplitude, wavenumber, depth, phase):
    z0 = amplitude**2 * stokes_setdown(wavenumber, depth)
    z1 = amplitude
    z2 = amplitude**2 * stokes_sum_ampitude(wavenumber, depth)

    return z0 + z1 * np.cos(phase) + z2 * np.cos(2 * phase), z0, z1, z2


def stokes_skewness(amplitude, wavenumber, depth, include_set_down=False):
    phase = np.linspace(0, 2 * np.pi, 100000)
    surface_elevation, z0, z1, z2 = stokes_wave(amplitude, wavenumber, depth, phase)
    # We only consider the skewness calculated up to O(4). I.e. we ignore contributions due to
    # z2*z2*z0 etc. Note that there are 3 terms in the skewness calculation, and that 0.5 is the mean of cos^2(x)
    # and 0.25 is the mean of cos^2(x) cos(2x)
    if include_set_down:
        return 1.5 * z1 * z1 * z0 + z1 * z1 * z2 * 3 / 4
    else:
        return z1 * z1 * z2 * 3 / 4


def test_coef_second_order_surface_elevation():
    depth = 1
    grav = 9.81
    for kd in np.linspace(0.1, 10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))
        c12 = _second_order_surface_elevation(w, k, k, 0.0, w, k, k, 0.0, depth, grav)
        assert np.isclose(c12, 2 * stokes_sum_ampitude(k, depth))

        _setdown = _second_order_surface_elevation(
            w, k, k, 0.0, -w, k, -k, 0.0, depth, grav
        )
        _stokes_setdown = stokes_setdown(k, depth)
        assert np.isclose(_setdown, 2 * _stokes_setdown, rtol=1e-5, atol=1e-5)

    # Test it works for infinite depth
    depth = np.inf
    k = 1
    w = np.sqrt(grav * k)
    c12 = _second_order_surface_elevation(w, k, k, 0.0, w, k, k, 0.0, depth, grav)
    assert np.isfinite(c12)

    c12 = _second_order_surface_elevation(w, k, k, 0.0, -w, k, -k, 0.0, depth, grav)
    assert np.isfinite(c12)


def test_coef_second_order_lagrangian_surface_elevation():

    # Deep water sum interactions must vanish
    depth = 1
    grav = 9.81
    for kd in range(10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))

        k2 = 1.5 * k
        w2 = np.sqrt(grav * k2 * np.tanh(k2 * depth))
        c12 = _second_order_lagrangian_surface_elevation(
            w, k, k, 0.0, w2, k2, k2, 0.0, depth, grav
        )
        assert np.isclose(c12, 0, rtol=1e-5, atol=1e-5)
