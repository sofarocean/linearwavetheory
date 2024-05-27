from linearwavetheory.stokes_theory._perturbation_theory_coeficients import (
    _second_order_surface_elevation,
    _second_order_lagrangian_surface_elevation,
)
from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory.stokes_theory._third_order_statistics import (
    surface_elevation_skewness,
)
import numpy as np

from tests._utils import spectrum2D


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
        c12 = _second_order_surface_elevation(k, 0, 1.0, k, 0, 1, depth, grav)
        assert np.isclose(c12, 2 * stokes_sum_ampitude(k, depth))

        _setdown = _second_order_surface_elevation(k, 0, 1.0, k, 0, -1, depth, grav)
        _stokes_setdown = stokes_setdown(k, depth)
        assert np.isclose(_setdown, 2 * _stokes_setdown, rtol=1e-5, atol=1e-5)


def test_coef_second_order_lagrangian_surface_elevation():

    # Deep water sum interactions must vanish
    depth = 1
    grav = 9.81
    for kd in range(10, 100):
        k = kd / depth
        c12 = _second_order_lagrangian_surface_elevation(
            k, 0, 1.0, 1.5 * k, 0, 1.0, depth, grav
        )
        assert np.isclose(c12, 0, rtol=1e-5, atol=1e-5)


def test_skewness_spectrum():
    dir = np.linspace(0, 360, 36, endpoint=False)
    freq = np.linspace(0.01, 0.5, 51, endpoint=True)
    kd = 0.5
    depth = 10
    k = kd / depth
    omega = np.sqrt(9.81 * k * np.tanh(k * depth))
    period = 2 * np.pi / omega

    waveheight = 3
    _var = waveheight**2 / 16

    spec = spectrum2D(
        waveheight=waveheight,
        meandir=0,
        tm01=period,
        spread=20,
        dir=dir,
        frequencies=freq,
    )

    skewness = surface_elevation_skewness(
        freq,
        dir,
        spec,
        depth=depth,
    )
    assert np.isclose(
        skewness / np.sqrt(_var**3), 0.48861658091281845, rtol=1e-5, atol=1e-5
    )


def test_skewness_stokeswave():

    df = 0.01
    ndir = 36
    period = 10
    dir = np.linspace(0, 360, ndir, endpoint=False)
    freq = np.linspace(1 / period, 1 / period + df, 2, endpoint=True)
    ddir = 360 / ndir

    omega = 2 * np.pi / period

    for kd in np.linspace(0.1, 10, 100):

        steepness = 0.1
        depth = 9.81 * kd * np.tanh(kd) / omega**2
        wavenumber = kd / depth
        amplitude = steepness / wavenumber

        # To note- we recalculate the wavenumber here as the skewness routine uses an approximate
        # inversion as well.
        wavenumber = inverse_intrinsic_dispersion_relation(2 * np.pi * freq[0], depth)

        _stokes_skewness = stokes_skewness(amplitude, wavenumber[0], depth)
        _var = amplitude**2 / 2
        spec = np.zeros((len(freq), len(dir)))
        spec[0, 0] = _var / ddir / df

        skewness = surface_elevation_skewness(
            freq,
            dir,
            spec,
            depth,
        )
        assert np.isclose(_stokes_skewness, skewness)
