from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory.stokes_theory._third_order_statistics import (
    surface_elevation_skewness,
    _reference_surface_skewness_calculation,
)
import numpy as np
from linearwavetheory.stokes_theory._timeseries import surface_time_series
from tests._utils import spectrum2D
from linearwavetheory.settings import _GRAV


def stokes_sum_ampitude(k, d):
    w = np.tanh(k * d)
    return k * (3 - w**2) / 4 / w**3


def stokes_setdown(wavenumber, depth):
    return -wavenumber / np.sinh(2 * wavenumber * depth) / 2


def stokes_wave(amplitude, wavenumber, depth, phase, include_set_down=True):
    if include_set_down:
        z0 = amplitude**2 * stokes_setdown(wavenumber, depth)
    else:
        z0 = 0.0
    z1 = amplitude
    z2 = amplitude**2 * stokes_sum_ampitude(wavenumber, depth)

    return z0 + z1 * np.cos(phase) + z2 * np.cos(2 * phase), z0, z1, z2


def stokes_skewness(amplitude, wavenumber, depth, include_set_down=False):
    phase = np.linspace(0, 2 * np.pi, 100000)
    surface_elevation, z0, z1, z2 = stokes_wave(amplitude, wavenumber, depth, phase)
    # We only consider the skewness calculated up to O(4). I.e. we ignore contributions due to
    # z2*z2*z0 etc. Note that there are 3 terms in the skewness calculation, that 0.5 is the mean of cos^2(x)
    # and 0.25 is the mean of cos^2(x) cos(2x), and that formally <AAAA> = 2 <A^2> for a Gaussian variable.
    if include_set_down:
        return 3 * z1 * z1 * z0 + z1 * z1 * z2 * 6 / 4
    else:
        return z1 * z1 * z2 * 6 / 4


def test_skewness_spectrum():
    dir = np.linspace(0, 360, 36, endpoint=False)
    freq = np.linspace(0.01, 0.5, 51, endpoint=True)
    kd = 0.5
    depth = 10
    k = kd / depth
    omega = np.sqrt(_GRAV * k * np.tanh(k * depth))
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
        skewness / np.sqrt(_var**3), 0.987365, rtol=1e-5, atol=1e-5
    ), f"Skewness is {skewness/ np.sqrt(_var**3)} and target {0.987365}"

    # Test it works for infinite depth
    depth = np.inf
    skewness = surface_elevation_skewness(
        freq,
        dir,
        spec,
        depth=depth,
    )


def test_implementation_consistency():
    dir = np.linspace(0, 360, 36, endpoint=False)
    freq = np.linspace(0.01, 0.5, 51, endpoint=True)
    kds = np.linspace(0.1, 2, 21, endpoint=True)

    spectra = np.zeros((len(kds), len(freq), len(dir)))

    depth = 10
    i = 0
    for kd in kds:

        k = kd / depth
        omega = np.sqrt(_GRAV * k * np.tanh(k * depth))
        period = 2 * np.pi / omega

        waveheight = 3

        spectra[i, :, :] = spectrum2D(
            waveheight=waveheight,
            meandir=0,
            tm01=period,
            spread=20,
            dir=dir,
            frequencies=freq,
        )
        i += 1

    skewness_fast_implementation = surface_elevation_skewness(freq, dir, spectra, depth)
    skewness_slow_implementation = _reference_surface_skewness_calculation(
        freq, dir, spectra, depth
    )

    assert np.allclose(skewness_fast_implementation, skewness_slow_implementation)


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
        depth = _GRAV * kd * np.tanh(kd) / omega**2
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
        assert np.isclose(_stokes_skewness, skewness, rtol=1e-2, atol=1e-2)


def _skewness_bichromatic_time_series_colinear(
    amplitudes, frequencies, directions, depth, cycle_length, include_mean_setdown=False
):

    # Calculate the skewness of the bichromatic wave. We need to create a timeseries of exactly one cycle.
    time = np.linspace(0, 1 * cycle_length, 100000, endpoint=True)

    # Get the linear and nonlinear surface elevation time series
    surface_linear = surface_time_series(
        amplitudes,
        frequencies,
        directions,
        depth,
        time,
        include_mean_setdown=include_mean_setdown,
        nonlinear=False,
    )

    surface_nonlinear = surface_time_series(
        amplitudes,
        frequencies,
        directions,
        depth,
        time,
        include_mean_setdown=False,
        nonlinear=True,
        linear=False,
    )

    self_interactions_surface_nonlinear = surface_time_series(
        amplitudes,
        frequencies,
        directions,
        depth,
        time,
        include_mean_setdown=False,
        nonlinear=True,
        linear=False,
        cross_interactions=False,
    )

    # Since we compare to weakly nonlinear theory, we only return the cross-correlation between the nonlinar signal
    # and the linear signal squared. (other contributions are either zero- or assumed to be of higher order and
    # neglected). Note that for a Gaussian variable the self interactions contribute twice statistically to the Skewness
    # since <aaaa> = 2 <aa>^2. Instead of drawing multiple realizations (the proper solution) I hack that here by simply
    # adding the self-interactions twice.
    return np.mean(
        3
        * (surface_nonlinear + self_interactions_surface_nonlinear)
        * surface_linear**2
    )


def test_skewness_bichromatic():
    cycle_length = 100
    df = 2 / cycle_length
    ndir = 36
    period = 1 / (5 * df)
    dir = np.linspace(0, 360, ndir, endpoint=False)
    freq = np.linspace(1 / period, 1 / period + df, 2, endpoint=True)
    ddir = 360 / ndir
    steepness = 0.05
    omega = 2 * np.pi / period
    for kd in np.linspace(0.1, 10, 100):

        depth = _GRAV * kd * np.tanh(kd) / omega**2
        wavenumber = kd / depth
        amplitude = steepness / wavenumber

        skewness_bichromatic_wave = _skewness_bichromatic_time_series_colinear(
            [amplitude / 2, amplitude / 2],
            freq,
            [0, 0],
            depth,
            cycle_length=cycle_length,
        )

        _var = amplitude**2 / 2
        spec = np.zeros((len(freq), len(dir)))
        spec[0, 0] = _var / ddir / df
        spec[1, 0] = _var / ddir / df

        skewness = surface_elevation_skewness(
            freq,
            dir,
            spec,
            depth,
        )
        assert np.isclose(
            skewness, skewness_bichromatic_wave, rtol=1e-2
        ), f"Skewness is {skewness} and {skewness_bichromatic_wave} at kd={kd}"
