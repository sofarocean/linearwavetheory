from linearwavetheory.stokes_theory._timeseries import surface_time_series
from tests.stokes_theory.test_third_order_statistics import stokes_wave
from linearwavetheory import inverse_intrinsic_dispersion_relation
import numpy as np


def test_surface_time_series():
    amplitude = 1
    phase = np.linspace(0, 2 * np.pi, 1000)
    frequency = 0.1
    depth = 10
    wavenumber = inverse_intrinsic_dispersion_relation(2 * np.pi * frequency, depth)

    z = stokes_wave(amplitude, wavenumber, depth, phase, include_set_down=True)

    amplitudes = np.array([amplitude]) / 2
    frequencies = np.array([frequency])
    directions = np.array([0])
    time = phase / (2 * np.pi * frequency)

    surface_elevation = surface_time_series(
        amplitudes,
        frequencies,
        directions,
        depth,
        time,
        include_mean_setdown=True,
        zero_mean=False,
    )

    relative_diff = np.abs(z[0] - surface_elevation) / amplitude
    assert np.all(
        relative_diff < 1e-3
    ), f"Relative difference is {np.max(relative_diff)}"
