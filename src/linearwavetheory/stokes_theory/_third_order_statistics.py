from linearwavetheory.stokes_theory._perturbation_theory_coeficients import (
    _second_order_surface_elevation,
)
from linearwavetheory.settings import PhysicsOptions, _parse_options
from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory._array_shape_preprocessing import atleast_1d
from typing import Union
from numba import jit
from linearwavetheory._numba_settings import numba_default
import numpy as np


@jit(**numba_default)
def _direction_bin(direction, wrap=2 * np.pi):
    _tmp = np.zeros(len(direction) + 2)
    _tmp[0] = direction[-1]
    _tmp[1:-1] = direction
    _tmp[-1] = direction[0]
    _angle_diff = (np.diff(_tmp) + wrap / 2) % wrap - wrap / 2
    return (_angle_diff[:-1] + _angle_diff[1:]) / 2


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


@jit(**numba_default)
def _skewness_from_spectrum(
    frequency,
    direction,
    variance_density,
    interaction_coefficient_function,
    depth,
    grav,
):
    """
    Calculate the skewness of the wave field from the energy spectrum according to stokes perturbation theory.

    :param frequency: frequency in Hz
    :param direction: direction in degrees
    :param energy: energy spectrum in m^2/Hz/degree
    :return: skewness
    """
    nf = len(frequency)
    nd = len(direction)

    radian_direction = np.deg2rad(direction)
    wavenumber = inverse_intrinsic_dispersion_relation(2 * np.pi * frequency, depth)

    frequency_bin = _frequency_bin(frequency)
    direction_bin = _direction_bin(direction, wrap=360)

    _skewness = 0.0
    for _freq1 in range(0, nf):
        for _freq2 in range(0, nf):
            for _dir1 in range(0, nd):
                for _dir2 in range(0, nd):
                    interaction_coefficient = 0.0
                    for sign_index1 in [-1, 1]:
                        for sign_index2 in [-1, 1]:
                            if (
                                _freq1 == _freq2
                                and _dir1 == _dir2
                                and sign_index1 == -sign_index2
                            ):
                                # Disregard the zero interaction coefficient- we assume the signal is zero-mean.
                                # This effectively neglects e.g. setdown which - because it is a constant contribution
                                # requires a different treatment.
                                continue

                            interaction_coefficient += interaction_coefficient_function(
                                wavenumber[_freq1],
                                radian_direction[_dir1],
                                sign_index1,
                                wavenumber[_freq2],
                                radian_direction[_dir2],
                                sign_index2,
                                depth,
                                grav,
                            )

                    hyper_volume = (
                        frequency_bin[_freq1]
                        * frequency_bin[_freq2]
                        * direction_bin[_dir1]
                        * direction_bin[_dir2]
                    )
                    # Divide by four to switch from a one-sided spectrum to a two-sided spectrum
                    interacting_wave_energy = (
                        variance_density[_freq1, _dir1]
                        * variance_density[_freq2, _dir2]
                    ) / 4.0

                    _skewness += (
                        interacting_wave_energy * interaction_coefficient * hyper_volume
                    )

    return 3 * _skewness


@jit(**numba_default)
def _skewness_from_spectra(
    frequency,
    direction,
    variance_density,
    interaction_coefficient_function,
    depth,
    grav,
):
    dims = variance_density.shape

    nspec = int(np.prod(np.array(dims[:-2])))
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    skewness = np.zeros(nspec)
    for i in range(nspec):
        skewness[i] = _skewness_from_spectrum(
            frequency,
            direction,
            variance_density[i, :, :],
            interaction_coefficient_function,
            depth[i],
            grav,
        )

    return skewness.reshape(dims[:-2])


@jit(**numba_default)
def surface_elevation_skewness(
    frequency,
    direction,
    variance_density,
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
):
    """
    Calculate the skewness of the wave field from the energy spectrum according to stokes perturbation theory.

    :param frequency: frequency in Hz
    :param direction: direction in degrees
    :param energy: energy spectrum in m^2/Hz/degree
    :return: skewness
    """
    _, physics_options = _parse_options(None, physics_options)

    return _skewness_from_spectra(
        frequency,
        direction,
        variance_density,
        _second_order_surface_elevation,
        depth,
        physics_options.grav,
    )
