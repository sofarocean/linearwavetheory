from linearwavetheory.stokes_theory._perturbation_theory_coeficients import (
    _second_order_surface_elevation,
)
from linearwavetheory.settings import PhysicsOptions, _parse_options
from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory._array_shape_preprocessing import atleast_1d
from typing import Union
from numba import jit, prange
from linearwavetheory._numba_settings import numba_default, numba_default_parallel
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
    angular_frequency = 2 * np.pi * frequency

    frequency_bin = _frequency_bin(frequency)
    direction_bin = _direction_bin(direction, wrap=360)

    dirgrid1 = np.empty((nd, nd, 2, 2))
    dirgrid2 = np.empty((nd, nd, 2, 2))
    signgrid1 = np.empty((nd, nd, 2, 2))
    signgrid2 = np.empty((nd, nd, 2, 2))
    wgrid1 = np.empty((nd, nd, 2, 2))
    wgrid2 = np.empty((nd, nd, 2, 2))

    for i in range(nd):
        for j in range(nd):
            dirgrid1[i, j, :, :] = radian_direction[i]
            dirgrid2[i, j, :, :] = radian_direction[j]
            signgrid1[i, j, 0, 0] = -1
            signgrid1[i, j, 0, 1] = 1
            signgrid1[i, j, 1, 0] = -1
            signgrid1[i, j, 1, 1] = 1
            signgrid2[i, j, 0, 0] = -1
            signgrid2[i, j, 0, 1] = -1
            signgrid2[i, j, 1, 0] = 1
            signgrid2[i, j, 1, 1] = 1

    # dirgrid1,dirgrid2,signgrid1,signgrid2 = np.meshgrid(radian_direction, radian_direction)
    _interaction_coefficient = np.zeros((nd, nd, 2, 2), dtype=np.float64)

    _skewness = 0.0
    for _freq1 in range(0, nf):
        wgrid1[:, :, :, :] = signgrid1 * angular_frequency[_freq1]
        for _freq2 in range(0, nf):
            wgrid2[:, :, :, :] = signgrid2 * angular_frequency[_freq2]
            _interaction_coefficient[:, :, :, :] = interaction_coefficient_function(
                wgrid1,
                wavenumber[_freq1],
                dirgrid1,
                signgrid1,
                wgrid2,
                wavenumber[_freq2],
                dirgrid2,
                signgrid2,
                depth,
                grav,
            )
            if _freq1 == _freq2:
                for _dir in range(0, nd):
                    _interaction_coefficient[_dir, _dir, 0, 1] = 0
                    _interaction_coefficient[_dir, _dir, 1, 0] = 0

            for _dir1 in range(0, nd):
                for _dir2 in range(0, nd):
                    interaction_coefficient = (
                        _interaction_coefficient[_dir1, _dir2, 0, 0]
                        + _interaction_coefficient[_dir1, _dir2, 0, 1]
                        + _interaction_coefficient[_dir1, _dir2, 1, 0]
                        + _interaction_coefficient[_dir1, _dir2, 1, 1]
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


#
@jit(**numba_default_parallel)
def _skewness_from_spectra(
    frequency,
    direction,
    variance_density,
    interaction_coefficient_function,
    depth,
    grav,
    progress_bar=None,
):
    dims = variance_density.shape

    nspec = int(np.prod(np.array(dims[:-2])))
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    skewness = np.zeros(nspec)
    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

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
    progress_bar=None,
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
        progress_bar,
    )
