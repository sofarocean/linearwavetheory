from linearwavetheory._utils import _direction_bin, _frequency_bin
from linearwavetheory.stokes_theory._second_order_coeficients import (
    _second_order_surface_elevation,
)
from linearwavetheory.settings import PhysicsOptions, _parse_options
from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory._array_shape_preprocessing import atleast_1d
from typing import Union
from numba import jit, prange
from linearwavetheory._numba_settings import numba_default, numba_default_parallel
import numpy as np
from numba_progress import ProgressBar


@jit(**numba_default)
def _skewness_from_spectrum(
    angular_frequency,
    direction,
    variance_density,
    interaction_coefficient_function,
    depth,
    grav,
):
    """
    Calculate the skewness of the wave field from the energy spectrum according to stokes perturbation theory.

    :param angular_frequency: frequency in Hz
    :param direction: direction in degrees
    :param energy: energy spectrum in m^2/Hz/degree
    :return: skewness
    """
    nf = len(angular_frequency)
    nd = len(direction)

    radian_direction = np.deg2rad(direction)
    wavenumber = inverse_intrinsic_dispersion_relation(angular_frequency, depth)

    frequency_bin = _frequency_bin(angular_frequency)
    direction_bin = _direction_bin(direction, wrap=360)

    _cos = np.cos(radian_direction)
    _sin = np.sin(radian_direction)

    _skewness = 0.00
    for _freq1 in range(0, nf):
        w1 = angular_frequency[_freq1]
        k1 = wavenumber[_freq1]
        for _freq2 in range(0, nf):
            w2 = angular_frequency[_freq2]
            k2 = wavenumber[_freq2]

            for _dir1 in range(0, nd):
                kx1 = k1 * _cos[_dir1]
                ky1 = k1 * _sin[_dir1]
                for _dir2 in range(0, nd):
                    kx2 = k2 * np.cos(radian_direction[_dir2])
                    ky2 = k2 * np.sin(radian_direction[_dir2])

                    interaction_coefficient = 0.0
                    for isign1 in [-1, 1]:
                        for isign2 in [-1, 1]:
                            if _freq1 == _freq2:
                                if isign1 * isign2 == -1:
                                    continue

                            interaction_coefficient += interaction_coefficient_function(
                                isign1 * w1,
                                k1,
                                isign1 * kx1,
                                isign1 * ky1,
                                isign2 * w2,
                                k2,
                                isign2 * kx2,
                                isign2 * ky2,
                                depth,
                                grav,
                            )

                    hyper_volume = (
                        frequency_bin[_freq1]
                        * frequency_bin[_freq2]
                        * direction_bin[_dir1]
                        * direction_bin[_dir2]
                    )
                    # Divide by four to switch from a two-sided spectrum to a one-sided  spectrum
                    interacting_wave_energy = (
                        variance_density[_freq1, _dir1]
                        * variance_density[_freq2, _dir2]
                    ) / 4.0

                    _skewness += (
                        interacting_wave_energy * interaction_coefficient * hyper_volume
                    )

    return 6 * _skewness


@jit(**numba_default)
def surface_elevation_skewness_from_spectrum(
    angular_frequency,
    direction,
    variance_density,
    depth,
    grav,
):
    """
    Calculate the skewness of the wave field from the energy spectrum according to stokes perturbation theory.

    :param angular_frequency: frequency in Hz
    :param direction: direction in degrees
    :param energy: energy spectrum in m^2/Hz/degree
    :return: skewness
    """
    nf = len(angular_frequency)
    nd = len(direction)

    radian_direction = np.deg2rad(direction)
    wavenumber = inverse_intrinsic_dispersion_relation(angular_frequency, depth)

    frequency_bin = _frequency_bin(angular_frequency)
    direction_bin = _direction_bin(direction, wrap=360)

    # pre-calculate wavenumber components
    kx = np.empty((nf, nd))
    ky = np.empty((nf, nd))
    for i in range(nf):
        for j in range(nd):
            kx[i, j] = np.cos(radian_direction[j]) * wavenumber[i]
            ky[i, j] = np.sin(radian_direction[j]) * wavenumber[i]

    # Set depth to a large number if it is infinite. There is a chance that the tanh function for tanh( ksum * depth)
    # is undifined below for ksum = 0. Instead of introducing an if statement in the loop, we set the depth to a large
    # number. This is a valid approximation as the tanh function will approach 1 for large values of the argument.
    if np.isinf(depth):
        depth = 1e10

    _skewness = 0.0
    # Sum interactions
    for sign_index in [-1, 1]:
        if sign_index == -1:
            jstart = 1
        else:
            jstart = 0

        for frequency_index_component1 in range(0, nf):
            w1 = angular_frequency[frequency_index_component1]
            k1 = wavenumber[frequency_index_component1]
            if w1 == 0.0:
                # Skip zero frequency as it introduces singularities. We assume a zero mean process.
                continue

            for frequency_index_component2 in range(
                frequency_index_component1 + jstart, nf
            ):
                k2 = wavenumber[frequency_index_component2]
                w2 = sign_index * angular_frequency[frequency_index_component2]
                if w2 == 0.0:
                    # Skip zero frequency as it introduces singularities. We assume a zero mean process.
                    continue

                if (
                    sign_index == -1
                    and frequency_index_component1 == frequency_index_component2
                ):
                    # Skip the zero mean interactions. We assume that the signal we are comparing to has zero-mean and
                    # exclude the mean set-down interactions.
                    continue

                # Because we only sum over the upper triangle of the interaction matrix for efficiency, we need to
                # multiply by two to get the full sum -- except if we are summing over the diagonal.
                fac = 2.0
                if frequency_index_component1 == frequency_index_component2:
                    fac = 1.0

                wsum = w1 + w2
                #
                # Pull these factors of the interaction coeficient outside the direction loops since they are
                # independent of the direction.
                fac1 = -grav / w1 / w2 * fac
                fac2 = (w1 * w2 + w1**2 + w2**2) / (2 * grav) * fac
                fac3 = -grav * (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2) * fac

                for direction_index_component1 in range(0, nd):
                    kx1 = kx[frequency_index_component1, direction_index_component1]
                    ky1 = ky[frequency_index_component1, direction_index_component1]
                    for direction_index_component2 in range(0, nd):
                        kx2 = (
                            sign_index
                            * kx[frequency_index_component2, direction_index_component2]
                        )
                        ky2 = (
                            sign_index
                            * ky[frequency_index_component2, direction_index_component2]
                        )

                        inner_product = kx1 * kx2 + ky1 * ky2
                        ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)

                        w12_squared = grav * ksum * np.tanh(ksum * depth)
                        resonance_factor = wsum / (w12_squared - wsum**2)

                        interaction_coefficient = (
                            fac1 * (wsum * resonance_factor + 0.5) * inner_product
                            + fac2 * (1 + wsum * resonance_factor)
                            + fac3 * resonance_factor
                        )

                        hyper_volume = (
                            frequency_bin[frequency_index_component1]
                            * frequency_bin[frequency_index_component2]
                            * direction_bin[direction_index_component1]
                            * direction_bin[direction_index_component2]
                        )
                        # Divide by four to switch from a one-sided spectrum to a two-sided spectrum
                        # but multiply by two as we have ++ and -- interactions which contribute equally
                        interacting_wave_energy = (
                            variance_density[
                                frequency_index_component1, direction_index_component1
                            ]
                            * variance_density[
                                frequency_index_component2, direction_index_component2
                            ]
                        ) / 2.0

                        _skewness += (
                            interacting_wave_energy
                            * interaction_coefficient
                            * hyper_volume
                        )

    return 6 * _skewness


#
@jit(**numba_default_parallel)
def _skewness_from_spectra(
    angular_frequency,
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
            angular_frequency,
            direction,
            variance_density[i, :, :],
            interaction_coefficient_function,
            depth[i],
            grav,
        )

    return skewness.reshape(dims[:-2])


@jit(**numba_default_parallel)
def _surface_elevation_skewness_from_spectra(
    angular_frequency,
    direction,
    variance_density,
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

        skewness[i] = surface_elevation_skewness_from_spectrum(
            angular_frequency,
            direction,
            variance_density[i, :, :],
            depth[i],
            grav,
        )

    return skewness.reshape(dims[:-2])


@jit(**numba_default)
def _reference_surface_skewness_calculation(
    angular_frequency: np.ndarray,
    direction: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
):
    """
    Calculate the skewness of the wave field for a general interaction coefficient

    :param angular_frequency: frequency in Hz
    :param direction: direction in degrees
    :param variance_density: energy spectrum in m^2/Hz/degree
    :param depth: depth in meters
    :param physics_options: physics options
    :param progress_bar: progress bar (optional). Pass a numba progress bar to show progress.
    :return: skewness as ndarray
    """
    _, physics_options, _ = _parse_options(None, physics_options, None)

    skewness = _skewness_from_spectra(
        angular_frequency,
        direction,
        variance_density,
        _second_order_surface_elevation,
        depth,
        physics_options.grav,
        None,
    )
    return skewness


def surface_elevation_skewness(
    angular_frequency: np.ndarray,
    direction: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
    display_progress_bar=True,
):
    """
    Calculate the skewness of the wave field from the energy spectrum according to stokes perturbation theory. See e.g.
    Herbers, T. H. C., & Janssen, T. T. (2016).

    Ref:
    Herbers, T. H. C., & Janssen, T. T. (2016). Lagrangian surface wave motion and Stokes drift fluctuations.
    Journal of Physical Oceanography, 46(4), 1009-1021.

    :param angular_frequency: frequency in rad/s
    :param direction: direction in degrees
    :param variance_density: energy spectrum in m^2 s/rad/degree
    :param depth: depth in meters
    :param physics_options: physics options
    :param progress_bar: progress bar (optional). Pass a numba progress bar to show progress.
    :return: skewness as ndarray
    """
    _, physics_options, _ = _parse_options(None, physics_options, None)

    dims = variance_density.shape
    ndim = variance_density.ndim

    if ndim < 2:
        raise ValueError("variance_density must have at least two dimensions")

    if not (dims[-1] == len(direction)):
        raise ValueError(
            "variance_density must have the same number of directions as the "
            "length of the direction array. The last dimension of variance_density is assumed to be "
            "the direction dimension."
        )

    if not (dims[-2] == len(angular_frequency)):
        raise ValueError(
            "variance_density must have the same number of frequencies as the "
            "length of the direction array. The second to last dimension of variance_density is assumed to "
            "be the direction dimension."
        )

    disable = not display_progress_bar
    if ndim > 2:
        number_of_spectra = np.prod(dims[:-2])
        if number_of_spectra < 10:
            disable = True
    else:
        number_of_spectra = 1
        disable = True

    with ProgressBar(
        total=number_of_spectra, disable=disable, desc="Calculating skewness"
    ) as progress_bar:
        skewness = _surface_elevation_skewness_from_spectra(
            angular_frequency,
            direction,
            variance_density,
            depth,
            physics_options.grav,
            progress_bar,
        )
    return skewness
