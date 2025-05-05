import numpy as np
from numba import jit, prange

from linearwavetheory import PhysicsOptions, inverse_intrinsic_dispersion_relation
from linearwavetheory._numba_settings import numba_default
from linearwavetheory._utils import _direction_bin
from linearwavetheory.settings import StokesTheoryOptions, _parse_options
from linearwavetheory.stokes_theory._third_order_coeficients import (
    third_order_dispersion_correction,
    third_order_dispersion_correction_sigma_coordinates,
)
from linearwavetheory._array_shape_preprocessing import atleast_1d
from typing import Union


@jit(**numba_default)
def _pointwise_estimate_nonlinear_dispersion(
    angular_frequency,
    angles_degrees,
    variance_density,
    depth,
    nonlinear_options: StokesTheoryOptions,
    physics_options: PhysicsOptions,
) -> np.ndarray:
    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.

    # Component 1
    # ----------------
    wavenumbers = inverse_intrinsic_dispersion_relation(angular_frequency, depth)

    # precalculate the trigonometric functions
    angles_rad_coordinates = np.deg2rad(angles_degrees)
    _cos = np.cos(angles_rad_coordinates)
    _sin = np.sin(angles_rad_coordinates)

    # Set up the integration
    # ----------------
    # Get angle stepsizes for integration. We use the midpoint rule for the angles.
    delta_angle = _direction_bin(angles_degrees, wrap=360)

    # Integrate
    # ----------------
    nfreq = len(angular_frequency)
    ndir = len(angles_degrees)

    if nonlinear_options.reference_frame == "eulerian":
        lagrangian = False
    elif nonlinear_options.reference_frame == "lagrangian":
        lagrangian = True
    else:
        raise Exception("Unknown kind must be one of 'eulerian', 'lagrangian'")

    out = np.zeros((nfreq, ndir), dtype=variance_density.dtype)
    for jfreq in range(nfreq):
        for idir in range(ndir):
            w1 = angular_frequency[jfreq]
            k1 = wavenumbers[jfreq]
            kx1 = k1 * _cos[idir]
            ky1 = k1 * _sin[idir]

            sum = 0.0
            for ifreq in range(nfreq):
                # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints
                # and df inbetween.
                iu = min(ifreq + 1, nfreq - 1)
                il = max(ifreq - 1, 0)
                delta_freq = (angular_frequency[iu] - angular_frequency[il]) / 2.0

                # Compoment 2 wavenumber, components and angular frequency. The sign index is used to determine if we
                # are calculating the sum (sign_index=1) or difference (sign_index=-1) interactions.
                k2 = wavenumbers[ifreq]
                w2 = angular_frequency[ifreq]
                kx2 = k2 * _cos
                ky2 = k2 * _sin

                # Calculate the interaction coefficients. Use interaction coefficient appropriate for Eulerian or
                # Lagrangian observations.

                if not nonlinear_options.use_s_theory:
                    interaction_coef = third_order_dispersion_correction(
                        w1,
                        k1,
                        kx1,
                        ky1,
                        w2,
                        k2,
                        kx2,
                        ky2,
                        depth,
                        physics_options.grav,
                        nonlinear_options.wave_driven_flow_included_in_mean_flow,
                        nonlinear_options.wave_driven_setup_included_in_mean_depth,
                        lagrangian,
                    )
                else:
                    interaction_coef = (
                        third_order_dispersion_correction_sigma_coordinates(
                            w1,
                            k1,
                            kx1,
                            ky1,
                            w2,
                            k2,
                            kx2,
                            ky2,
                            depth,
                            physics_options.grav,
                            nonlinear_options.wave_driven_flow_included_in_mean_flow,
                            nonlinear_options.wave_driven_setup_included_in_mean_depth,
                            lagrangian,
                        )
                    )

                if w1 == w2:
                    #
                    fac = 0.5
                else:
                    fac = 1.0
                sum += (
                    np.sum(variance_density[ifreq, :] * interaction_coef * delta_angle)
                    * delta_freq
                    * fac
                )

            out[jfreq, idir] = sum

    return out


@jit(**numba_default)
def nonlinear_dispersion(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
    progress_bar=None,
):
    """
    Calculate the nonlinear dispersion correction estimate for a given variance density.

    :param intrinsic_frequency_hz: intrinsic frequency of the variance density spectrum (Hz)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2/Hz/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: Nonlinear frequency correction
    """
    dims = variance_density.shape

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    nspec = int(np.prod(np.array(dims[:-2])))
    ndir = len(angle_degrees)
    nfreq = len(intrinsic_angular_frequency)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    output = np.zeros(
        (nspec, nfreq, ndir),
    )

    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

        output[i, :, :] = _pointwise_estimate_nonlinear_dispersion(
            intrinsic_angular_frequency,
            angle_degrees,
            variance_density[i, :, :],
            depth[i],
            nonlinear_options,
            physics_options,
        )

    return output.reshape(dims)
