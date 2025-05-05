from ._second_order_coeficients import (
    _second_order_lagrangian_horizontal_velocity,
    _second_order_horizontal_velocity,
)
from linearwavetheory import inverse_intrinsic_dispersion_relation
import numpy as np
from linearwavetheory._numba_settings import numba_default
from numba import jit
from linearwavetheory._array_shape_preprocessing import atleast_1d
from linearwavetheory._utils import _direction_bin
from typing import Union
from ..settings import PhysicsOptions, _parse_options, StokesTheoryOptions


@jit(**numba_default)
def lagrangian_drift_velocity(
    angular_frequency: np.ndarray,
    angles_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
):
    """
    Calculate the Lagrangian drift velocity due to the second order wave-wave interactions.

    :param angular_frequency: intrinsic frequency of the variance density spectrum (rad/s)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2 s/rad/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    dims = variance_density.shape

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    nspec = int(np.prod(np.array(dims[:-2])))
    nfreq = len(angular_frequency)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)
    angles_rad_coordinates = np.deg2rad(angles_degrees)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    vel_x = np.zeros((nspec), dtype=np.float64)
    vel_y = np.zeros((nspec), dtype=np.float64)

    delta_angle = _direction_bin(angles_degrees, wrap=360)

    _cos = np.cos(angles_rad_coordinates)
    _sin = np.sin(angles_rad_coordinates)
    grav = physics_options.grav

    for jx in range(nspec):
        wavenumbers = inverse_intrinsic_dispersion_relation(
            angular_frequency, depth[jx]
        )
        for ifreq in range(nfreq):
            iu = min(ifreq + 1, nfreq - 1)
            il = max(ifreq - 1, 0)
            delta_freq = (angular_frequency[iu] - angular_frequency[il]) / 2.0
            w = angular_frequency[ifreq]
            k = wavenumbers[ifreq]
            kx = k * _cos
            ky = k * _sin
            e = variance_density[jx, ifreq, :]

            lagrangian_interaction_coef_x = (
                _second_order_lagrangian_horizontal_velocity(
                    w, k, kx, ky, -w, k, -kx, -ky, 1, depth[jx], grav
                )
            )
            lagrangian_interaction_coef_y = (
                _second_order_lagrangian_horizontal_velocity(
                    w, k, kx, ky, -w, k, -kx, -ky, -1, depth[jx], grav
                )
            )

            if not nonlinear_options.include_eulerian_contribution_in_drift:
                lagrangian_interaction_coef_x -= _second_order_horizontal_velocity(
                    w, k, kx, ky, -w, k, -kx, -ky, 1, depth[jx], grav
                )
                lagrangian_interaction_coef_y -= _second_order_horizontal_velocity(
                    w, k, kx, ky, -w, k, -kx, -ky, -1, depth[jx], grav
                )

            vel_x[jx] += (
                np.sum(lagrangian_interaction_coef_x * e * delta_angle) * delta_freq
            )
            vel_y[jx] += (
                np.sum(lagrangian_interaction_coef_y * e * delta_angle) * delta_freq
            )

    return vel_x, vel_y
