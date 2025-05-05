"""
Nonlinear dispersion relation
==========================

This module provides functions to calculate the nonlinear dispersion relation for fourth order Stokes waves in both
Eulerian and Lagrangian reference frames.

The Eulerian solution for the nonlinear dispersion relation is based on the work of Zhao and Liu (2022),
while the Lagrangian solution is new.
"""


import numpy as np
from .settings import _DEFAULT_ORDER
from .lagrangian_displacement_amplitudes import ul20, ul40
from linearwavetheory.dispersion import intrinsic_dispersion_relation
from .settings import ReferenceFrame
from .utils import get_wave_regime


def nonlinear_dispersion_relation(
    steepness,
    wavenumber,
    depth,
    z=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs,
):
    """
    This function calculates the nonlinear dispersion relation for a fourth order Stokes wave.
    :param steepness: wave steepness (dimensionless).
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param z: height in the water column (m), default is 0 (surface)
    :param kwargs:
    :return: nonlinear dispersion relation (rad/s)
    """
    linear_angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )

    return (
        dimensionless_nonlinear_dispersion_relation(
            steepness, wavenumber * depth, wavenumber * z, reference_frame, **kwargs
        )
        * linear_angular_frequency
    )


def dimensionless_nonlinear_dispersion_relation(
    steepness,
    relative_depth,
    relative_z=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs,
):
    """
    This function calculates the dimensionless nonlinear dispersion relation for a fourth order Stokes wave. To get
    the actual angular frequency, multiply the result by the linear angular frequency (calculated using
    the linear dispersion relation).

    :param steepness: wave steepness (dimensionless).
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth.
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column.
    :param reference_frame: either 'eulerian' or 'lagrangian', default is 'eulerian'.
    :param kwargs:
    :return: dimensionless angular frequency
    """

    order = kwargs.get("order", _DEFAULT_ORDER)

    if reference_frame == ReferenceFrame.eulerian:
        _w2 = 0 if order < 2 else w2(relative_depth, **kwargs)
        _w4 = 0 if order < 4 else w4(relative_depth, **kwargs)
    elif reference_frame == ReferenceFrame.lagrangian:
        _w2 = 0 if order < 2 else w2l(relative_depth, relative_z, **kwargs)
        _w4 = 0 if order < 4 else w4l(relative_depth, relative_z, **kwargs)
    else:
        raise ValueError(
            f"Invalid reference frame {reference_frame}. Choose 'eulerian' or 'lagrangian'."
        )

    return 1 + steepness**2 * _w2 + steepness**4 * _w4


def w2(relative_depth, **kwargs):
    """
    Second-order nonlinear dispersion relation coeficient
    :param steepness: wave steepness (dimensionless).
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth.
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column.
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        _disp = 9 / relative_depth**4

    elif wave_regime == "deep":
        _disp = 0.5

    else:
        # See 3.7 in Zhao and Liu (2022). # Be aware that omega_2 in Zhao and Liu (2022) is off by a factor 1/4,
        # otherwise it does not match the deep water limit, nor other published results. Note also that I have
        # rewritten the relation into a form without "alpha" - as it is more commonly presented in the literature.
        _disp = (9 - 10 * mu**2 + 9 * mu**4) / 16 / mu**4
    return _disp


def w4(relative_depth, **kwargs):
    """
    Fourth-order nonlinear dispersion relation coeficient
    :param steepness: wave steepness (dimensionless).
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth.
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column.
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    if wave_regime == "shallow":
        # Leading order behaviour in shallow water limit mu -> 0
        _disp = 81 / 1024 * mu**10

    elif wave_regime == "deep":
        # Leading order behaviour in deep water limit mu -> 1
        _disp = 0.625

    else:
        _disp = (
            81
            - 603 * mu**2
            + 3618 * mu**4
            - 3662 * mu**6
            + 1869 * mu**8
            - 663 * mu**10
        ) / (1024 * mu**10)
    return _disp


def w2l(relative_depth, relative_z, **kwargs):
    """
    Second-order nonlinear dispersion relation coeficient in the Lagrangian frame of reference. Note that this is
    merely a shift of the Eulerian solution by the dimensionless stokes drift velocity (ul20).

    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth.
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column.
    :param kwargs:
    :return:
    """
    return w2(relative_depth, **kwargs) - ul20(relative_depth, relative_z, **kwargs)


def w4l(relative_depth, relative_z, **kwargs):
    """
    Fourth-order nonlinear dispersion relation coeficient in the Lagrangian frame of reference. Note that this is
    merely a shift of the Eulerian solution by the dimensionless stokes drift velocity (ul40).

    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth.
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column.
    :param kwargs:
    :return:
    """
    return w4(relative_depth, **kwargs) - ul40(relative_depth, relative_z, **kwargs)
