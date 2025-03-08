import numpy as np

from linearwavetheory.settings import _parse_options
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_surface_amplitude_first_harmonic,
    dimensionless_surface_amplitude_third_harmonic,
    dimensionless_surface_amplitude_second_harmonic,
    dimensionless_surface_amplitude_fourth_harmonic,
    dimensionless_surface_amplitude_fifth_harmonic,
)

from linearwavetheory.dispersion import intrinsic_dispersion_relation


def waveheight(steepness, wavenumber, depth, **kwargs):
    """
    This function calculates the wave height of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    a1 = dimensionless_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a3 = dimensionless_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    return 2 * (a1 + a3) / wavenumber


def dimensionless_surface_variance(steepness, relative_depth, **kwargs):
    """
    This function calculates the wave height of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    dimensionless_surface_amplitude_third_harmonic()
    return 1


def dimensionless_crest_height(steepness, wavenumber, depth, **kwargs):
    a1 = dimensionless_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a2 = dimensionless_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a3 = dimensionless_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a4 = dimensionless_surface_amplitude_fourth_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a5 = dimensionless_surface_amplitude_fifth_harmonic(
        steepness, wavenumber, depth, **kwargs
    )

    return a1 + a2 + a3 + a4 + a5


def dimensionless_trough_height(steepness, wavenumber, depth, **kwargs):
    a1 = dimensionless_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a2 = dimensionless_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a3 = dimensionless_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a4 = dimensionless_surface_amplitude_fourth_harmonic(
        steepness, wavenumber, depth, **kwargs
    )
    a5 = dimensionless_surface_amplitude_fifth_harmonic(
        steepness, wavenumber, depth, **kwargs
    )

    return -a1 + a2 - a3 + a4 - a5


def dimensionless_stokes_drift(steepness, relative_depth, relative_height=0, **kwargs):

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    order = kwargs.get("order", _DEFAULT_ORDER)
    mu = np.tanh(relative_depth)
    if order < 2:
        return np.zeros_like(steepness * mu)

    if physics_options.wave_regime == "shallow":
        second_order = 0.5 / relative_depth**2

    elif physics_options.wave_regime == "deep":
        second_order = np.exp(2 * relative_height)

    else:
        ch2 = np.cosh(2 * relative_depth + 2 * relative_height) / np.cosh(
            2 * relative_depth
        )
        second_order = (1 + mu**2) / 2 / mu**2 * ch2

    if order < 4:
        return steepness**2 * second_order

    if physics_options.wave_regime == "shallow":
        fourth_order = 9 / 96 / relative_depth**10
    elif physics_options.wave_regime == "deep":
        fourth_order = 2 * np.exp(4 * relative_height) - 0.5 * np.exp(
            2 * relative_height
        )
    else:
        c = 0
        b = (-132 * mu**8 + 40 * mu**6 + 64 * mu**4 - 168 * mu**2 - 60) / (
            128 * mu**6 * (3 * mu**2 + 1)
        )
        a = (
            -21 * mu**10
            - 115 * mu**8
            + 78 * mu**6
            + 218 * mu**4
            + 87 * mu**2
            + 9
        ) / (32 * mu**8 * (3 * mu**2 + 1))
        ch2 = np.cosh(2 * relative_depth + 2 * relative_height) / np.cosh(
            2 * relative_depth
        )
        ch4 = np.cosh(4 * relative_depth + 4 * relative_height) / np.cosh(
            4 * relative_depth
        )

        fourth_order = a * ch4 + b * ch2

    return steepness**2 * second_order + steepness**4 * fourth_order


def dimensionless_lagrangian_mean_location(
    steepness, wavenumber, depth, time, xcoordinate, height, **kwargs
):
    relative_depth = wavenumber * depth
    relative_height = wavenumber * height

    us = dimensionless_stokes_drift(
        steepness, relative_depth, relative_height, **kwargs
    )
    linear_angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    return wavenumber * xcoordinate - us * time * linear_angular_frequency


def dimensionless_lagrangian_setup(
    steepness, relative_depth, relative_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(relative_depth)

    if order < 2:
        return 0

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        lagrangian_amplitude = 1 / 2 * steepness**2 * np.exp(2 * relative_height)

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh2 = np.sinh(2 * relative_depth + 2 * relative_height) / np.cosh(
            2 * relative_depth
        )
        lagrangian_amplitude = (1 + mu**2) / 4 / mu**2 * sh2 * steepness**2

    if order < 4:
        return lagrangian_amplitude

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order setup not implemented for shallow water"
        )
    elif physics_options.wave_regime == "deep":
        # raise NotImplementedError("Fourth order setup not implemented for deep water")
        lagrangian_amplitude += (
            -1 / 2 * np.exp(2 * relative_height) + 1.5 * np.exp(4 * relative_height)
        ) * steepness**4
    else:
        a = (-27 * mu**8 + 12 * mu**6 + 10 * mu**4 - 44 * mu**2 - 15) / (
            32 * mu**6 * (3 * mu**2 + 1)
        )
        b = (
            -45.0 * mu**10
            - 195.0 * mu**8
            + 462.0 * mu**6
            + 426.0 * mu**4
            + 111.0 * mu**2
            + 9.0
        ) / (128 * mu**8 * (3 * mu**2 + 1))

        z = relative_depth + relative_height
        sh2 = np.sinh(2 * z) / np.cosh(2 * relative_depth)
        sh4 = np.sinh(4 * z) / np.cosh(4 * relative_depth)
        lagrangian_amplitude += (a * sh2 + b * sh4) * steepness**4

    return lagrangian_amplitude


def lagrangian_setup(steepness, wavenumber, depth, mean_z, **kwargs):
    relative_depth = wavenumber * depth
    relative_height = wavenumber * mean_z
    return (
        dimensionless_lagrangian_setup(
            steepness, relative_depth, relative_height, **kwargs
        )
        / wavenumber
    )


def stokes_drift(steepness, wavenumber, depth, mean_z, **kwargs):
    """
    This function calculates the wave height of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    linear_angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    c = linear_angular_frequency / wavenumber

    return (
        dimensionless_stokes_drift(
            steepness, wavenumber * depth, wavenumber * mean_z, **kwargs
        )
        * c
    )
