"""
This module implements the third order solution from:

Zhao, K., & Liu, P. L. F. (2022). On Stokes wave solutions. Proceedings of the Royal Society A, 478(2258), 20210732.
"""

import numpy as np

from linearwavetheory.settings import _parse_options
from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_material_surface_amplitude_first_harmonic,
    dimensionless_material_surface_amplitude_second_harmonic,
    dimensionless_material_surface_amplitude_third_harmonic,
    dimensionless_material_surface_amplitude_fourth_harmonic,
)
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER


def lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(
    steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    eulerian_amplitude = dimensionless_material_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
    )

    if order < 3:
        return eulerian_amplitude

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        sh3 = np.exp(3 * wavenumber * lagrangian_mean_surface_height)
        lagrangian_amplitude = 3 / 8 * steepness**3 * sh3

    else:
        kd = wavenumber * depth
        mu = np.tanh(kd)

        sh3 = np.sinh(
            3 * kd + 3 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(3 * kd)
        sh1 = np.sinh(kd + wavenumber * lagrangian_mean_surface_height) / np.cosh(kd)

        # alpha = 3 * (1 - mu ** 2) / mu ** 5 / 32

        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)
        # alpha = alpha - (1 - mu ** 2) / mu ** 3 / 8

        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32

        lagrangian_amplitude = (
            alpha * steepness**3 * sh1 + beta * steepness**3 * sh3
        )

    return lagrangian_amplitude + eulerian_amplitude


def lagrangian_dimensionless_horizontal_displacement_first_harmonic(
    steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(wavenumber * depth)

    kd = wavenumber * depth

    ch1 = np.cosh(kd + wavenumber * lagrangian_mean_surface_height) / np.cosh(
        wavenumber * depth
    )

    if order < 3:
        return -steepness / mu * ch1

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)

        ch1 = np.exp(wavenumber * lagrangian_mean_surface_height)
        ch3 = np.exp(3 * wavenumber * lagrangian_mean_surface_height)
        first_order_lagrangian_amplitude = -1 * ch1

        third_order_lagrangian_amplitude = 19 / 32 * ch1 - 67 / 32 * ch3

    else:

        ch3 = np.cosh(
            3 * kd + 3 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(3 * kd)

        first_order_lagrangian_amplitude = -1 / mu * ch1

        # alpha = (16 - 12 * mu ** 2 + 15 * mu ** 4) / mu ** 5 / 32
        # beta = (- 28 - 72 * mu ** 2 + 33 * mu ** 4) / 32 / mu ** 5
        #
        # alpha = alpha - (1-mu**2)/mu**3/8
        alpha = (10 * mu**4 - 12 * mu**2 + 18) / (32 * mu**5)
        beta = (42 * mu**4 - 76 * mu**2 - 30) / (32 * mu**5)

        third_order_lagrangian_amplitude = alpha * ch1 + beta * ch3

    lagrangian_amplitude = (
        first_order_lagrangian_amplitude * steepness
        + third_order_lagrangian_amplitude * steepness**3
    )
    return lagrangian_amplitude


def lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
    steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
):

    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(wavenumber * depth)

    eulerian_amplitude = dimensionless_material_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
    )

    if order < 2:
        return eulerian_amplitude

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        lagrangian_amplitude = 1 / 2 * steepness**2

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh2 = np.sinh(
            2 * kd + 2 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(2 * kd)
        lagrangian_amplitude = -(1 + mu**2) / 4 / mu**2 * sh2 * steepness**2

    if order < 4:
        return lagrangian_amplitude + eulerian_amplitude

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Fourth order not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        raise NotImplementedError(
            "Fourth order amplitude relation not implemented for deep water waves"
        )

    else:

        a = (
            87 * mu**10
            + 119 * mu**8
            + 150 * mu**6
            + 238 * mu**4
            + 147 * mu**2
            + 27
        ) / (384 * mu**8 * (3 * mu**2 + 1))
        b = (
            141 * mu**10
            + 587 * mu**8
            - 1434 * mu**6
            - 358 * mu**4
            + 141 * mu**2
            + 27
        ) / (192 * mu**8 * (3 * mu**2 + 1))

        sh2 = np.sinh(
            2 * kd + 2 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(2 * kd)
        sh4 = np.sinh(
            4 * kd + 4 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(4 * kd)

        lagrangian_amplitude += (a * sh2 + b * sh4) * steepness**4

    return lagrangian_amplitude + eulerian_amplitude


def lagrangian_dimensionless_horizontal_displacement_second_harmonic(
    steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(kd)
    if order < 2:
        return 0 * steepness * wavenumber

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        lagrangian_amplitude = 0
    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        ch2 = np.cosh(
            2 * kd + 2 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(2 * kd)
        lagrangian_amplitude = (1 - mu**2) / 4 / mu**2 - 3 * (
            1 - mu**4
        ) / mu**4 / 8 * ch2
    return lagrangian_amplitude * steepness**2


def lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(
    steepness, wavenumber, depth, mean_surface_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    if order < 3:
        return np.zeros_like(steepness * wavenumber)

    eulerian_amplitude = dimensionless_material_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, mean_surface_height, **kwargs
    )

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Third harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        sh3 = np.exp(3 * wavenumber * mean_surface_height)

        lagrangian_amplitude = 3 / 8 * steepness**3 * sh3
    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        kd = wavenumber * depth
        mu = np.tanh(kd)

        sh3 = np.sinh(3 * kd + 3 * wavenumber * mean_surface_height) / np.cosh(3 * kd)
        sh1 = np.sinh(kd + wavenumber * mean_surface_height) / np.cosh(kd)

        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32
        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)

        lagrangian_amplitude = (
            -alpha * steepness**3 * sh1 - beta * steepness**3 * sh3
        )

    return lagrangian_amplitude + eulerian_amplitude


def lagrangian_dimensionless_vertical_displacement_amplitude_fourth_harmonic(
    steepness, wavenumber, depth, mean_surface_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    if order < 4:
        return np.zeros_like(steepness * wavenumber)

    eulerian_amplitude = dimensionless_material_surface_amplitude_fourth_harmonic(
        steepness, wavenumber, depth, mean_surface_height, **kwargs
    )

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Fourth harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError("Fourth harmonic not implemented for deep water")
    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        kd = wavenumber * depth
        mu = np.tanh(kd)

        sh4 = np.sinh(4 * kd + 4 * wavenumber * mean_surface_height) / np.cosh(4 * kd)
        sh2 = np.sinh(2 * kd + 2 * wavenumber * mean_surface_height) / np.cosh(2 * kd)

        alpha = (
            237 * mu**10
            - 263 * mu**8
            - 270 * mu**6
            + 290 * mu**4
            + 33 * mu**2
            - 27
        ) / (384 * mu**8 * (3 * mu**2 + 1))
        beta = (
            -147 * mu**10
            - 589 * mu**8
            + 1482 * mu**6
            - 562 * mu**4
            - 615 * mu**2
            - 81
        ) / (384 * mu**8 * (3 * mu**2 + 1))

        lagrangian_amplitude = (
            alpha * steepness**4 * sh2 + beta * steepness**4 * sh4
        )

    return lagrangian_amplitude + eulerian_amplitude


def lagrangian_dimensionless_horizontal_displacement_third_harmonic(
    steepness, wavenumber, depth, lagrangian_mean_surface_height, **kwargs
):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(kd)
    if order < 3:
        return 0 * steepness * wavenumber

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        lagrangian_amplitude = -1 / 32 * np.exp(
            wavenumber * lagrangian_mean_surface_height
        ) + 1 / 32 * np.exp(3 * wavenumber * lagrangian_mean_surface_height)
    else:
        ch1 = np.cosh(kd + wavenumber * lagrangian_mean_surface_height) / np.cosh(kd)
        ch3 = np.cosh(
            3 * kd + 3 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(3 * kd)

        beta = (-39 * mu**6 + 53 * mu**4 - 5 * mu**2 - 9) / (64 * mu**7)
        alpha = (38 * mu**4 - 68 * mu**2 + 30) / (96 * mu**5)
        third_order_lagrangian_amplitude = alpha * ch1 + beta * ch3
        lagrangian_amplitude = third_order_lagrangian_amplitude * steepness**3
    return lagrangian_amplitude
