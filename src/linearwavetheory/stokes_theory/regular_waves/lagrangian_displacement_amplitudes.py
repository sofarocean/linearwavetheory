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
import linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes as eulerian_elevation
from .vertical_eigen_functions import ch, sh
from .utils import get_wave_regime
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

        a = (29 * mu**8 + 30 * mu**6 + 40 * mu**4 + 66 * mu**2 + 27) / (
            384 * mu**8
        )
        b = (47 * mu**8 + 180 * mu**6 - 538 * mu**4 + 60 * mu**2 + 27) / (
            192 * mu**8
        )

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
    dimensionless_depth = kd
    dimensionless_height = wavenumber * lagrangian_mean_surface_height
    mu = np.tanh(dimensionless_depth)
    if order < 2:
        return 0 * steepness * wavenumber

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        second_order = 0
    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        ch2 = np.cosh(
            2 * kd + 2 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(2 * kd)
        second_order = (1 - mu**2) / 4 / mu**2 - 3 * (
            1 - mu**4
        ) / mu**4 / 8 * ch2

    if order < 4:
        return second_order * steepness**2

    if physics_options.wave_regime == "shallow":
        fourth_order = 27 / 256 / mu**10

    elif physics_options.wave_regime == "deep":
        fourth_order = (
            -0.5 * np.exp(dimensionless_height) + np.exp(4 * dimensionless_height) / 6
        )

    else:
        c = (11 * mu**6 - 19 * mu**4 + 29 * mu**2 - 21) / (96 * mu**6)
        b = (
            39 * mu**12
            + 724 * mu**10
            - 1269 * mu**8
            - 2032 * mu**6
            + 381 * mu**4
            + 540 * mu**2
            + 81
        ) / (768 * mu**10 * (3 * mu**2 + 1))
        a = (
            -345 * mu**10
            - 1411 * mu**8
            + 3462 * mu**6
            - 358 * mu**4
            - 957 * mu**2
            - 135
        ) / (384 * mu**8 * (3 * mu**2 + 1))

        ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)
        ch4 = ch(dimensionless_depth, dimensionless_height, 4, **kwargs)
        fourth_order = c + b * ch2 + a * ch4

    return second_order * steepness**2 + fourth_order * steepness**4


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
            79 * mu**8
            - 114 * mu**6
            - 52 * mu**4
            + 114 * mu**2
            - 27
            # 237 * mu ** 10
            # - 263 * mu ** 8
            # - 270 * mu ** 6
            # + 290 * mu ** 4
            # + 33 * mu ** 2
            # - 27
        ) / (384 * mu**8)
        beta = (
            -49 * mu**8
            - 180 * mu**6
            + 554 * mu**4
            - 372 * mu**2
            - 81
            # -147 * mu ** 10
            # - 589 * mu ** 8
            # + 1482 * mu ** 6
            # - 562 * mu ** 4
            # - 615 * mu ** 2
            # - 81
        ) / (384 * mu**8)

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
        lagrangian_amplitude = 0
    else:
        ch1 = np.cosh(kd + wavenumber * lagrangian_mean_surface_height) / np.cosh(kd)
        ch3 = np.cosh(
            3 * kd + 3 * wavenumber * lagrangian_mean_surface_height
        ) / np.cosh(3 * kd)

        beta = (-39 * mu**6 + 53 * mu**4 - 5 * mu**2 - 9) / (64 * mu**7)
        alpha = (38 * mu**4 - 68 * mu**2 + 30) / (96 * mu**5)
        lagrangian_amplitude = alpha * ch1 + beta * ch3
    return lagrangian_amplitude * steepness**3


def x11(dimensionless_depth, dimensionless_height, **kwargs):

    wave_regime = get_wave_regime(**kwargs)
    ch1 = ch(dimensionless_depth, dimensionless_height, 1, **kwargs)

    if wave_regime == "shallow":
        mu = dimensionless_depth
        return 1 / mu

    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        amplitude = -ch1
    else:
        mu = np.tanh(dimensionless_depth)
        amplitude = -1 / mu * ch1
    return amplitude


def x22(dimensionless_depth, dimensionless_height, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(dimensionless_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        amplitude = 0
    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)
        amplitude = (1 - mu**2) / 4 / mu**2 - 3 * (1 - mu**4) / mu**4 / 8 * ch2
    return amplitude


def x31(dimensionless_depth, dimensionless_height, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(dimensionless_depth)

    ch1 = ch(dimensionless_depth, dimensionless_height, 1, **kwargs)
    ch3 = ch(dimensionless_depth, dimensionless_height, 3, **kwargs)

    if wave_regime == "shallow":
        amplitude = -3 / (8 * mu**5)
    elif wave_regime == "deep":
        amplitude = 0.5 * ch1 - 2 * ch3
    else:
        a = (10 * mu**4 - 12 * mu**2 + 18) / (32 * mu**5)
        b = (42 * mu**4 - 76 * mu**2 - 30) / (32 * mu**5)
        amplitude = a * ch1 + b * ch3

    return amplitude


def x33(dimensionless_depth, dimensionless_height, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(dimensionless_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        amplitude = -9 / (64 * mu**7)
    elif wave_regime == "deep":
        amplitude = 0
    else:
        ch1 = ch(dimensionless_depth, dimensionless_height, 1, **kwargs)
        ch3 = ch(dimensionless_depth, dimensionless_height, 3, **kwargs)
        D = 1 - mu**2
        beta = D * (39 * mu**4 - 14 * mu**2 - 9) / (64 * mu**7)
        alpha = D * (30 - 38 * mu**2) / (96 * mu**5)
        amplitude = alpha * ch1 + beta * ch3
    return amplitude


def x42(dimensionless_depth, dimensionless_height, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(dimensionless_depth)

    if wave_regime == "shallow":
        amplitude = 27 / 256 / mu**10

    elif wave_regime == "deep":
        amplitude = (
            -0.5 * np.exp(dimensionless_height) + np.exp(4 * dimensionless_height) / 6
        )
    else:
        # c = (11 * mu ** 6 - 19 * mu ** 4 + 29 * mu ** 2 - 21) / (96 * mu ** 6)
        # b = (39 * mu ** 12 + 724 * mu ** 10 - 1269 * mu ** 8 - 2032 * mu ** 6 + 381 * mu ** 4 + 540 * mu ** 2 + 81) / (
        #         768 * mu ** 10 * (3 * mu ** 2 + 1))
        # a = (-345 * mu ** 10 - 1411 * mu ** 8 + 3462 * mu ** 6 - 358 * mu ** 4 - 957 * mu ** 2 - 135) / (
        #         384 * mu ** 8 * (3 * mu ** 2 + 1))

        c = (1 - mu**2) * (-11 * mu**4 + 8 * mu**2 - 21) / (96 * mu**6)
        b = (
            13 * mu**10
            + 237 * mu**8
            - 502 * mu**6
            - 510 * mu**4
            + 297 * mu**2
            + 81
        ) / (768 * mu**10)
        a = (-115 * mu**8 - 432 * mu**6 + 1298 * mu**4 - 552 * mu**2 - 135) / (
            384 * mu**8
        )

        ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)
        ch4 = ch(dimensionless_depth, dimensionless_height, 4, **kwargs)
        amplitude = c + b * ch2 + a * ch4
    return amplitude


def x44(dimensionless_depth, dimensionless_height, **kwargs):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    wave_regime = get_wave_regime(**kwargs)

    mu = np.tanh(dimensionless_depth)
    ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)
    ch4 = ch(dimensionless_depth, dimensionless_height, 4, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        amplitude = -27 / (512 * mu**10)
    elif wave_regime == "deep":
        amplitude = 0
    else:
        D = 1 - mu**2
        c = D**2 * (83 * mu**4 - 102 * mu**2 + 27) / (384 * mu**8)
        b = D * (263 * mu**6 - 127 * mu**4 - 255 * mu**2 + 135) / (768 * mu**8)
        a = D * (
            (
                -197 * mu**10
                - 1929 * mu**8
                - 3410 * mu**6
                + 6462 * mu**4
                - 1161 * mu**2
                - 405
            )
            / (1536 * mu**10 * (mu**2 + 5))
        )

        amplitude = c + b * ch2 + a * ch4
    return amplitude


def ul20(dimensionless_depth, dimensionless_height, **kwargs):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    wave_regime = get_wave_regime(**kwargs)

    mu = np.tanh(dimensionless_depth)
    ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        amplitude = 1 / 2 / mu**2
    elif wave_regime == "deep":
        amplitude = ch2
    else:
        amplitude = (1 + mu**2) / 2 / mu**2 * ch2
    return amplitude


def ul40(dimensionless_depth, dimensionless_height, **kwargs):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    wave_regime = get_wave_regime(**kwargs)

    ch2 = ch(dimensionless_depth, dimensionless_height, 2, **kwargs)
    ch4 = ch(dimensionless_depth, dimensionless_height, 4, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        mu = dimensionless_depth
        amplitude = 9 / 96 / mu**10
    elif wave_regime == "deep":
        amplitude = 2 * ch4 - 0.5 * ch2
    else:
        mu = np.tanh(dimensionless_depth)
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

        b = (-11 * mu**6 + 7 * mu**4 + 3 * mu**2 - 15) / (32 * mu**6)
        a = (-7 * mu**8 - 36 * mu**6 + 38 * mu**4 + 60 * mu**2 + 9) / (
            32 * mu**8
        )

        amplitude = a * ch4 + b * ch2
    return amplitude


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


def eta11(relative_depth, relative_height, **kwargs):
    return eulerian_elevation.eta11(relative_depth, relative_height, **kwargs)

def eta31(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_height, 1, **kwargs)
    sh3 = sh(relative_depth, relative_height, 3, **kwargs)

    if wave_regime == "shallow":
        beta = 9 / mu**5 / 32
        alpha =3 / mu**5 / 32

        lagrangian_amplitude = alpha *sh1 + beta * sh3
    elif wave_regime == "deep":
        lagrangian_amplitude = 3 / 8 * sh3
    else:
        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)
        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32

        lagrangian_amplitude = alpha *sh1 + beta * sh3

    return lagrangian_amplitude + eulerian_elevation.eta31(relative_depth, relative_height, **kwargs)


def eta22(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2, **kwargs)


    if wave_regime == "shallow":
        lagrangian_amplitude = -1 / 4 / mu**2 * sh2

    elif wave_regime == "deep":
        lagrangian_amplitude = -1 / 2 * sh2

    else:

        lagrangian_amplitude = -(1 + mu**2) / 4 / mu**2 * sh2

    return lagrangian_amplitude + eulerian_elevation.eta22(relative_depth, relative_height, **kwargs)

def eta42(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2, **kwargs)
    sh4 = sh(relative_depth, relative_height, 4, **kwargs)

    if wave_regime == "shallow":
        a =  27 / (384 * mu**8)
        b =27 / (192 * mu**8)

        lagrangian_amplitude = (a * sh2 + b * sh4)

    elif wave_regime == "deep":
        lagrangian_amplitude = 0.5 * sh2 - 7/6 * sh4

    else:

        a = (29 * mu**8 + 30 * mu**6 + 40 * mu**4 + 66 * mu**2 + 27) / (
            384 * mu**8
        )
        b = (47 * mu**8 + 180 * mu**6 - 538 * mu**4 + 60 * mu**2 + 27) / (
            192 * mu**8
        )

        lagrangian_amplitude = (a * sh2 + b * sh4)

    return lagrangian_amplitude + eulerian_elevation.eta42(relative_depth, relative_height, **kwargs)


def eta33(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_height, 1, **kwargs)
    sh3 = sh(relative_depth, relative_height, 3, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        alpha =3 / mu ** 5 / 32
        beta = 9 / mu ** 5 / 32

        lagrangian_amplitude = -alpha * sh1 - beta * sh3

    elif wave_regime == "deep":
        lagrangian_amplitude = 3 / 8 *  sh3
    else:

        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32
        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)

        lagrangian_amplitude = -alpha  * sh1 - beta *  sh3

    return lagrangian_amplitude + eulerian_elevation.eta33(relative_depth, relative_height, **kwargs)

def eta44(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2, **kwargs)
    sh4 = sh(relative_depth, relative_height, 4, **kwargs)

    if wave_regime == "shallow":
        alpha = - 3/ (128 * mu**8)
        beta = - 27 / (128 * mu**8)
        lagrangian_amplitude = alpha  * sh2 + beta  * sh4

    elif wave_regime == "deep":
        lagrangian_amplitude = - sh4/3
    else:
        alpha = (
            79 * mu**8
            - 114 * mu**6
            - 52 * mu**4
            + 114 * mu**2
            - 27
        ) / (384 * mu**8)
        beta = (
            -49 * mu**8
            - 180 * mu**6
            + 554 * mu**4
            - 372 * mu**2
            - 81
        ) / (384 * mu**8)

        lagrangian_amplitude = alpha  * sh2 + beta  * sh4

    return lagrangian_amplitude + eulerian_elevation.eta44(relative_depth, relative_height, **kwargs)


def eta20(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2, **kwargs)
    sh4 = sh(relative_depth, relative_height, 4, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        lagrangian_amplitude = 1/ 4 / mu**2 * sh2

    elif wave_regime == "deep":
        lagrangian_amplitude = 1 / 2  * sh2

    else:
        lagrangian_amplitude = (1 + mu**2) / 4 / mu**2 * sh2

    return lagrangian_amplitude


def eta40(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2, **kwargs)
    sh4 = sh(relative_depth, relative_height, 4, **kwargs)

    if wave_regime == "shallow":
        lagrangian_amplitude = 9.0 / (128 * mu**8) * sh4

    elif wave_regime == "deep":
        lagrangian_amplitude = -1 / 2 * sh2 + 1.5 * sh4

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

        lagrangian_amplitude = (a * sh2 + b * sh4)

    return lagrangian_amplitude