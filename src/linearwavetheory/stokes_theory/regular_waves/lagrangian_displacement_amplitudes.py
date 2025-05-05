"""
This module implements the third order solution from:

Zhao, K., & Liu, P. L. F. (2022). On Stokes wave solutions. Proceedings of the Royal Society A, 478(2258), 20210732.
"""

import numpy as np
import linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes as eulerian_elevation
from .vertical_eigen_functions import ch, sh
from .utils import get_wave_regime
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER


def dimensionless_horizontal_displacement_amplitude(
    steepness, relative_depth, relative_z, harmonic_number, **kwargs
):
    amplitude = 0.0
    order = kwargs.get("order", _DEFAULT_ORDER)
    match harmonic_number:
        case 1:
            _x11 = x11(relative_depth, relative_z, **kwargs)
            _x31 = 0 if order < 3 else x31(relative_depth, relative_z, **kwargs)
            amplitude = steepness * _x11 + _x31 * steepness**3
        case 2:
            _x22 = 0 if order < 2 else x22(relative_depth, relative_z, **kwargs)
            _x42 = 0 if order < 4 else x42(relative_depth, relative_z, **kwargs)
            amplitude = _x22 * steepness**2 + _x42 * steepness**4
        case 3:
            _x33 = 0 if order < 3 else x33(relative_depth, relative_z, **kwargs)
            amplitude = _x33 * steepness**3
        case 4:
            _x44 = 0 if order < 4 else x44(relative_depth, relative_z, **kwargs)
            amplitude = _x44 * steepness**4
        case 5:
            raise NotImplementedError(
                "Fifth order horizontal displacement amplitude not implemented"
            )
        case _:
            raise ValueError(
                f"Harmonic number {harmonic_number} is not supported. Supported values are 1, 2, 3, 4, and 5."
            )

    return amplitude


def dimensionless_vertical_displacement_amplitude(
    steepness, relative_depth, relative_z, harmonic_number, **kwargs
):
    amplitude = 0.0
    order = kwargs.get("order", _DEFAULT_ORDER)
    match harmonic_number:
        case 1:
            _eta11 = eta11(relative_depth, relative_z, **kwargs)
            _eta31 = 0 if order < 3 else eta31(relative_depth, relative_z, **kwargs)
            amplitude = steepness * _eta11 + _eta31 * steepness**3
        case 2:
            _eta22 = 0 if order < 2 else eta22(relative_depth, relative_z, **kwargs)
            _eta42 = 0 if order < 4 else eta42(relative_depth, relative_z, **kwargs)
            amplitude = _eta22 * steepness**2 + _eta42 * steepness**4
        case 3:
            _eta33 = 0 if order < 3 else eta33(relative_depth, relative_z, **kwargs)
            amplitude = _eta33 * steepness**3
        case 4:
            _eta44 = 0 if order < 4 else eta44(relative_depth, relative_z, **kwargs)
            amplitude = _eta44 * steepness**4
        case 5:
            raise NotImplementedError(
                "Fifth order vertical displacement amplitude not implemented"
            )
        case _:
            raise ValueError(
                f"Harmonic number {harmonic_number} is not supported. Supported values are 1, 2, 3, 4, and 5."
            )

    return amplitude


def x11(relative_depth, relative_z, **kwargs):

    wave_regime = get_wave_regime(**kwargs)
    ch1 = ch(relative_depth, relative_z, 1, **kwargs)

    if wave_regime == "shallow":
        mu = relative_depth
        return 1 / mu

    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        amplitude = -ch1
    else:
        mu = np.tanh(relative_depth)
        amplitude = -1 / mu * ch1
    return amplitude


def x22(relative_depth, relative_z, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif wave_regime == "deep":
        amplitude = 0
    else:

        ch2 = ch(relative_depth, relative_z, 2, **kwargs)
        amplitude = (1 - mu**2) / 4 / mu**2 - 3 * (1 - mu**4) / mu**4 / 8 * ch2
    return amplitude


def x31(relative_depth, relative_z, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    ch1 = ch(relative_depth, relative_z, 1, **kwargs)
    ch3 = ch(relative_depth, relative_z, 3, **kwargs)

    if wave_regime == "shallow":
        amplitude = -3 / (8 * mu**5)
    elif wave_regime == "deep":
        amplitude = 0.5 * ch1 - 2 * ch3
    else:
        a = (10 * mu**4 - 12 * mu**2 + 18) / (32 * mu**5)
        b = (42 * mu**4 - 76 * mu**2 - 30) / (32 * mu**5)
        amplitude = a * ch1 + b * ch3

    return amplitude


def x33(relative_depth, relative_z, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        amplitude = -9 / (64 * mu**7)
    elif wave_regime == "deep":
        amplitude = 0
    else:
        ch1 = ch(relative_depth, relative_z, 1, **kwargs)
        ch3 = ch(relative_depth, relative_z, 3, **kwargs)
        D = 1 - mu**2
        beta = D * (39 * mu**4 - 14 * mu**2 - 9) / (64 * mu**7)
        alpha = D * (30 - 38 * mu**2) / (96 * mu**5)
        amplitude = alpha * ch1 + beta * ch3
    return amplitude


def x42(relative_depth, relative_z, **kwargs):
    """

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        amplitude = 27 / 256 / mu**10

    elif wave_regime == "deep":
        amplitude = -0.5 * np.exp(relative_z) + np.exp(4 * relative_z) / 6
    else:
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

        ch2 = ch(relative_depth, relative_z, 2, **kwargs)
        ch4 = ch(relative_depth, relative_z, 4, **kwargs)
        amplitude = c + b * ch2 + a * ch4
    return amplitude


def x44(relative_depth, relative_z, **kwargs):
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

    mu = np.tanh(relative_depth)
    ch2 = ch(relative_depth, relative_z, 2, **kwargs)
    ch4 = ch(relative_depth, relative_z, 4, **kwargs)

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


def ul20(relative_depth, relative_z, **kwargs):
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

    mu = np.tanh(relative_depth)
    ch2 = ch(relative_depth, relative_z, 2, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        amplitude = 1 / 2 / mu**2
    elif wave_regime == "deep":
        amplitude = ch2
    else:
        amplitude = (1 + mu**2) / 2 / mu**2 * ch2
    return amplitude


def ul40(relative_depth, relative_z, **kwargs):
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

    ch2 = ch(relative_depth, relative_z, 2, **kwargs)
    ch4 = ch(relative_depth, relative_z, 4, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        mu = relative_depth
        amplitude = 9 / 96 / mu**10
    elif wave_regime == "deep":
        amplitude = 2 * ch4 - 0.5 * ch2
    else:
        mu = np.tanh(relative_depth)
        b = (-11 * mu**6 + 7 * mu**4 + 3 * mu**2 - 15) / (32 * mu**6)
        a = (-7 * mu**8 - 36 * mu**6 + 38 * mu**4 + 60 * mu**2 + 9) / (
            32 * mu**8
        )

        amplitude = a * ch4 + b * ch2
    return amplitude


def eta11(relative_depth, relative_z, **kwargs):
    return eulerian_elevation.eta11(relative_depth, relative_z, **kwargs)


def eta31(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_z, 1, **kwargs)
    sh3 = sh(relative_depth, relative_z, 3, **kwargs)

    if wave_regime == "shallow":
        beta = 9 / mu**5 / 32
        alpha = 3 / mu**5 / 32

        lagrangian_amplitude = alpha * sh1 + beta * sh3
    elif wave_regime == "deep":
        lagrangian_amplitude = 3 / 8 * sh3
    else:
        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)
        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32

        lagrangian_amplitude = alpha * sh1 + beta * sh3

    return lagrangian_amplitude + eulerian_elevation.eta31(
        relative_depth, relative_z, **kwargs
    )


def eta22(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2, **kwargs)

    if wave_regime == "shallow":
        lagrangian_amplitude = -1 / 4 / mu**2 * sh2

    elif wave_regime == "deep":
        lagrangian_amplitude = -1 / 2 * sh2

    else:

        lagrangian_amplitude = -(1 + mu**2) / 4 / mu**2 * sh2

    return lagrangian_amplitude + eulerian_elevation.eta22(
        relative_depth, relative_z, **kwargs
    )


def eta42(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2, **kwargs)
    sh4 = sh(relative_depth, relative_z, 4, **kwargs)

    if wave_regime == "shallow":
        a = 27 / (384 * mu**8)
        b = 27 / (192 * mu**8)

        lagrangian_amplitude = a * sh2 + b * sh4

    elif wave_regime == "deep":
        lagrangian_amplitude = 0.5 * sh2 - 7 / 6 * sh4

    else:

        a = (29 * mu**8 + 30 * mu**6 + 40 * mu**4 + 66 * mu**2 + 27) / (
            384 * mu**8
        )
        b = (47 * mu**8 + 180 * mu**6 - 538 * mu**4 + 60 * mu**2 + 27) / (
            192 * mu**8
        )

        lagrangian_amplitude = a * sh2 + b * sh4

    return lagrangian_amplitude + eulerian_elevation.eta42(
        relative_depth, relative_z, **kwargs
    )


def eta33(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_z, 1, **kwargs)
    sh3 = sh(relative_depth, relative_z, 3, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        alpha = 3 / mu**5 / 32
        beta = 9 / mu**5 / 32

        lagrangian_amplitude = -alpha * sh1 - beta * sh3

    elif wave_regime == "deep":
        lagrangian_amplitude = 3 / 8 * sh3
    else:

        alpha = (1 - mu**2) * (3 - 4 * mu**2) / mu**5 / 32
        beta = 1 / mu**5 / 32 * (21 * mu**2 - 18 * mu**4 + 9)

        lagrangian_amplitude = -alpha * sh1 - beta * sh3

    return lagrangian_amplitude + eulerian_elevation.eta33(
        relative_depth, relative_z, **kwargs
    )


def eta44(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2, **kwargs)
    sh4 = sh(relative_depth, relative_z, 4, **kwargs)

    if wave_regime == "shallow":
        alpha = -3 / (128 * mu**8)
        beta = -27 / (128 * mu**8)
        lagrangian_amplitude = alpha * sh2 + beta * sh4

    elif wave_regime == "deep":
        lagrangian_amplitude = -sh4 / 3
    else:
        alpha = (79 * mu**8 - 114 * mu**6 - 52 * mu**4 + 114 * mu**2 - 27) / (
            384 * mu**8
        )
        beta = (-49 * mu**8 - 180 * mu**6 + 554 * mu**4 - 372 * mu**2 - 81) / (
            384 * mu**8
        )

        lagrangian_amplitude = alpha * sh2 + beta * sh4

    return lagrangian_amplitude + eulerian_elevation.eta44(
        relative_depth, relative_z, **kwargs
    )


def eta20(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2, **kwargs)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        lagrangian_amplitude = 1 / 4 / mu**2 * sh2

    elif wave_regime == "deep":
        lagrangian_amplitude = 1 / 2 * sh2

    else:
        lagrangian_amplitude = (1 + mu**2) / 4 / mu**2 * sh2

    return lagrangian_amplitude


def eta40(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2, **kwargs)
    sh4 = sh(relative_depth, relative_z, 4, **kwargs)

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

        lagrangian_amplitude = a * sh2 + b * sh4

    return lagrangian_amplitude
