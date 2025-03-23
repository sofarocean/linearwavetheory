import numpy as np

from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from .utils import get_wave_regime
from .vertical_eigen_functions import sh


def dimensionless_surface_amplitude(
    steepness, relative_depth, harmonic_number, **kwargs
):
    amplitude = 0.0
    order = kwargs.get("order", _DEFAULT_ORDER)
    match harmonic_number:
        case 1:
            _a11 = a11(relative_depth, **kwargs)
            _a31 = 0 if order < 3 else a31(relative_depth, **kwargs)
            _a51 = 0 if order < 5 else a51(relative_depth, **kwargs)
            amplitude = steepness * _a11 + _a31 * steepness**3 + _a51 * steepness**5
        case 2:
            _a22 = 0 if order < 2 else a22(relative_depth, **kwargs)
            _a42 = 0 if order < 4 else a42(relative_depth, **kwargs)
            amplitude = _a22 * steepness**2 + _a42 * steepness**4
        case 3:
            _a33 = 0 if order < 3 else a33(relative_depth, **kwargs)
            _a53 = 0 if order < 5 else a53(relative_depth, **kwargs)
            amplitude = _a33 * steepness**3 + _a53 * steepness**5
        case 4:
            _a44 = 0 if order < 4 else a44(relative_depth, **kwargs)
            amplitude = _a44 * steepness**4
        case 5:
            _a55 = 0 if order < 5 else a55(relative_depth, **kwargs)
            amplitude = _a55 * steepness**5
        case _:
            raise ValueError(
                f"Harmonic number {harmonic_number} is not supported. Supported values are 1, 2, 3, 4, and 5."
            )

    return amplitude


def dimensionless_material_surface_amplitude(
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
                "Fifth order material surface amplitude not implemented"
            )
        case _:
            raise ValueError(
                f"Harmonic number {harmonic_number} is not supported. Supported values are 1, 2, 3, 4, and 5."
            )

    return amplitude


def a11(relative_depth, **kwargs):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    return 1


def a31(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        third_order = 3 / 16 / mu**4
    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        third_order = 0.125

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        third_order = (3 + 8 * mu**2 - 9 * mu**4) / 16 / mu**4

    return third_order


def a51(relative_depth, **kwargs):
    """
    This is the direct coeficient as in Zhao and Liu (2022) for the fifth order surface amplitude. I
    did not re-express it in terms of mu.
    :param relative_depth:
    :param kwargs:
    :return:
    """
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for deep water waves"
        )
    else:
        alpha = (1 + mu**2) / (1 - mu**2)
        fifth_order = (
            121 * alpha**5
            + 263 * alpha**4
            + 376 * alpha**3
            - 1999 * alpha**2
            + 2509 * alpha
            - 1108
        ) / (192 * (alpha - 1) ** 5)
    return fifth_order


def a22(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        second_order = 3 / 4 / mu**3

    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        second_order = 0.5

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        second_order = (3 - mu**2) / 4 / mu**3

    return second_order


def a42(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fourtg order amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for deep water waves"
        )

    else:
        fourth_order = (
            129 * mu**8 - 826 * mu**6 + 1152 * mu**4 - 54 * mu**2 - 81
        ) / (384 * mu**9)

    return fourth_order


def a33(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        third_order = 27 / 64 / mu**6

    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        third_order = 0.375

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        third_order = (27 - 9 * mu**2 + 9 * mu**4 - 3 * mu**6) / 64 / mu**6
    return third_order


def a53(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for deep water waves"
        )

    else:
        alpha = (1 + mu**2) / (1 - mu**2)
        fifth_order = (
            9
            * (
                57 * alpha**7
                + 204 * alpha**6
                - 53 * alpha**5
                - 782 * alpha**4
                - 741 * alpha**3
                - 52 * alpha**2
                + 371 * alpha
                + 186
            )
            / (128 * (alpha - 1) ** 6 * (3 * alpha + 2))
        )
    return fifth_order


def a44(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        fourth_order = 81 / 384 / mu**9
    elif wave_regime == "deep":
        fourth_order = 1 / 3
    else:
        fourth_order = (
            84 * mu**11
            + 4 * mu**9
            - 1048 * mu**7
            + 2088 * mu**5
            + 324 * mu**3
            + 1620 * mu
        ) / (1536 * mu**10 * (mu**2 + 5))

    return fourth_order


def a55(relative_depth, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for deep water waves"
        )

    else:
        alpha = np.cosh(2 * relative_depth)  # ((1 + mu ** 2) / (1 - mu ** 2))
        fifth_order = (
            5
            * (
                300 * alpha**8
                + 1579 * alpha**7
                + 3176 * alpha**6
                + 2949 * alpha**5
                + 1188 * alpha**4
                + 675 * alpha**3
                + 1326 * alpha**2
                + 827 * alpha
                + 130
            )
            / (384 * (alpha - 1) ** 6 * (12 * alpha**2 + 11 * alpha + 2))
        )
    return fifth_order


def eta11(relative_depth, relative_z, **kwargs):
    """
    This function calculates the surface amplitude of the primary harmonic wave of a third order Stokes solution. Note
    that the primary harmonic contains a third order correction.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_z, 1)
    return 1 / mu * sh1


def eta22(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2)

    if wave_regime == "shallow":

        raise NotImplementedError("Second harmonic not implemented for shallow water")

    elif wave_regime == "deep":
        second_order = 0.5 * sh2

    else:
        second_order = (1 + mu**2) * (3 - mu**2) / 8 / mu**4 * sh2
    return second_order


def eta31(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh1 = sh(relative_depth, relative_z, 1)
    sh3 = sh(relative_depth, relative_z, 3)

    if wave_regime == "shallow":
        a = -21 / 32 / mu**5
        b = 9 / 32 / mu**5
        amplitude = a * sh1 + b * sh3

    elif wave_regime == "deep":
        amplitude = -1 / 2 * sh1 + 0.625 * sh3

    else:
        a = +(-21 + 19 * mu**2 - 14 * mu**4) / 32 / mu**5
        b = (9 + 23 * mu**2 - 12 * mu**4) / 32 / mu**5
        amplitude = a * sh1 + b * sh3
    return amplitude


def eta42(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_z, 2)
    sh4 = sh(relative_depth, relative_z, 4)

    if wave_regime == "shallow":
        fourth_order = -27 / (256 * mu**10) * sh2

    elif wave_regime == "deep":
        fourth_order = 5 * sh4 / 6
    else:

        a = (
            89 * mu**10
            - 681 * mu**8
            + 262 * mu**6
            + 762 * mu**4
            - 351 * mu**2
            - 81
        ) / (768 * mu**10)
        b = (
            (1 + 6 * mu**2 + mu**4)
            * (27 - 12 * mu**2 + 5 * mu**4)
            / mu**8
            / 192
        )
        fourth_order = a * sh2 + b * sh4

    return fourth_order


def eta33(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh1 = sh(relative_depth, relative_z, 1)
    sh3 = sh(relative_depth, relative_z, 3)

    if wave_regime == "shallow":

        amplitude = 9 / mu**7 / 64 * sh3

    elif wave_regime == "deep":
        amplitude = 3 / 8 * sh3

    else:

        a = (1 - mu**2) * (-9 + 6 * mu**2) / 32 / mu**5
        b = (27 + 69 * mu**2 + 9 * mu**6 - 33 * mu**4) / mu**7 / 64
        amplitude = (a * sh1 + b * sh3) / 3

    return amplitude


def eta44(relative_depth, relative_z, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh2 = sh(relative_depth, relative_z, 2)
    sh4 = sh(relative_depth, relative_z, 4)

    if wave_regime == "shallow":
        amplitude = 81 / mu**10 / 1536 * sh4

    elif wave_regime == "deep":
        amplitude = sh4 / 3

    else:

        a = (1 - mu**4) * (-27 + 30 * mu**2 - 11 * mu**4) / mu**8 / 384
        b = (
            (
                -(mu**12)
                - 32 * mu**10
                - 97 * mu**8
                + 280 * mu**6
                + 141 * mu**4
                + 2376 * mu**2
                + 405
            )
            / mu**10
            / 1536
            / (5 + mu**2)
        )

        amplitude = a * sh2 + b * sh4
    return amplitude
