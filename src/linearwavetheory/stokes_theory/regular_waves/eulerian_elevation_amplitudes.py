import numpy as np

from linearwavetheory.settings import _parse_options
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from .utils import get_wave_regime
from .vertical_eigen_functions import sh, ch

def dimensionless_surface_amplitude_first_harmonic(
    steepness, wavenumber, depth, **kwargs
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

    if order < 3:
        return steepness * wavenumber / wavenumber

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        third_order = steepness * (1 + steepness**2 * 3 / 16 / mu**4)
    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        third_order = steepness * (1 + 0.125 * steepness**2)

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        third_order = steepness * (
            1 + steepness**2 * (3 + 8 * mu**2 - 9 * mu**4) / 16 / mu**4
        )

    if order < 5:
        return third_order

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for deep water waves"
        )

    else:
        alpha = (1 + mu**2) / (1 - mu**2)
        fifth_order = (
            (
                121 * alpha**5
                + 263 * alpha**4
                + 376 * alpha**3
                - 1999 * alpha**2
                + 2509 * alpha
                - 1108
            )
            / (192 * (alpha - 1) ** 5)
        ) * steepness**5
    return third_order + fifth_order


def dimensionless_surface_amplitude_second_harmonic(
    steepness, wavenumber, depth, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(wavenumber * depth)

    if order < 2:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        second_order = steepness**2 * 3 / 4 / kd**3

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        second_order = 0.5 * steepness**2

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        second_order = steepness**2 * ((3 - mu**2) / 4 / mu**3)

    if order < 4:
        return second_order

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourtg order amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for deep water waves"
        )

    else:
        fourth_order = (
            steepness**4
            * (129 * mu**8 - 826 * mu**6 + 1152 * mu**4 - 54 * mu**2 - 81)
            / (384 * mu**9)
        )

    return second_order + fourth_order


def dimensionless_surface_amplitude_third_harmonic(
    steepness, wavenumber, depth, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(kd)
    if order < 3:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        third_order = steepness**3 * 27 / 64 / kd**6

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        third_order = 0.375 * steepness**3

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        third_order = (
            steepness**3
            * (27 - 9 * mu**2 + 9 * mu**4 - 3 * mu**6)
            / 64
            / mu**6
        )

    if order < 5:
        return third_order

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
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
        ) * steepness**5
    return third_order + fifth_order


def dimensionless_surface_amplitude_fourth_harmonic(
    steepness, wavenumber, depth, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(kd)
    if order < 4:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourtg order amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for deep water waves"
        )

    else:
        fourth_order = (
            84 * mu**11
            + 4 * mu**9
            - 1048 * mu**7
            + 2088 * mu**5
            + 324 * mu**3
            + 1620 * mu
        ) / (1536 * mu**10 * (mu**2 + 5))

    return fourth_order * steepness**4


def dimensionless_surface_amplitude_fifth_harmonic(
    steepness, wavenumber, depth, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    kd = wavenumber * depth
    mu = np.tanh(kd)
    if order < 5:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order surface amplitude not implemented for deep water waves"
        )

    else:
        alpha = np.cosh(2 * kd)  # ((1 + mu ** 2) / (1 - mu ** 2))
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
    return fifth_order * steepness**5


def dimensionless_material_surface_amplitude_first_harmonic(
    steepness, wavenumber, depth, mean_material_surface_height, **kwargs
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
    sh1 = np.sinh(kd + wavenumber * mean_material_surface_height) / np.cosh(
        wavenumber * depth
    )

    if order < 3:
        return steepness / mu * sh1

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")
    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        sh3 = np.sinh(3 * kd + 3 * wavenumber * mean_material_surface_height) / np.cosh(
            3 * kd
        )
        return steepness * (1 - steepness**2 / 2) * np.exp(
            wavenumber * mean_material_surface_height
        ) + 0.625 * steepness**3 * np.exp(
            3 * wavenumber * mean_material_surface_height
        )

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh3 = np.sinh(3 * kd + 3 * wavenumber * mean_material_surface_height) / np.cosh(
            3 * kd
        )
        a = +(-21 + 19 * mu**2 - 14 * mu**4) / 32 / mu**5

        b = (9 + 23 * mu**2 - 12 * mu**4) / 32 / mu**5
        return (
            steepness * (1 / mu + steepness**2 * a) * sh1 + steepness**3 * b * sh3
        )


def dimensionless_material_surface_amplitude_second_harmonic(
    steepness, wavenumber, depth, mean_material_surface_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )
    kd = wavenumber * depth
    mu = np.tanh(wavenumber * depth)

    if order < 2:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Second harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)
        second_order = (
            0.5 * steepness**2 * np.exp(wavenumber * mean_material_surface_height)
        )

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh2 = np.sinh(2 * kd + 2 * wavenumber * mean_material_surface_height) / np.cosh(
            2 * kd
        )
        second_order = (
            steepness**2 * (1 + mu**2) * (3 - mu**2) / 8 / mu**4 * sh2
        )

    if order < 4:
        return second_order

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order amplitude not implemented for deep water waves"
        )

    else:
        sh2 = np.sinh(2 * kd + 2 * wavenumber * mean_material_surface_height) / np.cosh(
            2 * kd
        )
        sh4 = np.sinh(4 * kd + 4 * wavenumber * mean_material_surface_height) / np.cosh(
            4 * kd
        )
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
        fourth_order = steepness**4 * (a * sh2 + b * sh4)

    return second_order + fourth_order


def dimensionless_material_surface_amplitude_third_harmonic(
    steepness, wavenumber, depth, mean_material_surface_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )
    kd = wavenumber * depth
    mu = np.tanh(kd)

    if order < 3:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Third harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        return (
            3
            / 8
            * steepness**3
            * np.exp(3 * wavenumber * mean_material_surface_height)
        )

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh1 = np.sinh(kd + wavenumber * mean_material_surface_height) / np.cosh(kd)
        sh3 = np.sinh(3 * kd + 3 * wavenumber * mean_material_surface_height) / np.cosh(
            3 * kd
        )

        a = (1 - mu**2) * (-9 + 6 * mu**2) / 32 / mu**5
        b = (27 + 69 * mu**2 + 9 * mu**6 - 33 * mu**4) / mu**7 / 64
        return steepness**3 * (a * sh1 + b * sh3) / 3


def dimensionless_material_surface_amplitude_fourth_harmonic(
    steepness, wavenumber, depth, mean_material_surface_height, **kwargs
):
    order = kwargs.get("order", _DEFAULT_ORDER)
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )
    kd = wavenumber * depth
    mu = np.tanh(kd)

    if order < 4:
        return np.zeros_like(steepness * wavenumber)

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        raise NotImplementedError("Fourth harmonic not implemented for shallow water")

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth harmonic not implemented for deep water waves"
        )

    else:
        # See 3.3, 3.4 in Zhao and Liu (2022)
        sh2 = np.sinh(2 * kd + 2 * wavenumber * mean_material_surface_height) / np.cosh(
            2 * kd
        )
        sh4 = np.sinh(4 * kd + 4 * wavenumber * mean_material_surface_height) / np.cosh(
            4 * kd
        )

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

        return steepness**4 * (a * sh2 + b * sh4)



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
        third_order =  (3 + 8 * mu**2 - 9 * mu**4) / 16 / mu**4

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
            (
                121 * alpha**5
                + 263 * alpha**4
                + 376 * alpha**3
                - 1999 * alpha**2
                + 2509 * alpha
                - 1108
            )
            / (192 * (alpha - 1) ** 5)
        )
    return fifth_order


def a22(
    relative_depth, **kwargs
):
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
        second_order =  ((3 - mu**2) / 4 / mu**3)

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
            (129 * mu**8 - 826 * mu**6 + 1152 * mu**4 - 54 * mu**2 - 81)
            / (384 * mu**9)
        )

    return fourth_order


def a33(
    relative_depth, **kwargs
):
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
        third_order = (
            (27 - 9 * mu**2 + 9 * mu**4 - 3 * mu**6)
            / 64
            / mu**6
        )
    return third_order

def a53(
            relative_depth, **kwargs
    ):
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


def eta11(
    relative_depth,relative_height, **kwargs
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

    mu = np.tanh(relative_depth)
    sh1 = sh(relative_depth, relative_height,1)
    return 1/mu * sh1

def eta31(
    relative_depth, relative_height, **kwargs
    ):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh1 = sh(relative_depth, relative_height, 1)
    sh3 = sh(relative_depth, relative_height, 3)

    if wave_regime == "shallow":
        a = -21 / 32 / mu**5
        b = 9  / 32 / mu**5
        amplitude = a * sh1 + b * sh3
    elif wave_regime == "deep":
        # See 3.10 in Zhao and Liu (2022)

        amplitude =  - 1 / 2 * sh1 + 0.625 * sh3
    else:

        a = +(-21 + 19 * mu**2 - 14 * mu**4) / 32 / mu**5
        b = (9 + 23 * mu**2 - 12 * mu**4) / 32 / mu**5
        amplitude = a * sh1 + b * sh3
    return amplitude

def eta22(
    relative_depth, relative_height, **kwargs
    ):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2)

    if wave_regime == "shallow":

        raise NotImplementedError("Second harmonic not implemented for shallow water")

    elif wave_regime == "deep":
        second_order = 0.5 * sh2

    else:
        second_order = (1 + mu**2) * (3 - mu**2) / 8 / mu**4 * sh2
    return second_order

def eta42(
    relative_depth, relative_height, **kwargs
    ):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)
    sh2 = sh(relative_depth, relative_height, 2)
    sh4 = sh(relative_depth, relative_height, 4)

    if wave_regime == "shallow":
        fourth_order =  - 27/ (256 * mu**10) * sh2

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
        fourth_order = (a * sh2 + b * sh4)

    return fourth_order


def eta33(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh1 = sh(relative_depth, relative_height, 1)
    sh3 = sh(relative_depth, relative_height, 3)

    if wave_regime == "shallow":

        amplitude =  9  / mu**7 / 64 * sh3

    elif wave_regime == "deep":
        amplitude = 3 / 8 * sh3

    else:

        a = (1 - mu**2) * (-9 + 6 * mu**2) / 32 / mu**5
        b = (27 + 69 * mu**2 + 9 * mu**6 - 33 * mu**4) / mu**7 / 64
        amplitude =  (a * sh1 + b * sh3) / 3

    return amplitude

def eta44(relative_depth, relative_height, **kwargs):
    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    sh2 = sh(relative_depth, relative_height, 2)
    sh4 = sh(relative_depth, relative_height, 4)

    if wave_regime == "shallow":
        amplitude = 81 / mu**10/ 1536 * sh4

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

        amplitude =  (a * sh2 + b * sh4)
    return amplitude
