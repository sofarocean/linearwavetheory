import numpy as np
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from .utils import get_wave_regime
from .vertical_eigen_functions import ch, sh


def dimensionless_horizontal_velocity_amplitude(
    steepness, relative_depth, relative_height, n, **kwargs
):
    """
    This function calculates the dimensionless horizontal velocity amplitude for a given harmonic of a Stokes wave at
    a given relative depth and height. The calculation is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (dimensionless), typically defined as wave amplitude times wavenumber
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_height: relative height in the water column (dimensionless), typically kz where z is the height in
        the water column
    :param n: the harmonic number (1, 2, 3, 4, or 5) for which to calculate the velocity amplitude
    :param kwargs:
    :return: dimensionless horizontal velocity amplitude for the specified harmonic
    """
    return dimensionless_velocity_amplitude(
        steepness, relative_depth, n, **kwargs
    ) * ch(relative_depth, relative_height, n, **kwargs)


def dimensionless_vertical_velocity_amplitude(
    steepness, relative_depth, relative_height, n, **kwargs
):
    """
    This function calculates the dimensionless vertical velocity amplitude for a given harmonic of a Stokes wave at
    a given relative depth and height. The calculation is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (dimensionless), typically defined as wave amplitude times wavenumber
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_height: relative height in the water column (dimensionless), typically kz where z is the height
        in the water column
    :param n: the harmonic number (1, 2, 3, 4, or 5) for which to calculate the velocity amplitude
    :param kwargs:
    :return: dimensionless vertical velocity amplitude for the specified harmonic
    """
    return dimensionless_velocity_amplitude(
        steepness, relative_depth, n, **kwargs
    ) * sh(relative_depth, relative_height, n, **kwargs)


def dimensionless_velocity_amplitude(
    steepness, relative_depth, harmonic_number, **kwargs
):
    """
    This function calculates the dimensionless velocity amplitude for a given harmonic of a Stokes wave. The
    calculation is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (dimensionless), typically defined as wave amplitude times wavenumber
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the
        water depth
    :param harmonic_number: the harmonic number (1, 2, 3, 4, or 5) for which to calculate the velocity amplitude
    :param kwargs:
    :return: dimensionless velocity amplitude for the specified harmonic
    """
    order = kwargs.get("order", _DEFAULT_ORDER)
    if harmonic_number == 1:
        _phi11 = phi11(relative_depth, **kwargs)
        amplitude = steepness * _phi11

    elif harmonic_number == 2:
        _phi22 = 0 if order < 2 else phi22(relative_depth, **kwargs)
        _phi42 = 0 if order < 4 else phi42(relative_depth, **kwargs)
        amplitude = steepness**2 * _phi22 + steepness**4 * _phi42

    elif harmonic_number == 3:
        _phi33 = 0 if order < 3 else phi33(relative_depth, **kwargs)
        _phi53 = 0 if order < 5 else phi53(relative_depth, **kwargs)
        amplitude = steepness**3 * _phi33 + steepness**5 * _phi53
    elif harmonic_number == 4:
        _phi44 = 0 if order < 4 else phi44(relative_depth, **kwargs)
        amplitude = steepness**4 * _phi44
    elif harmonic_number == 5:
        _phi55 = 0 if order < 5 else phi55(relative_depth, **kwargs)
        amplitude = steepness**5 * _phi55
    else:
        raise ValueError("Invalid order")

    return amplitude


def phi11(relative_depth, **kwargs):
    """
    This calculates the linear coeficient of the velicity potential for a Stokes wave. This is the first order
    approximation of the velocity amplitude.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return: dimensionless velocity amplitude
    """

    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        velocity_coefficient = 1 / relative_depth

    elif wave_regime == "deep":
        velocity_coefficient = 1

    else:
        velocity_coefficient = 1 / mu

    return velocity_coefficient


def phi22(relative_depth, **kwargs):
    """
    This function calculates the second order coefficient of the velocity potential for the second harmonic of a Stokes
    wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the second harmonic
    """

    mu = np.tanh(relative_depth)
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        velocity_coefficient = 0.75 / relative_depth**4

    elif wave_regime == "deep":
        velocity_coefficient = 0

    else:
        velocity_coefficient = 0.75 * (1 / mu**4 - 1)

    return velocity_coefficient


def phi42(relative_depth, **kwargs):
    """
    This function calculates the fourth order coefficient of the velocity potential for the second harmonic of a Stokes
    wave.

    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the second harmonic
    """

    mu = np.tanh(relative_depth)
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        return -81 / 384 / mu**10

    elif wave_regime == "deep":
        return 1.0

    else:
        velocity_coefficient42 = 2 * (
            (
                -81
                - 135 * mu**2
                + 810 * mu**4
                + 182 * mu**6
                - 537 * mu**8
                + 145 * mu**10
            )
            / (768 * mu**10)
        )
    return velocity_coefficient42


def phi33(relative_depth, **kwargs):
    """
    This function calculates the third order coefficient of the velocity potential for the third harmonic of a Stokes
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the third harmonic
    """

    wave_regime = get_wave_regime(**kwargs)
    mu = np.tanh(relative_depth)

    if wave_regime == "shallow":
        velocity_coefficient = 9 / relative_depth / 64 / mu**6

    elif wave_regime == "deep":
        velocity_coefficient = 0

    else:
        velocity_coefficient = (
            3 / 64 / mu * (9 / mu**6 + 5 / mu**4 + 39 - 53 / mu**2)
        )

    return velocity_coefficient


def phi53(relative_depth, **kwargs):
    """
    This function calculates the fifth order coefficient of the velocity potential for the third harmonic of a Stokes
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the third harmonic
    """
    mu = np.tanh(relative_depth)
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for deep water waves"
        )

    else:
        # Note this is the same form as the ZL2022 paper. We did not simplify the coeficients to the 5th order.
        # As a consequence it is ill-behaved in deep water.
        alpha = (1 + mu**2) / (1 - mu**2)
        A53 = (
            8 * alpha**6
            + 138 * alpha**5
            + 384 * alpha**4
            - 568 * alpha**3
            - 2388 * alpha**2
            + 237 * alpha
            + 974
        ) / (64 * (alpha - 1) ** 6 * (3 * alpha + 2) * np.sinh(relative_depth))
        velocity_coefficient53 = 3 * A53 * np.cosh(3 * relative_depth)

    return velocity_coefficient53


def phi44(relative_depth, **kwargs):
    """
    This function calculates the fourth order coefficient of the velocity potential for the fourth harmonic of a Stokes
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the fourth harmonic
    """

    mu = np.tanh(relative_depth)
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        return 0.0

    else:
        # Take derivative of potential gives factor 4
        velocity_coefficient = (
            4
            * (1 - mu**2)
            * (
                (
                    405
                    + 1161 * mu**2
                    - 6462 * mu**4
                    + 3410 * mu**6
                    + 1929 * mu**8
                    + 197 * mu**10
                )
                / (1536 * mu**10 * (5 + mu**2))
            )
        )

    return velocity_coefficient


def phi55(relative_depth, **kwargs):
    """
    This function calculates the fifth order coefficient of the velocity potential for the fifth harmonic of a Stokes
    wave.
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless velocity amplitude for the fifth harmonic
    """
    mu = np.tanh(relative_depth)
    wave_regime = get_wave_regime(**kwargs)

    if wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for shallow water waves"
        )

    elif wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for deep water waves"
        )

    else:
        # Note this is the same form as the ZL2022 paper. We did not simplify the coeficients to the 5th order.
        alpha = (1 + mu**2) / (1 - mu**2)
        A55 = (
            -6 * alpha**5
            + 272 * alpha**4
            - 1552 * alpha**3
            + 852 * alpha**2
            + 2029 * alpha
            + 430
        ) / (
            64
            * (alpha - 1) ** 6
            * (3 * alpha + 2)
            * (4 * alpha + 1)
            * np.sinh(relative_depth)
        )

        velocity_coefficient = 5 * A55 * np.cosh(5 * relative_depth)

    return velocity_coefficient
