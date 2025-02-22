import numpy as np

from linearwavetheory.settings import _parse_options
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER


def dimensionless_horizontal_velocity_amplitude(
    steepness, relative_depth, relative_height, order, **kwargs
):
    return (
        dimensionless_velocity_amplitude(steepness, relative_depth, order, **kwargs)
        * np.cosh(order * (relative_height + relative_depth))
        / np.cosh(order * relative_depth)
    )


def dimensionless_vertical_velocity_amplitude(
    steepness, relative_depth, relative_height, order, **kwargs
):
    return (
        dimensionless_velocity_amplitude(steepness, relative_depth, order, **kwargs)
        * np.sinh(order * (relative_height + relative_depth))
        / np.cosh(order * relative_depth)
    )


def dimensionless_velocity_amplitude(steepness, relative_depth, harmonic, **kwargs):

    if harmonic == 1:
        amplitude = dimensionless_velocity_amplitude_first_harmonic(
            steepness, relative_depth, **kwargs
        )
    elif harmonic == 2:
        amplitude = dimensionless_velocity_amplitude_second_harmonic(
            steepness, relative_depth, **kwargs
        )
    elif harmonic == 3:
        amplitude = dimensionless_velocity_amplitude_third_harmonic(
            steepness, relative_depth, **kwargs
        )
    elif harmonic == 4:
        amplitude = dimensionless_velocity_amplitude_fourth_harmonic(
            steepness, relative_depth, **kwargs
        )
    elif harmonic == 5:
        amplitude = dimensionless_velocity_amplitude_fifth_harmonic(
            steepness, relative_depth, **kwargs
        )
    else:
        raise ValueError("Invalid order")

    return amplitude


def dimensionless_velocity_amplitude_first_harmonic(
    steepness, relative_depth, **kwargs
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

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(relative_depth)

    if physics_options.wave_regime == "shallow":
        velocity_coefficient = 1 / relative_depth

    elif physics_options.wave_regime == "deep":
        velocity_coefficient = 1

    else:
        velocity_coefficient = 1 / mu

    return steepness * velocity_coefficient


def dimensionless_velocity_amplitude_second_harmonic(
    steepness, relative_depth, **kwargs
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

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    order = kwargs.get("order", _DEFAULT_ORDER)
    if order < 2:
        return np.zeros_like(steepness * relative_depth)

    relative_depth = relative_depth
    mu = np.tanh(relative_depth)

    if physics_options.wave_regime == "shallow":
        velocity_coefficient = 0.75 / relative_depth**4

    elif physics_options.wave_regime == "deep":
        velocity_coefficient = 0

    else:
        velocity_coefficient = 0.75 * (1 / mu**4 - 1)

    velocity_coefficient = steepness**2 * velocity_coefficient

    if order < 4:
        return velocity_coefficient

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for deep water waves"
        )

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
    return velocity_coefficient + steepness**4 * velocity_coefficient42


def dimensionless_velocity_amplitude_third_harmonic(
    steepness, relative_depth, **kwargs
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

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(relative_depth)

    order = kwargs.get("order", _DEFAULT_ORDER)
    if order < 3:
        return np.zeros_like(steepness * relative_depth)

    if physics_options.wave_regime == "shallow":
        velocity_coefficient = 9 / relative_depth / 64 / mu**6

    elif physics_options.wave_regime == "deep":
        velocity_coefficient = 0

    else:
        velocity_coefficient = (
            3 / 64 / mu * (9 / mu**6 + 5 / mu**4 + 39 - 53 / mu**2)
        )

    velocity_coefficient = steepness**3 * velocity_coefficient
    if order < 5:
        return velocity_coefficient

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for deep water waves"
        )

    else:
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

    return velocity_coefficient + steepness**5 * velocity_coefficient53


def dimensionless_velocity_amplitude_fourth_harmonic(
    steepness, relative_depth, **kwargs
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

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(relative_depth)

    order = kwargs.get("order", _DEFAULT_ORDER)
    if order < 4:
        return np.zeros_like(steepness * relative_depth)

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order velocity amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        return np.zeros_like(steepness * relative_depth)

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

    return steepness**4 * velocity_coefficient


def dimensionless_velocity_amplitude_fifth_harmonic(
    steepness, relative_depth, **kwargs
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

    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )

    mu = np.tanh(relative_depth)

    order = kwargs.get("order", _DEFAULT_ORDER)
    if order < 5:
        return np.zeros_like(steepness * relative_depth)

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fifth order velocity amplitude not implemented for deep water waves"
        )

    else:
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

    return steepness**5 * velocity_coefficient
