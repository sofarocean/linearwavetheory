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

import linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes as eulerian_amplitudes
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lagrangian_amplitudes
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


def dimensionless_variance(steepness, relative_depth, relative_height, **kwargs):
    order = kwargs.get("order", 4)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", False)

    surface_variance = steepness**2 / 2
    second_order = surface_variance *eulerian_amplitudes.eta11(relative_depth, relative_height, **kwargs)**2
    if order <= 2:
        return second_order

    if stochastic:
        factor = 2
    else:
        factor = 1

    if reference_frame == "eulerian":
        fourth_order = 2 * factor * surface_variance ** 2 * (
            eulerian_amplitudes.eta22(relative_depth, relative_height, **kwargs)**2
            + 2 * eulerian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
            * eulerian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
        )

    elif reference_frame == "lagrangian":
        fourth_order = 2 * surface_variance ** 2 * (
                factor * lagrangian_amplitudes.eta22(relative_depth, relative_height, **kwargs) ** 2
                + factor * 2 * lagrangian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
                * lagrangian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
                + 2 * (factor - 1) * lagrangian_amplitudes.eta20(relative_depth, relative_height, **kwargs)**2
        )

    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")

    return second_order + fourth_order


def dimensionless_skewness(steepness, relative_depth, relative_height, **kwargs):
    order = kwargs.get("order", 6)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", False)

    if stochastic:
        # Raw moments from the rayleigh distribution. <epsilon^n>=epsilon^n Gamma(1+n/2)
        fo_ray_coef = 2 # Fourth order raw rayleigh moment coefficient
        so_ray_coef = 6 # Sixth order raw rayleigh moment coefficient
    else:
        # Amplitudes deterministic. <epsilon^n>=epsilon^n
        fo_ray_coef = 1
        so_ray_coef = 1

    if reference_frame == "eulerian":
        eta11 = eulerian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
        eta22 = eulerian_amplitudes.eta22(relative_depth, relative_height, **kwargs)
        eta20 = 0
        eta31 = eulerian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
        eta33 = eulerian_amplitudes.eta33(relative_depth, relative_height, **kwargs)
        eta42 = eulerian_amplitudes.eta42(relative_depth, relative_height, **kwargs)
        eta40 = 0

    elif reference_frame == "lagrangian":
        eta11 = lagrangian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
        eta22 = lagrangian_amplitudes.eta22(relative_depth, relative_height, **kwargs)
        eta20 = lagrangian_amplitudes.eta20(relative_depth, relative_height, **kwargs)
        eta31 = lagrangian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
        eta33 = lagrangian_amplitudes.eta33(relative_depth, relative_height, **kwargs)
        eta42 = lagrangian_amplitudes.eta42(relative_depth, relative_height, **kwargs)
        eta40 = lagrangian_amplitudes.eta40(relative_depth, relative_height, **kwargs)
    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")


    if order < 4:
        fourth_order = 0
    else:
        fourth_order = 3 / 2 * eta11 * (fo_ray_coef / 2 * eta22 + (fo_ray_coef - 1) * eta20)

    if order < 6:
        sixth_order = 0
    else:
        sixth_order = (
            + ( 2 +so_ray_coef - 3 * fo_ray_coef ) * eta20**3
            + 3*(so_ray_coef-fo_ray_coef)*eta31*eta11*eta20
            + 3/2 * (so_ray_coef - fo_ray_coef) * eta22**2 * eta20
            + 3/2 * so_ray_coef * (eta33+eta31) * eta11 * eta22
            + 3/2 * ( so_ray_coef - fo_ray_coef)*eta40 * eta11**2
            + 3/4 * so_ray_coef * eta42 * eta11 ** 2
        )
    return fourth_order * steepness**4 + sixth_order * steepness**6


def dimensionless_kurtosis(steepness, relative_depth, relative_height, **kwargs):
    order = kwargs.get("order", 6)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", False)

    if stochastic:
        factor_fo = 2
        factor_so = 6
        factor_eo = 6
    else:
        factor_fo = 1
        factor_so = 1
        factor_eo = 1

    if reference_frame == "eulerian":
        eta11 = eulerian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
        eta22 = eulerian_amplitudes.eta22(relative_depth, relative_height, **kwargs)
        eta20 = 0
        eta31 = eulerian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
        eta33 = eulerian_amplitudes.eta33(relative_depth, relative_height, **kwargs)

    elif reference_frame == "lagrangian":
        eta11 = lagrangian_amplitudes.eta11(relative_depth, relative_height, **kwargs)
        eta22 = lagrangian_amplitudes.eta22(relative_depth, relative_height, **kwargs)
        eta20 = lagrangian_amplitudes.eta20(relative_depth, relative_height, **kwargs)
        eta31 = lagrangian_amplitudes.eta31(relative_depth, relative_height, **kwargs)
        eta33 = lagrangian_amplitudes.eta33(relative_depth, relative_height, **kwargs)

    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")


    if order < 4:
        fourth_order = 0
    else:
        fourth_order = 3 *factor_fo * eta11**4 / 4

    if order < 6:
        sixth_order = 0
    else:
        sixth_order = (
            + 3/2 * factor_so * eta11**2 * eta22**2
            + 3 * factor_so * eta11**2 * eta20**2
            + 3/2 * eta11**3 * eta31 * factor_so
            + 1/2 * eta11**3 * eta33 * factor_so
            + 3 * eta20**2 * eta11**2
            - 6 * eta20**2 * eta11**2 * alpha
            - 3 * eta20 *eta22 * eta11 ** 2 * alpha
        )

    return fourth_order * steepness**4 + sixth_order * steepness**6



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
