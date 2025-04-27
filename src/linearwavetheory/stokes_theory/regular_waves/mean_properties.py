from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_surface_amplitude,
)
import linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes as eulerian_amplitudes
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lagrangian_amplitudes
from linearwavetheory.dispersion import intrinsic_dispersion_relation


def waveheight(steepness, wavenumber, depth, **kwargs):
    """
    This function calculates the wave height up to the fifth order. Default order is 4. Note that
    waveheight is invariant in reference frames for a regular wave and may thus be
    calculated from Eulerian amplitudes only.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    relative_depth = wavenumber * depth
    a1 = dimensionless_surface_amplitude(steepness, relative_depth, 1, **kwargs)
    a3 = dimensionless_surface_amplitude(steepness, relative_depth, 3, **kwargs)
    a5 = dimensionless_surface_amplitude(steepness, relative_depth, 5, **kwargs)
    return 2 * (a1 + a3 + a5) / wavenumber


def dimensionless_variance(steepness, relative_depth, relative_z=0, **kwargs):
    """
    Dimensionless variance of the material surface elevation for a fourth order Stokes wave. To obtain
    variance divide result by the wave number squared.

    Note that by default we calculate the variance assuming primary amplitude is Rayleigh
    distributed and phases are random. To obtain the variance for the deterministic amplitudes set stochastic=False.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_z: relative mean height in the water column (dimensionless), typically kz where z is the height in
        the water column
    :param kwargs:
    :return: dimensionless variance of the surface elevation (dimensionless)
    """

    order = kwargs.get("order", 4)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", True)

    surface_variance = steepness**2 / 2
    second_order = (
        surface_variance
        * eulerian_amplitudes.eta11(relative_depth, relative_z, **kwargs) ** 2
    )
    if order <= 2:
        return second_order

    if stochastic:
        factor = 2
    else:
        factor = 1

    if reference_frame == "eulerian":
        fourth_order = (
            2
            * factor
            * surface_variance**2
            * (
                eulerian_amplitudes.eta22(relative_depth, relative_z, **kwargs) ** 2
                + 2
                * eulerian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
                * eulerian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
            )
        )

    elif reference_frame == "lagrangian":
        fourth_order = (
            2
            * surface_variance**2
            * (
                factor
                * lagrangian_amplitudes.eta22(relative_depth, relative_z, **kwargs) ** 2
                + factor
                * 2
                * lagrangian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
                * lagrangian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
                + 2
                * (factor - 1)
                * lagrangian_amplitudes.eta20(relative_depth, relative_z, **kwargs) ** 2
            )
        )

    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")

    return second_order + fourth_order


def dimensionless_skewness(steepness, relative_depth, relative_z, **kwargs):
    """
    Dimensionless skewness of the material surface elevation for a fourth order Stokes wave. To obtain
    skewness divide result by the wave number cubed.

    Note that by default we calculate the skewness assuming primary amplitude is Rayleigh distributed and phases are
    random. To obtain the skewness for the deterministic amplitudes set stochastic=False.

    :param steepness:  wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column
    :param kwargs:
    :return: dimensionless skewness of the surface elevation (dimensionless)
    """
    order = kwargs.get("order", 6)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", False)

    if stochastic:
        # Raw moments from the rayleigh distribution. <epsilon^n>=epsilon^n Gamma(1+n/2)
        fo_ray_coef = 2  # Fourth order raw rayleigh moment coefficient
        so_ray_coef = 6  # Sixth order raw rayleigh moment coefficient
    else:
        # Amplitudes deterministic. <epsilon^n>=epsilon^n
        fo_ray_coef = 1
        so_ray_coef = 1

    if reference_frame == "eulerian":
        eta11 = eulerian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
        eta22 = eulerian_amplitudes.eta22(relative_depth, relative_z, **kwargs)
        eta20 = 0
        eta31 = eulerian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
        eta33 = eulerian_amplitudes.eta33(relative_depth, relative_z, **kwargs)
        eta42 = eulerian_amplitudes.eta42(relative_depth, relative_z, **kwargs)
        eta40 = 0

    elif reference_frame == "lagrangian":
        eta11 = lagrangian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
        eta22 = lagrangian_amplitudes.eta22(relative_depth, relative_z, **kwargs)
        eta20 = lagrangian_amplitudes.eta20(relative_depth, relative_z, **kwargs)
        eta31 = lagrangian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
        eta33 = lagrangian_amplitudes.eta33(relative_depth, relative_z, **kwargs)
        eta42 = lagrangian_amplitudes.eta42(relative_depth, relative_z, **kwargs)
        eta40 = lagrangian_amplitudes.eta40(relative_depth, relative_z, **kwargs)
    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")

    if order < 4:
        fourth_order = 0
    else:
        fourth_order = (
            3 / 2 * eta11 * (fo_ray_coef / 2 * eta22 + (fo_ray_coef - 1) * eta20)
        )

    if order < 6:
        sixth_order = 0
    else:
        sixth_order = (
            +(2 + so_ray_coef - 3 * fo_ray_coef) * eta20**3
            + 3 * (so_ray_coef - fo_ray_coef) * eta31 * eta11 * eta20
            + 3 / 2 * (so_ray_coef - fo_ray_coef) * eta22**2 * eta20
            + 3 / 2 * so_ray_coef * (eta33 + eta31) * eta11 * eta22
            + 3 / 2 * (so_ray_coef - fo_ray_coef) * eta40 * eta11**2
            + 3 / 4 * so_ray_coef * eta42 * eta11**2
        )
    return fourth_order * steepness**4 + sixth_order * steepness**6


def dimensionless_kurtosis(steepness, relative_depth, relative_z, **kwargs):
    """
    Dimensionless kurtosis of the material surface elevation for a fourth order Stokes wave. To obtain
    kurtosis divide result by the wave number to the power of four.

    Note that by default we calculate the kurtosis assuming primary amplitude is Rayleigh distributed and phases are
    random. To obtain the kurtosis for the deterministic amplitudes set stochastic=False.

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column
    :param kwargs:
    :return: dimensionless kurtosis of the surface elevation (dimensionless)
    """
    order = kwargs.get("order", 6)
    reference_frame = kwargs.get("reference_frame", "eulerian")
    stochastic = kwargs.get("stochastic", False)

    if stochastic:
        factor_fo = 2
        factor_so = 6
    else:
        factor_fo = 1
        factor_so = 1

    if reference_frame == "eulerian":
        eta11 = eulerian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
        eta22 = eulerian_amplitudes.eta22(relative_depth, relative_z, **kwargs)
        eta20 = 0
        eta31 = eulerian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
        eta33 = eulerian_amplitudes.eta33(relative_depth, relative_z, **kwargs)

    elif reference_frame == "lagrangian":
        eta11 = lagrangian_amplitudes.eta11(relative_depth, relative_z, **kwargs)
        eta22 = lagrangian_amplitudes.eta22(relative_depth, relative_z, **kwargs)
        eta20 = lagrangian_amplitudes.eta20(relative_depth, relative_z, **kwargs)
        eta31 = lagrangian_amplitudes.eta31(relative_depth, relative_z, **kwargs)
        eta33 = lagrangian_amplitudes.eta33(relative_depth, relative_z, **kwargs)

    else:
        raise ValueError(f"Invalid reference frame: {reference_frame}")

    if order < 4:
        fourth_order = 0
    else:
        fourth_order = 3 * factor_fo * eta11**4 / 8

    if order < 6:
        sixth_order = 0
    else:
        sixth_order = (
            +3 / 2 * factor_so * eta11**2 * eta22**2
            + 3 * factor_so * eta11**2 * eta20**2
            + 3 / 2 * eta11**3 * eta31 * factor_so
            + 1 / 2 * eta11**3 * eta33 * factor_so
            + 3 * eta20**2 * eta11**2
            - 6 * eta20**2 * eta11**2 * factor_fo
            - 3 * eta20 * eta22 * eta11**2 * factor_fo
        )

    return fourth_order * steepness**4 + sixth_order * steepness**6


def dimensionless_crest_height(steepness, relative_depth, **kwargs):
    """
    Dimensionless crest height of the surface elevation for a fourth order Stokes wave. To obtain
    crest height divide result by the wave number.

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless crest height of the surface elevation (dimensionless)
    """

    crest_height = 0
    for harmonic_number in range(1, 6):
        crest_height += dimensionless_surface_amplitude(
            steepness, relative_depth, harmonic_number, **kwargs
        )

    return crest_height


def dimensionless_trough_height(steepness, relative_depth, **kwargs):
    """
    Dimensionless trough height of the surface elevation for a fourth order Stokes wave. To obtain
    trough height divide result by the wave number.

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param kwargs:
    :return: dimensionless trough height of the surface elevation (dimensionless)
    """
    trough_height = 0
    sign = 1
    for harmonic_number in range(1, 6):
        sign *= -1

        trough_height += sign * dimensionless_surface_amplitude(
            steepness, relative_depth, harmonic_number, **kwargs
        )
    return trough_height


def dimensionless_stokes_drift(steepness, relative_depth, relative_z=0, **kwargs):
    """
    Dimensionless Stokes drift for a fourth order Stokes wave. To get the actual Stokes drift, multiply the result by
    the linear phase speed c (calculated using the linear dispersion relation).

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column
    :param kwargs:
    :return: stokes drift at desired height (dimensionless)
    """
    order = kwargs.get("order", _DEFAULT_ORDER)
    us2 = (
        0
        if order < 2
        else lagrangian_amplitudes.ul20(relative_depth, relative_z, **kwargs)
    )
    us4 = (
        0
        if order < 4
        else lagrangian_amplitudes.ul40(relative_depth, relative_z, **kwargs)
    )

    return steepness**2 * us2 + steepness**4 * us4


def dimensionless_lagrangian_mean_location(
    steepness, relative_depth, relative_time, relative_x, relative_z, **kwargs
):
    """
    This function calculates the dimensionless mean location of a Lagrangian particle in a fourth order Stokes wave.

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless) at t=0, typically k*x where k is the wavenumber and
        x is the horizontal position
    :param relative_z: relative mean height in the water column of the material surface the particle belongs to
        (dimensionless), typically kz where z is the height in the water column
    :param kwargs:
    :return: mean location of a Lagrangian particle in a fourth order Stokes wave (dimensionless)
    """

    us = dimensionless_stokes_drift(steepness, relative_depth, relative_z, **kwargs)

    return relative_x + us * relative_time


def dimensionless_lagrangian_setup(steepness, relative_depth, relative_z, **kwargs):
    """
    This function calculates the dimensionless setup of a Lagrangian particle in a fourth order Stokes wave.
    To get the actual setup, divide the result by the wave number.

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_z: relative height in the water column (dimensionless), typically kz where z is the height in the
        water column
    :param kwargs:
    :return: dimensionless setup of a Lagrangian particle in a fourth order Stokes wave (dimensionless)
    """
    order = kwargs.get("order", _DEFAULT_ORDER)
    eta20 = (
        0
        if order < 2
        else lagrangian_amplitudes.eta20(relative_depth, relative_z, **kwargs)
    )
    eta40 = (
        0
        if order < 4
        else lagrangian_amplitudes.eta40(relative_depth, relative_z, **kwargs)
    )
    return eta20 * steepness**2 + eta40 * steepness**4


def lagrangian_setup(steepness, wavenumber, depth, z=0, **kwargs):
    """
    This function calculates the setup of a Lagrangian particle in a fourth order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param z: height in the water column (m), default is 0 (surface)
    :param kwargs:
    :return:
    """
    relative_depth = wavenumber * depth
    relative_z = wavenumber * z
    return (
        dimensionless_lagrangian_setup(steepness, relative_depth, relative_z, **kwargs)
        / wavenumber
    )


def stokes_drift(steepness, wavenumber, depth, z, **kwargs):
    """
    This function calculates the Stokes drift for a fourth order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param kwargs:
    :return: Stokes drift (m/s)
    """
    linear_angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    c = linear_angular_frequency / wavenumber

    return (
        dimensionless_stokes_drift(
            steepness, wavenumber * depth, wavenumber * z, **kwargs
        )
        * c
    )
