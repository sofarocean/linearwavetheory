import numpy as np

from linearwavetheory import intrinsic_dispersion_relation
from linearwavetheory.stokes_theory.regular_waves.eularian_velocity_amplitudes import (
    dimensionless_velocity_amplitude_first_harmonic,
    dimensionless_velocity_amplitude_second_harmonic,
    dimensionless_velocity_amplitude_third_harmonic,
    dimensionless_velocity_amplitude_fourth_harmonic,
    dimensionless_velocity_amplitude_fifth_harmonic,
    dimensionless_horizontal_velocity_amplitude,
    dimensionless_vertical_velocity_amplitude,
)

from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.settings import _parse_options, stokes_theory_options
from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_surface_amplitude_first_harmonic,
    dimensionless_surface_amplitude_second_harmonic,
    dimensionless_surface_amplitude_third_harmonic,
    dimensionless_surface_amplitude_fourth_harmonic,
    dimensionless_surface_amplitude_fifth_harmonic,
    dimensionless_material_surface_amplitude_first_harmonic,
    dimensionless_material_surface_amplitude_second_harmonic,
    dimensionless_material_surface_amplitude_third_harmonic,
    dimensionless_material_surface_amplitude_fourth_harmonic,
)
from linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes import (
    lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic,
    lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic,
    lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic,
    lagrangian_dimensionless_horizontal_displacement_first_harmonic,
    lagrangian_dimensionless_horizontal_displacement_second_harmonic,
    lagrangian_dimensionless_horizontal_displacement_third_harmonic,
)
from linearwavetheory.stokes_theory.regular_waves.mean_properties import (
    dimensionless_lagrangian_setup,
    dimensionless_lagrangian_mean_location,
)
from linearwavetheory.stokes_theory.regular_waves.phase import (
    phase_function,
    dimensionless_phase_function,
)


def free_surface_elevation(
    steepness, wavenumber, depth, time, xcoordinate, relative_phase_offset=0, **kwargs
):
    _, _, nonlinear_options = _parse_options(
        None, None, kwargs.get("nonlinear_options", None)
    )

    # In a Lagrangian reference frame we need to take into account the Stokes drift.
    if nonlinear_options.reference_frame == "eulerian":
        elevation = dimensionless_eulerian_free_surface_elevation
    elif nonlinear_options.reference_frame == "lagrangian":
        elevation = dimensionless_eulerian_free_surface_elevation
    else:
        raise ValueError("Invalid reference frame")

    return (
        elevation(
            steepness,
            wavenumber,
            depth,
            time,
            xcoordinate,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def dimensionless_lagrangian_free_surface_elevation(
    steepness, wavenumber, depth, time, xcoordinate, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """
    height = 0
    return vertical_particle_displacement(
        steepness,
        wavenumber,
        depth,
        time,
        xcoordinate,
        height,
        relative_phase_offset,
        **kwargs
    )


def dimensionless_eulerian_free_surface_elevation(
    steepness, wavenumber, depth, time, xcoordinate, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    phase = phase_function(
        steepness, wavenumber, depth, time, xcoordinate, relative_phase_offset, **kwargs
    )

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

    return (
        a1 * np.cos(phase)
        + a2 * np.cos(2 * phase)
        + a3 * np.cos(3 * phase)
        + a4 * np.cos(4 * phase)
        + a5 * np.cos(5 * phase)
    )


def material_surface_vertical_elevation(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    _, _, nonlinear_options = _parse_options(
        None, None, kwargs.get("nonlinear_options", None)
    )

    if nonlinear_options.reference_frame == "eulerian":
        elevation = eulerian_dimensionless_material_surface_vertical_displacement
    elif nonlinear_options.reference_frame == "lagrangian":
        raise ValueError(
            "Material surface elevation not implemented in Lagrangian reference frame"
        )
    else:
        raise ValueError("Invalid reference frame")

    return (
        elevation(
            steepness,
            wavenumber,
            depth,
            time,
            xcoordinate,
            height,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
        + height
    )


def eulerian_dimensionless_material_surface_vertical_displacement(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    phase = phase_function(
        steepness, wavenumber, depth, time, xcoordinate, relative_phase_offset, **kwargs
    )

    a1 = dimensionless_material_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, height, **kwargs
    )
    a2 = dimensionless_material_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, height, **kwargs
    )
    a3 = dimensionless_material_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, height, **kwargs
    )
    a4 = dimensionless_material_surface_amplitude_fourth_harmonic(
        steepness, wavenumber, depth, height, **kwargs
    )
    return (
        a1 * np.cos(phase)
        + a2 * np.cos(2 * phase)
        + a3 * np.cos(3 * phase)
        + a4 * np.cos(4 * phase)
    )


def vertical_particle_displacement(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    # For convinience set the default to lagrangian reference frame - otherwise raise an error
    _nonlinear_options = kwargs.get("nonlinear_options", None)
    if _nonlinear_options is None:
        _nonlinear_options = stokes_theory_options(reference_frame="lagrangian")
    else:
        if not _nonlinear_options.reference_frame == "lagrangian":
            raise ValueError("Invalid reference frame")
    kwargs["nonlinear_options"] = _nonlinear_options

    phase = phase_function(
        steepness,
        wavenumber,
        depth,
        time,
        xcoordinate,
        height,
        relative_phase_offset,
        **kwargs
    )

    scaling_factor = 1 / wavenumber
    a1 = (
        scaling_factor
        * lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )
    a2 = (
        scaling_factor
        * lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )
    a3 = (
        scaling_factor
        * lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )

    return a1 * np.cos(phase) + a2 * np.cos(2 * phase) + a3 * np.cos(3 * phase)


def vertical_particle_location(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    # For convinience set the default to lagrangian reference frame - otherwise raise an error
    _nonlinear_options = kwargs.get("nonlinear_options", None)
    if _nonlinear_options is None:
        _nonlinear_options = stokes_theory_options(reference_frame="lagrangian")
    else:
        if not _nonlinear_options.reference_frame == "lagrangian":
            raise ValueError("Invalid reference frame")
    kwargs["nonlinear_options"] = _nonlinear_options

    setup = (
        dimensionless_lagrangian_setup(
            steepness, wavenumber * depth, wavenumber * height, **kwargs
        )
        / wavenumber
    )

    return (
        vertical_particle_displacement(
            steepness,
            wavenumber,
            depth,
            time,
            xcoordinate,
            height,
            relative_phase_offset,
            **kwargs
        )
        + height
        + setup
    )


def horizontal_particle_displacement(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    # For convinience set the default to lagrangian reference frame - otherwise raise an error
    _nonlinear_options = kwargs.get("nonlinear_options", None)
    if _nonlinear_options is None:
        _nonlinear_options = stokes_theory_options(reference_frame="lagrangian")
    else:
        if not _nonlinear_options.reference_frame == "lagrangian":
            raise ValueError("Invalid reference frame")
    kwargs["nonlinear_options"] = _nonlinear_options

    phase = phase_function(
        steepness,
        wavenumber,
        depth,
        time,
        xcoordinate,
        height,
        relative_phase_offset,
        **kwargs
    )

    scaling_factor = 1 / wavenumber
    a1 = (
        scaling_factor
        * lagrangian_dimensionless_horizontal_displacement_first_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )
    a2 = (
        scaling_factor
        * lagrangian_dimensionless_horizontal_displacement_second_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )
    a3 = (
        scaling_factor
        * lagrangian_dimensionless_horizontal_displacement_third_harmonic(
            steepness, wavenumber, depth, height, **kwargs
        )
    )

    return a1 * np.sin(phase) + a2 * np.sin(2 * phase) + a3 * np.sin(3 * phase)


def horizontal_particle_location(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height,
    relative_phase_offset=0,
    **kwargs
):
    # For convinience set the default to lagrangian reference frame - otherwise raise an error
    _nonlinear_options = kwargs.get("nonlinear_options", None)
    if _nonlinear_options is None:
        _nonlinear_options = stokes_theory_options(reference_frame="lagrangian")
    else:
        if not _nonlinear_options.reference_frame == "lagrangian":
            raise ValueError("Invalid reference frame")
    kwargs["nonlinear_options"] = _nonlinear_options

    xmean = (
        dimensionless_lagrangian_mean_location(
            steepness, wavenumber, depth, time, xcoordinate, height, **kwargs
        )
        / wavenumber
    )

    return (
        horizontal_particle_displacement(
            steepness,
            wavenumber,
            depth,
            time,
            xcoordinate,
            height,
            relative_phase_offset,
            **kwargs
        )
        + xmean
    )


def dimensionless_velocity(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_height,
        relative_phase_offset,
        **kwargs
    )

    # Sum the harmonices to the desired order
    result = np.zeros_like(phase)
    for iorder in range(1, order + 1):
        harmonic = np.cos(iorder * phase)
        amplitude = dimensionless_horizontal_velocity_amplitude(
            steepness, relative_depth, relative_height, iorder, **kwargs
        )
        result += amplitude * harmonic

    return result


def horizontal_velocity(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    def ch(m):
        return np.cosh(m * wavenumber * (depth + z)) / np.cosh(m * wavenumber * depth)

    angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    phase = phase_function(
        steepness, wavenumber, depth, time, x, relative_phase_offset, **kwargs
    )

    c = angular_frequency / wavenumber
    relative_depth = wavenumber * depth
    a1 = (
        dimensionless_velocity_amplitude_first_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a2 = (
        dimensionless_velocity_amplitude_second_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a3 = (
        dimensionless_velocity_amplitude_third_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a4 = (
        dimensionless_velocity_amplitude_fourth_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a5 = (
        dimensionless_velocity_amplitude_fifth_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )

    return (
        a1 * np.cos(phase) * ch(1)
        + a2 * np.cos(2 * phase) * ch(2)
        + a3 * np.cos(3 * phase) * ch(3)
        + a4 * np.cos(4 * phase) * ch(4)
        + a5 * np.cos(5 * phase) * ch(5)
    )


def vertical_velocity(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the surface elevation of a third order Stokes wave.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth
    :param kwargs:
    :return:
    """

    def sh(m):
        return np.sinh(m * wavenumber * (depth + z)) / np.cosh(m * wavenumber * depth)

    angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    phase = phase_function(
        steepness, wavenumber, depth, time, x, relative_phase_offset, **kwargs
    )

    c = angular_frequency / wavenumber
    relative_depth = wavenumber * depth
    a1 = (
        dimensionless_velocity_amplitude_first_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a2 = (
        dimensionless_velocity_amplitude_second_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a3 = (
        dimensionless_velocity_amplitude_third_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a4 = (
        dimensionless_velocity_amplitude_fourth_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )
    a5 = (
        dimensionless_velocity_amplitude_fifth_harmonic(
            steepness, relative_depth, **kwargs
        )
        * c
    )

    return (
        a1 * np.sin(phase) * sh(1)
        + a2 * np.sin(2 * phase) * sh(2)
        + a3 * np.sin(3 * phase) * sh(3)
        + a4 * np.sin(4 * phase) * sh(4)
        + a5 * np.sin(5 * phase) * sh(5)
    )
