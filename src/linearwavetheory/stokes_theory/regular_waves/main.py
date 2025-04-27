"""

References
----------
Zhao, K., & Liu, P. L. F. (2022). On Stokes wave solutions. Proceedings of the Royal Society A, 478(2258), 20210732.
"""

import numpy as np

from linearwavetheory import intrinsic_dispersion_relation
from linearwavetheory.stokes_theory.regular_waves.eularian_velocity_amplitudes import (
    dimensionless_horizontal_velocity_amplitude,
    dimensionless_vertical_velocity_amplitude,
)
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_surface_amplitude,
    dimensionless_material_surface_amplitude,
)
from linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes import (
    dimensionless_vertical_displacement_amplitude,
    dimensionless_horizontal_displacement_amplitude,
)
from linearwavetheory.stokes_theory.regular_waves.mean_properties import (
    dimensionless_lagrangian_setup,
    dimensionless_lagrangian_mean_location,
)
from linearwavetheory.stokes_theory.regular_waves.phase import (
    dimensionless_phase_function,
)
from .settings import ReferenceFrame


def free_surface_elevation(
    steepness,
    wavenumber,
    depth,
    time,
    x,
    relative_phase_offset=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs
):
    """
    This function calculates the free surface elevation of a fourth order Stokes wave in a given reference frame.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param time: time (s)
    :param x: horizontal position (m). For the Eulerian case this is just the horizontal position we evalutate at.
        For the Lagrangian case this is the mean horizontal position of the particle at t=0.
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param reference_frame: either 'eulerian' or 'lagrangian', default is 'eulerian'.
    :param kwargs:
    :return: Free surface elevation (m) as a function of time in the given reference frame.
    """

    # In a Lagrangian reference frame we need to take into account the Stokes drift.
    if reference_frame == ReferenceFrame.eulerian:
        elevation = dimensionless_eulerian_free_surface_elevation
    elif reference_frame == ReferenceFrame.lagrangian:
        elevation = dimensionless_lagrangian_free_surface_elevation
    else:
        raise ValueError("Invalid reference frame")

    w0 = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    relative_depth = wavenumber * depth
    relative_time = w0 * time
    relative_x = wavenumber * x

    return (
        elevation(
            steepness,
            relative_depth,
            relative_time,
            relative_x,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def dimensionless_lagrangian_free_surface_elevation(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_phase_offset=0,
    **kwargs
):
    return dimensionless_vertical_particle_location(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        0,
        relative_phase_offset,
        **kwargs
    )


def dimensionless_eulerian_free_surface_elevation(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the surface elevation of a fifth order Stokes wave (default =4).

    :param steepness: steepness (wave amplitude times wavenumber)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: surface elevation (dimensionless)
    """

    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        0,
        relative_phase_offset,
        reference_frame=ReferenceFrame.eulerian,
        **kwargs
    )

    phase = np.atleast_1d(phase)
    out = np.zeros(len(phase))
    order = kwargs.get("order", _DEFAULT_ORDER)
    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_surface_amplitude(
            steepness, relative_depth, harmonic_number, **kwargs
        )
        out += amplitude * np.cos(harmonic_number * phase)

    return out


def material_surface_vertical_elevation(
    steepness,
    wavenumber,
    depth,
    time,
    x,
    z,
    relative_phase_offset=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs
):

    if reference_frame == ReferenceFrame.eulerian:
        location = eulerian_dimensionless_material_surface_vertical_location
    elif reference_frame == ReferenceFrame.lagrangian:
        location = dimensionless_vertical_particle_location
    else:
        raise ValueError("Invalid reference frame")

    relative_depth = wavenumber * depth
    relative_time = (
        intrinsic_dispersion_relation(
            wavenumber, depth, kwargs.get("physics_options", None)
        )
        * time
    )
    relative_x = wavenumber * x
    relative_z = wavenumber * z

    return (
        location(
            steepness,
            relative_depth,
            relative_time,
            relative_x,
            relative_z,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def eulerian_dimensionless_material_surface_vertical_displacement(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z,
    relative_phase_offset=0,
    **kwargs
):

    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_z,
        relative_phase_offset,
        reference_frame=ReferenceFrame.eulerian,
        **kwargs
    )

    phase = np.atleast_1d(phase)
    out = np.zeros(len(phase))
    order = kwargs.get("order", _DEFAULT_ORDER)
    if order > 4:
        raise ValueError("Order > 4 not implemented for material surface elevation")

    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_material_surface_amplitude(
            steepness, relative_depth, relative_z, harmonic_number, **kwargs
        )
        out += amplitude * np.cos(harmonic_number * phase)

    return out


def eulerian_dimensionless_material_surface_vertical_location(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z,
    relative_phase_offset=0,
    **kwargs
):

    displacement = eulerian_dimensionless_material_surface_vertical_displacement(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_z,
        relative_phase_offset,
        **kwargs
    )

    return displacement + relative_z


def vertical_particle_displacement(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the vertical particle displacement of a fourth order Stokes wave. The particle is
    identified with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the
    material surface the particle is located on. The function returns the vertical displacement of the particle at time
    t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth (m)
    :param time: (time in seconds)
    :param x: mean horizontal position (m) of the particle at t=0
    :param z: mean vertical position (m) of the material surface the partical is located on.
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle displacement (m)
    """

    w0 = np.squeeze(
        intrinsic_dispersion_relation(
            wavenumber, depth, kwargs.get("physics_options", None)
        )
    )
    return (
        dimensionless_vertical_particle_displacement(
            steepness,
            wavenumber * depth,
            w0 * time,
            wavenumber * x,
            wavenumber * z,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def dimensionless_vertical_particle_displacement(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the vertical particle displacement of a fourth order Stokes wave. The particle is
    identified with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the
    material surface the particle is located on. The function returns the vertical displacement of the particle at time
    t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x
        is the horizontal position
    :param relative_z: relative height in the water column (dimensionless), typically k*z where z is the height in the
        water column
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle displacement (dimensionless)
    """

    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_z,
        relative_phase_offset,
        reference_frame=ReferenceFrame.lagrangian,
        **kwargs
    )

    phase = np.atleast_1d(phase)
    out = np.zeros(len(phase))
    order = kwargs.get("order", _DEFAULT_ORDER)
    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_vertical_displacement_amplitude(
            steepness, relative_depth, relative_z, harmonic_number, **kwargs
        )
        out += amplitude * np.cos(harmonic_number * phase)

    return out


def vertical_particle_location(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the vertical particle location of a fourth order Stokes wave. The particle is identified
    with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the material
    surface the particle is located on. The function returns the vertical location of the particle at time t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth (m)
    :param time: (time in seconds)
    :param x: mean horizontal position (m) of the particle at t=0
    :param z: mean vertical position (m) of the material surface the partical is located on.
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle location (m) at time t
    """

    w0 = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    return (
        dimensionless_vertical_particle_location(
            steepness,
            wavenumber * depth,
            w0 * time,
            wavenumber * x,
            wavenumber * z,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def dimensionless_vertical_particle_location(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the vertical particle location of a fourth order Stokes wave. The particle is identified
    with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the material
    surface the particle is located on. The function returns the vertical location of the particle at time t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_z: relative height in the water column (dimensionless), typically k*z where z is the height in the
        water column
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle displacement (dimensionless)
    """

    setup = dimensionless_lagrangian_setup(
        steepness, relative_depth, relative_z, **kwargs
    )

    return (
        dimensionless_vertical_particle_displacement(
            steepness,
            relative_depth,
            relative_time,
            relative_x,
            relative_z,
            relative_phase_offset,
            **kwargs
        )
        + relative_z
        + setup
    )


def horizontal_particle_displacement(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the horizontal particle displacement of a fourth order Stokes wave. The particle is
    identified with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the
    material surface the particle is located on. The function returns the horizontal displacement of the particle at
    time t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth (m)
    :param time: (time in seconds)
    :param x: mean horizontal position (m) of the particle at t=0
    :param z: mean vertical position (m) of the material surface the partical is located on.
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle displacement (m)
    """

    w0 = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    return (
        dimensionless_horizontal_particle_displacement(
            steepness,
            wavenumber * depth,
            w0 * time,
            wavenumber * x,
            wavenumber * z,
            relative_phase_offset,
            **kwargs
        )
        / wavenumber
    )


def dimensionless_horizontal_particle_displacement(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the vertical particle displacement of a fourth order Stokes wave. The particle is
    identified with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the
    material surface the particle is located on. The function returns the vertical displacement of the particle at time
    t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_z: relative height in the water column (dimensionless), typically k*z where z is the height in the
        water column
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: vertical particle displacement (dimensionless)
    """

    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_z,
        relative_phase_offset,
        reference_frame=ReferenceFrame.lagrangian,
        **kwargs
    )

    phase = np.atleast_1d(phase)
    out = np.zeros(len(phase))
    order = kwargs.get("order", _DEFAULT_ORDER)

    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_horizontal_displacement_amplitude(
            steepness, relative_depth, relative_z, harmonic_number, **kwargs
        )
        out += amplitude * np.sin(harmonic_number * phase)

    return out


def horizontal_particle_location(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the horizontal particle location of a fourth order Stokes wave. The particle is identified
    with x,z where x is the mean horizontal position at t=0 and z is the mean vertical position of the material
    surface the particle is located on. The function returns the horizontal location of the particle at time t.

    :param steepness: steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber
    :param depth: depth (m)
    :param time: (time in seconds)
    :param x: mean horizontal position (m) of the particle at t=0
    :param z: mean vertical position (m) of the material surface the partical is located on.
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: horizontal particle location (m) at time t
    """

    w0 = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    xmean = (
        dimensionless_lagrangian_mean_location(
            steepness,
            wavenumber * depth,
            w0 * time,
            wavenumber * x,
            wavenumber * z,
            **kwargs
        )
        / wavenumber
    )

    return (
        horizontal_particle_displacement(
            steepness, wavenumber, depth, time, x, z, relative_phase_offset, **kwargs
        )
        + xmean
    )


def dimensionless_horizontal_velocity(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the dimensionless velocity in a Stokes waves in a Eulerian reference frame.
    Implementation is valid up to the 5-th order and is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_height: relative height in the water column (dimensionless), typically k*z where z is the height in
        the water column
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: dimensionless velocity
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_height,
        relative_phase_offset,
        reference_frame=ReferenceFrame.eulerian,
        **kwargs
    )

    # Sum the harmonices to the desired order
    phase = np.atleast_1d(phase)
    result = np.zeros_like(phase)
    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_horizontal_velocity_amplitude(
            steepness, relative_depth, relative_height, harmonic_number, **kwargs
        )
        result += amplitude * np.cos(harmonic_number * phase)

    return result


def horizontal_velocity(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the velocity in a Stokes waves in a Eulerian reference frame.
    Implementation is valid up to the 5-th order and is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param time: time (s)
    :param x: horizontal position (m)
    :param z: height in the water column (m)
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: velocity (m/s)
    """

    angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )

    c = angular_frequency / wavenumber
    return (
        dimensionless_horizontal_velocity(
            steepness,
            wavenumber * depth,
            angular_frequency * time,
            wavenumber * x,
            wavenumber * z,
            relative_phase_offset,
            **kwargs
        )
        * c
    )


def dimensionless_vertical_velocity(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_height,
    relative_phase_offset=0,
    **kwargs
):
    """
    This function calculates the dimensionless velocity in a Stokes waves in a Eulerian reference frame.
    Implementation is valid up to the 5-th order and is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_height: relative height in the water column (dimensionless), typically k*z where z is the height in
        the water column
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: dimensionless velocity
    """

    order = kwargs.get("order", _DEFAULT_ORDER)
    phase = dimensionless_phase_function(
        steepness,
        relative_depth,
        relative_time,
        relative_x,
        relative_height,
        relative_phase_offset,
        reference_frame=ReferenceFrame.eulerian,
        **kwargs
    )

    # Sum the harmonices to the desired order
    phase = np.atleast_1d(phase)
    result = np.zeros_like(phase)
    for harmonic_number in range(1, order + 1):
        amplitude = dimensionless_vertical_velocity_amplitude(
            steepness, relative_depth, relative_height, harmonic_number, **kwargs
        )
        result += amplitude * np.sin(harmonic_number * phase)

    return result


def vertical_velocity(
    steepness, wavenumber, depth, time, x, z, relative_phase_offset=0, **kwargs
):
    """
    This function calculates the velocity in a Stokes waves in a Eulerian reference frame.
    Implementation is valid up to the 5-th order and is based on the work of Zhao and Liu (2022).

    :param steepness: wave steepness (wave amplitude times wavenumber)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param time: time (s)
    :param x: horizontal position (m)
    :param z: height in the water column (m)
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return: velocity (m/s)
    """

    angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )

    c = angular_frequency / wavenumber
    return (
        dimensionless_vertical_velocity(
            steepness,
            wavenumber * depth,
            angular_frequency * time,
            wavenumber * x,
            wavenumber * z,
            relative_phase_offset,
            **kwargs
        )
        * c
    )
