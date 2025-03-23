import numpy as np
from linearwavetheory.stokes_theory.regular_waves.nonlinear_dispersion import (
    dimensionless_nonlinear_dispersion_relation,
)
from linearwavetheory.dispersion import intrinsic_dispersion_relation
from .settings import ReferenceFrame


def phase_function(
    steepness,
    wavenumber,
    depth,
    time,
    x,
    z=0,
    relative_phase_offset=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs
):
    """
    This function calculates the phase of a fourth order Stokes wave at a given time and position.
    :param steepness: wave steepness (dimensionless)
    :param wavenumber: wavenumber (rad/m)
    :param depth: depth (m)
    :param time: time (s)
    :param x: horizontal position (m)
    :param z: height in the water column (m), default is 0 (surface)
    :param relative_phase_offset: phase offset (dimensionless), default is 0 (no offset)
    :param kwargs:
    :return:
    """

    w0 = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )
    return dimensionless_phase_function(
        steepness,
        wavenumber * depth,
        w0 * time,
        wavenumber * x,
        height=wavenumber * z,
        relative_phase_offset=relative_phase_offset,
        reference_frame=reference_frame,
        **kwargs,
    )


def dimensionless_phase_function(
    steepness,
    relative_depth,
    relative_time,
    relative_x,
    relative_z=0,
    relative_phase_offset=0,
    reference_frame: ReferenceFrame = ReferenceFrame.eulerian,
    **kwargs
):
    """
    This function calculates the phase of a fourth order Stokes wave at a given time and position in a dimensionless
    form.
    :param steepness: wave steepness (dimensionless)
    :param relative_depth: relative depth (dimensionless), typically kd where k is the wavenumber and d is the water
        depth
    :param relative_time: relative time (dimensionless), typically omega*t where omega is the angular frequency
    :param relative_x: relative horizontal position (dimensionless), typically k*x where k is the wavenumber and x is
        the horizontal position
    :param relative_z: relative height in the water column (dimensionless), typically k*z where z is the height in the
        water column
    :param relative_phase_offset:
    :param kwargs:
    :return:
    """

    offset = relative_phase_offset * 2 * np.pi
    non_linear_angular_frequency = dimensionless_nonlinear_dispersion_relation(
        steepness, relative_depth, relative_z, reference_frame, **kwargs
    )

    phase = relative_x - non_linear_angular_frequency * relative_time + offset

    return phase
