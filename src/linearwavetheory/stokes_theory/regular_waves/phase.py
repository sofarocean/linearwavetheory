import numpy as np

from linearwavetheory import intrinsic_dispersion_relation
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.stokes_theory.regular_waves.nonlinear_dispersion import (
    dimensionless_nonlinear_dispersion_relation,
    nonlinear_dispersion_relation,
)
from linearwavetheory.settings import _parse_options


def phase_function(
    steepness,
    wavenumber,
    depth,
    time,
    xcoordinate,
    height=0,
    relative_phase_offset=0,
    **kwargs
):

    offset = relative_phase_offset * 2 * np.pi
    non_linear_angular_frequency = nonlinear_dispersion_relation(
        steepness, wavenumber, depth, height, **kwargs
    )
    phase = wavenumber * xcoordinate - non_linear_angular_frequency * time + offset

    return phase


def dimensionless_phase_function(
    steepness,
    relative_depth,
    relative_time,
    relative_xcoordinate,
    relative_height=0,
    relative_phase_offset=0,
    **kwargs
):

    offset = relative_phase_offset * 2 * np.pi
    non_linear_angular_frequency = dimensionless_nonlinear_dispersion_relation(
        steepness, relative_depth, relative_height, **kwargs
    )

    phase = relative_xcoordinate - non_linear_angular_frequency * relative_time + offset

    return phase
