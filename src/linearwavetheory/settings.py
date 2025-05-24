"""
This module contains classes for the numerical and physical options for the linear wave theory package. These options
are used to determine the numerical parameters for the linear wave theory package, as well as the physical parameters
for the linear wave theory package. The numerical parameters are used to determine the accuracy of the numerical
solution, while the physical parameters are used to determine the physical properties of the fluid and its environment.
"""
from numba import jit
from ._numba_settings import numba_default

from collections import namedtuple
import numpy as np

_GRAV = 9.80665
_SURFACE_TENSION = 0.074
_WATER_DENSITY = 1025.0
_KINEMATIC_SURFACE_TENSION = _SURFACE_TENSION / _WATER_DENSITY

# Default Numerical Parameters
_RELATIVE_TOLERANCE = 1e-4
_MAXIMUM_NUMBER_OF_ITERATIONS = 10
_ABSOLUTE_TOLERANCE = np.inf
_UNDER_RELAXATION_FACTOR = 0.8

_default_stokes_theory_options = {
    "reference_frame": "eulerian",
    "include_nonlinear_dispersion": True,
    "include_nonlinear_amplitude_correction": True,
    "include_bound_waves": True,
    "wave_driven_flow_included_in_mean_flow": True,
    "wave_driven_setup_included_in_mean_depth": True,
    "include_sum_interactions": True,
    "include_difference_interactions": True,
    "include_eulerian_contribution_in_drift": True,
    "angle_integration_convention": "wavenumber_aligned",
    "use_s_theory": False,
}

StokesTheoryOptions = namedtuple(
    "StokesTheoryOptions", _default_stokes_theory_options.keys()
)


_default_physics_options = {
    "kinematic_surface_tension": _KINEMATIC_SURFACE_TENSION,
    "grav": _GRAV,
    "wave_type": "gravity-capillary",
    "wave_regime": "intermediate",
    "density": _WATER_DENSITY,
}

PhysicsOptions = namedtuple(
    "PhysicsOptions", [*list(_default_physics_options.keys()), "wave_regime_enum"]
)


def stokes_theory_options(**kwargs) -> StokesTheoryOptions:
    for key in kwargs:
        if key not in _default_stokes_theory_options:
            raise ValueError(f"Unknown key {key}")

    _stokes_theory_options = _default_stokes_theory_options.copy() | kwargs

    _stokes_theory_options["reference_frame"] = _stokes_theory_options[
        "reference_frame"
    ].lower()
    if _stokes_theory_options["reference_frame"] not in [
        "eulerian",
        "lagrangian",
    ]:
        raise ValueError("Reference frame must be one of 'eulerian' or 'lagrangian'")
    return StokesTheoryOptions(**_stokes_theory_options)


def physics_options(**kwargs) -> PhysicsOptions:
    for key in kwargs:
        if key not in _default_physics_options:
            raise ValueError(f"Unknown key {key}")

    _physics_options = _default_physics_options.copy() | kwargs
    if _physics_options["grav"] < 0:
        raise ValueError("Gravity must be positive")

    if _physics_options["kinematic_surface_tension"] < 0:
        raise ValueError("Surface tension must be positive")

    if _physics_options["wave_type"] not in [
        "gravity",
        "capillary",
        "gravity-capillary",
    ]:
        raise ValueError(
            "Wave type must be one of 'gravity', 'capillary', or 'gravity-capillary'"
        )

    if _physics_options["wave_regime"] not in ["deep", "intermediate", "shallow"]:
        raise ValueError(
            "Wave regime must be one of 'deep', 'intermediate', or 'shallow'"
        )

    if _physics_options["wave_type"] == "gravity":
        _physics_options["kinematic_surface_tension"] = 0.0

    elif _physics_options["wave_type"] == "capillary":
        _physics_options["grav"] = 0.0

    if _physics_options["wave_regime"] == "deep":
        _physics_options["wave_regime_enum"] = 1
    elif _physics_options["wave_regime"] == "shallow":
        _physics_options["wave_regime_enum"] = 2
    else:
        _physics_options["wave_regime_enum"] = 0

    return PhysicsOptions(**_physics_options)


_default_numerical_options = {
    "relative_tolerance": _RELATIVE_TOLERANCE,
    "absolute_tolerance": _ABSOLUTE_TOLERANCE,
    "maximum_number_of_iterations": _MAXIMUM_NUMBER_OF_ITERATIONS,
    "under_relaxation_factor": _UNDER_RELAXATION_FACTOR,
}

NumericalOptions = namedtuple(
    "NumericalOptions",
    [
        *list(_default_numerical_options.keys()),
    ],
)


def numerical_options(**kwargs) -> NumericalOptions:
    for key in kwargs:
        if key not in _default_numerical_options:
            raise ValueError(f"Unknown key {key}")

    _numerical_options = _default_numerical_options.copy() | kwargs
    if _numerical_options["relative_tolerance"] < 0:
        raise ValueError("Relative tolerance must be positive")

    if _numerical_options["absolute_tolerance"] < 0:
        raise ValueError("Absolute tolerance must be positive")

    if _numerical_options["maximum_number_of_iterations"] < 1:
        raise ValueError("Maximum number of iterations must be at least 1")

    return NumericalOptions(**_numerical_options)


default_numerical_options = numerical_options()
default_physical_options = physics_options()
default_stokes_theory_options = stokes_theory_options()


@jit(**numba_default)
def _parse_options(numerical, physical, nonlinear):
    """
    parse input options and return default classes if None. A word of warning. Calling this function from within a numba
    jitted function prevents us from using keyword or default arguments. This is because numba does not support them it
    seems (???). Below is a workaround where we explicitly set the default values if they are None. This is not ideal.

    :param numerical:
    :param physical:
    :return:
    """
    if numerical is None:
        numerical = default_numerical_options

    if physical is None:
        physical = default_physical_options

    if nonlinear is None:
        nonlinear = default_stokes_theory_options

    return numerical, physical, nonlinear
