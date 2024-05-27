"""
This module contains classes for the numerical and physical options for the linear wave theory package. These options
are used to determine the numerical parameters for the linear wave theory package, as well as the physical parameters
for the linear wave theory package. The numerical parameters are used to determine the accuracy of the numerical
solution, while the physical parameters are used to determine the physical properties of the fluid and its environment.
"""
from numba import float64, int64, jit
from ._numba_settings import numba_default
from numba.core.types import unicode_type
from numba.experimental import jitclass
from typing import Literal
import numpy as np

_GRAV = 9.80665
_SURFACE_TENSION = 0.074
_WATER_DENSITY = 1025.0
_KINEMATIC_SURFACE_TENSION = 7.21951219512195e-05

# Default Numerical Parameters
_RELATIVE_TOLERANCE = 1e-3
_MAXIMUM_NUMBER_OF_ITERATIONS = 10
_ABSOLUTE_TOLERANCE = np.inf


@jitclass(
    [
        ("_kinematic_surface_tension", float64),
        ("_grav", float64),
        ("wave_type", unicode_type),
        ("wave_regime", unicode_type),
        ("density", float64),
    ]
)
class PhysicsOptions(object):
    """
    Physics options for the linear wave theory package. Contains the following attributes:

    - kinematic_surface_tension: kinematic surface tension of the fluid
    - grav: gravitational acceleration
    - wave_type: "gravity", "capillary", or "gravity-capillary", determines which dispersion relation to use for
        properties derived from the linear dispersion relation (wavenumber, groupspeed, etc.)
    - wave_regime: "deep", "intermediate", or "shallow", determines which limit of the dispersion relation to use for
        properties derived from the linear dispersion relation (wavenumber, groupspeed, etc.)

    Default values are: kinematic_surface_tension=0.074, grav=9.80665, wave_type="gravity-capillary",
    wave_regime="intermediate"
    """

    def __init__(
        self,
        kinematic_surface_tension: float = _KINEMATIC_SURFACE_TENSION,
        grav: float = _GRAV,
        wave_type: Literal[
            "gravity-capillary", "gravity", "capillary"
        ] = "gravity-capillary",
        wave_regime: Literal["deep", "intermediate", "shallow"] = "intermediate",
        water_density: float = _WATER_DENSITY,
    ):
        """
        Create object to represent physical properties of the fluid and environment.

        :param kinematic_surface_tension: kinematic surface tension of the fluid
        :param grav: gravitational acceleration
        :param wave_type: which dispersion relation to use for properties derived from the linear dispersion relation
            one of "gravity", "capillary", or "gravity-capillary"
        :param wave_regime: which limit of the dispersion relation to use for properties derived from the linear
        dispersion relation, one of "deep", "intermediate", or "shallow"
        :param water_density: density of the fluid
        """
        if grav < 0:
            raise ValueError("Gravity must be positive")

        if kinematic_surface_tension < 0:
            raise ValueError("Surface tension must be positive")

        if wave_type not in ["gravity", "capillary", "gravity-capillary"]:
            raise ValueError(
                "Wave type must be one of 'gravity', 'capillary', or 'gravity-capillary'"
            )

        if wave_regime not in ["deep", "intermediate", "shallow"]:
            raise ValueError(
                "Wave regime must be one of 'deep', 'intermediate', or 'shallow'"
            )

        self._kinematic_surface_tension = kinematic_surface_tension
        self._grav = grav
        self.wave_type = wave_type
        self.wave_regime = wave_regime
        self.density = water_density

    @property
    def kinematic_surface_tension(self) -> float:
        if self.wave_type == "gravity":
            return 0.0
        else:
            return self._kinematic_surface_tension

    @property
    def grav(self) -> float:
        if self.wave_type == "capillary":
            return 0.0
        else:
            return self._grav

    @property
    def wave_regime_enum(self) -> int:
        if self.wave_regime == "deep":
            return 1
        elif self.wave_regime == "shallow":
            return 2
        else:
            return 0


@jitclass(
    [
        ("relative_tolerance", float64),
        ("absolute_tolerance", float64),
        ("maximum_number_of_iterations", int64),
    ]
)
class NumericalOptions(object):
    """
    Numerical options for the linear wave theory package. These options are used when solving the inverse dispersion
    relation numerically using Newton iteration. Contains the following attributes:
    - relative_tolerance: relative tolerance for the numerical solver
    - absolute_tolerance: absolute tolerance for the numerical solver
    - maximum_number_of_iterations: maximum number of iterations for the numerical solver
    """

    def __init__(
        self,
        relative_tolerance: float = _RELATIVE_TOLERANCE,
        absolute_tolerance: float = _ABSOLUTE_TOLERANCE,
        maximum_number_of_iterations: int = _MAXIMUM_NUMBER_OF_ITERATIONS,
    ):

        if relative_tolerance < 0.0:
            raise ValueError("Relative tolerance must be positive")

        if absolute_tolerance < 0.0:
            raise ValueError("Absolute tolerance must be positive")

        if maximum_number_of_iterations < 1:
            raise ValueError("Maximum number of iterations must be at least 1")

        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.maximum_number_of_iterations = maximum_number_of_iterations


@jit(**numba_default)
def _parse_options(numerical, physical):
    """
    parse input options and return default classes if None. A word of warning. Calling this function from within a numba
    jitted function prevents us from using keyword or default arguments. This is because numba does not support them it
    seems (???). Below is a workaround where we explicitly set the default values if they are None. This is not ideal.

    :param numerical:
    :param physical:
    :return:
    """
    if numerical is None:
        numerical = NumericalOptions(
            _RELATIVE_TOLERANCE, _ABSOLUTE_TOLERANCE, _MAXIMUM_NUMBER_OF_ITERATIONS
        )

    if physical is None:
        physical = PhysicsOptions(
            _KINEMATIC_SURFACE_TENSION,
            _GRAV,
            "gravity-capillary",
            "intermediate",
            _WATER_DENSITY,
        )
    return numerical, physical
