from numba import float64, int64, jit
from ._numba_settings import numba_default
from numba.core.types import unicode_type
from numba.experimental import jitclass

from ._constants import (
    GRAV,
    KINEMATIC_SURFACE_TENSION,
    RELATIVE_TOLERANCE,
    MAXIMUM_NUMBER_OF_ITERATIONS,
    ABSOLUTE_TOLERANCE,
)


@jitclass(
    [
        ("_kinematic_surface_tension", float64),
        ("_grav", float64),
        ("wave_type", unicode_type),
        ("wave_regime", unicode_type),
    ]
)
class PhysicsOptions(object):
    def __init__(
        self,
        kinematic_surface_tension=KINEMATIC_SURFACE_TENSION,
        grav=GRAV,
        wave_type="gravity-capillary",
        wave_regime="intermediate",
    ):

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

    @property
    def kinematic_surface_tension(self):
        if self.wave_type == "gravity":
            return 0.0
        else:
            return self._kinematic_surface_tension

    @property
    def grav(self):
        if self.wave_type == "capillary":
            return 0.0
        else:
            return self._grav

    @property
    def wave_regime_enum(self):
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
    def __init__(
        self,
        relative_tolerance=RELATIVE_TOLERANCE,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        maximum_number_of_iterations=MAXIMUM_NUMBER_OF_ITERATIONS,
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
def parse_options(numerical, physical):
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
            RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE, MAXIMUM_NUMBER_OF_ITERATIONS
        )

    if physical is None:
        physical = PhysicsOptions(
            KINEMATIC_SURFACE_TENSION, GRAV, "gravity-capillary", "intermediate"
        )
    return numerical, physical
