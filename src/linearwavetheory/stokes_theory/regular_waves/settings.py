from enum import StrEnum
_DEFAULT_ORDER = 4


class ReferenceFrame(StrEnum):
    """
    Enum for the reference frame used in the nonlinear dispersion relation.
    """

    eulerian = "eulerian"
    lagrangian = "lagrangian"
