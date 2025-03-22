import numpy as np
from linearwavetheory.stokes_theory.regular_waves.settings import _DEFAULT_ORDER
from linearwavetheory.stokes_theory.regular_waves.mean_properties import (
    dimensionless_stokes_drift,
)
from linearwavetheory.dispersion import intrinsic_dispersion_relation
from .settings import ReferenceFrame

def nonlinear_dispersion_relation(steepness, wavenumber, depth, height=0, **kwargs):

    linear_angular_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, kwargs.get("physics_options", None)
    )

    return (
        dimensionless_nonlinear_dispersion_relation(
            steepness, wavenumber * depth, wavenumber * height, **kwargs
        )
        * linear_angular_frequency
    )


def dimensionless_nonlinear_dispersion_relation(
    steepness, relative_depth, relative_height=0,reference_frame:ReferenceFrame=ReferenceFrame.eulerian, **kwargs
):
    """
    This function calculates the nonlinear dispersion relation for a fourth order Stokes wave.
    :param steepness:
    :param wavenumber:
    :param kwargs:
    :return:
    """

    order = kwargs.get("order", _DEFAULT_ORDER)

    # In a Lagrangian reference frame we need to take into account the Stokes drift.
    if reference_frame == ReferenceFrame.eulerian:
        us = 0.0
    elif reference_frame == ReferenceFrame.lagrangian:
        us = dimensionless_stokes_drift(steepness, relative_depth, relative_height)
    else:
        raise ValueError("Invalid reference frame")

    mu = np.tanh(relative_depth)

    if order < 2:
        return 1

    if physics_options.wave_regime == "shallow":
        # Limit of the general case as kd -> 0. Note that the limit is singular, and the perturbation expansion formally
        # breaks down if steepness/kd**3 (Ursell number) is not small.
        _disp = 1 + 9 * steepness**2 / relative_depth**4

    elif physics_options.wave_regime == "deep":
        _disp = 1 + 0.5 * steepness**2

    else:
        # See 3.7 in Zhao and Liu (2022). # Be aware that omega_2 in Zhao and Liu (2022) is off by a factor 1/4,
        # otherwise it does not match the deep water limit, nor other published results. Note also that I have
        # rewritten the relation into a form without "alpha" - as it is more commonly presented in the literature.
        _disp = (
            1 + steepness**2 * (9 - 10 * mu**2 + 9 * mu**4) / 16 / mu**4 - us
        )

    if order < 4:
        return _disp

    if physics_options.wave_regime == "shallow":
        raise NotImplementedError(
            "Fourth order dispersion relation not implemented for shallow water waves"
        )

    elif physics_options.wave_regime == "deep":
        raise NotImplementedError(
            "Fourth order dispersion relation not implemented for deep water waves"
        )

    else:
        _disp += (
            steepness**4
            * (
                81
                - 603 * mu**2
                + 3618 * mu**4
                - 3662 * mu**6
                + 1869 * mu**8
                - 663 * mu**10
            )
            / (1024 * mu**10)
        )

    return _disp
