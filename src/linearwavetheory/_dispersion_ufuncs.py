import numpy as np
from numba import vectorize, float64, float32, int64, int32
from ._numba_settings import numba_default_vectorize


_intrinsic_relation_signatures = [
    float64(float64, float64, float64, float64),
    float32(float32, float32, float32, float32),
    float32(float32, float32, float64, float64),
]

_inverse_intrinsic_relation_signatures = [
    float64(float64, float64, float64, float64, float64, int64, float64, float64),
    float32(float32, float32, float32, float32, float32, int32, float32, float32),
    float32(float32, float32, float64, float64, float64, int64, float64, float64),
]


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_dispersion_relation_deep(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    return np.sqrt(
        (
            grav * wavenumber_magnitude
            + kinematic_surface_tension * wavenumber_magnitude**3
        )
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_dispersion_relation_shallow(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    return np.sqrt(
        (
            grav * wavenumber_magnitude**2
            + kinematic_surface_tension * wavenumber_magnitude**4
        )
        * depth
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_dispersion_relation_intermediate(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    return np.sqrt(
        (
            grav * wavenumber_magnitude
            + kinematic_surface_tension * wavenumber_magnitude**3
        )
        * np.tanh(wavenumber_magnitude * depth)
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_phase_speed_intermediate(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    return np.sqrt(
        (grav / wavenumber_magnitude + kinematic_surface_tension * wavenumber_magnitude)
        * np.tanh(wavenumber_magnitude * depth)
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_phase_speed_deep(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.inf

    return np.sqrt(
        grav / wavenumber_magnitude + kinematic_surface_tension * wavenumber_magnitude
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_phase_speed_shallow(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    return np.sqrt(depth) * np.sqrt(
        grav + kinematic_surface_tension * wavenumber_magnitude**2
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_group_speed_shallow(
    wavenumber_magnitude,
    depth,
    kinematic_surface_tension,
    grav,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    """
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    return (
        np.sqrt(depth)
        * (grav + 2 * kinematic_surface_tension * wavenumber_magnitude**2)
        / np.sqrt(grav + kinematic_surface_tension * wavenumber_magnitude**2)
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_group_speed_deep(
    wavenumber_magnitude,
    depth,
    kinematic_surface_tension,
    grav,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    """
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    # deep water
    return 0.5 * (
        (grav + 3 * kinematic_surface_tension * wavenumber_magnitude**2)
        / np.sqrt(
            grav * wavenumber_magnitude
            + kinematic_surface_tension * wavenumber_magnitude**3
        )
    )


@vectorize(_intrinsic_relation_signatures, **numba_default_vectorize)
def _intrinsic_group_speed_intermediate(
    wavenumber_magnitude,
    depth,
    kinematic_surface_tension,
    grav,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    """
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    surface_tension_term = np.sqrt(
        grav + kinematic_surface_tension * wavenumber_magnitude**2
    )
    kd = wavenumber_magnitude * depth
    if not np.isfinite(kd):
        # if depth is infinite, the division is undefined
        n = 0.5
    else:
        n = 1 / 2 + kd / np.sinh(2 * kd)

    c = np.sqrt(np.tanh(kd) / wavenumber_magnitude)
    w = wavenumber_magnitude * c

    return (
        n * c * surface_tension_term
        + w * wavenumber_magnitude * kinematic_surface_tension / surface_tension_term
    )


@vectorize(
    _inverse_intrinsic_relation_signatures,
    **numba_default_vectorize,
)
def _inverse_intrinsic_dispersion_relation_shallow(
    intrinsic_angular_frequency,
    depth,
    innerproduct,
    kinematic_surface_tension: float,
    grav: float,
    maximum_number_of_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    """
    ** Internal function. Call inverse_intrinsic_dispersion_relation instead. **
    """

    # == Input Validation ==

    # Since we can try to solve for the negative intrinsic frequency branch of the dispersion relation, we allow
    # negative intrinsic frequencies. In this case the wavenumber magnitude is the same as for the positive branch.
    intrinsic_angular_frequency = np.abs(intrinsic_angular_frequency)

    # For zero or negative depths the solution is undefined.
    if depth <= 0:
        return np.nan

    if innerproduct > 0.0 and intrinsic_angular_frequency < 0.0:
        # No solutions possible
        return np.nan

    # For zero intrinsic frequency the wavenumber is zero
    if intrinsic_angular_frequency == 0:
        return 0.0

    # == Initial Estimate ==
    wavenumber_estimate = intrinsic_angular_frequency / np.sqrt(grav * depth)

    # If the initial estimate turns out to be in the capillary range- use the pure capillary dispersion relation for
    # the initial estimate.
    if wavenumber_estimate**3 * kinematic_surface_tension / grav > 5:
        wavenumber_estimate = (
            intrinsic_angular_frequency**2 / kinematic_surface_tension
        ) ** (1 / 3)

    # == Newton Iteration ==
    for ii in range(0, maximum_number_of_iterations):
        error = (
            _intrinsic_dispersion_relation_shallow(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            + innerproduct * wavenumber_estimate
            - intrinsic_angular_frequency
        )
        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / intrinsic_angular_frequency) < relative_tolerance
        ):
            break

        # Calculate the derivative of the error function with respect to wavenumber.
        error_derivative_to_wavenumber = _intrinsic_group_speed_shallow(
            wavenumber_estimate, depth, kinematic_surface_tension, grav
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        ) + innerproduct

    return wavenumber_estimate


@vectorize(
    _inverse_intrinsic_relation_signatures,
    **numba_default_vectorize,
)
def _inverse_intrinsic_dispersion_relation_intermediate(
    intrinsic_angular_frequency,
    depth,
    innerproduct,
    kinematic_surface_tension: float,
    grav: float,
    maximum_number_of_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    """
    ** Internal function. Call inverse_intrinsic_dispersion_relation instead. **
    """

    # == Input Validation ==

    # Since we can try to solve for the negative intrinsic frequency branch of the dispersion relation, we allow
    # negative intrinsic frequencies. In this case the wavenumber magnitude is the same as for the positive branch.
    intrinsic_angular_frequency = np.abs(intrinsic_angular_frequency)

    # For zero or negative depths the solution is undefined.
    if depth <= 0:
        return np.nan

    # For zero intrinsic frequency the wavenumber is zero
    if intrinsic_angular_frequency == 0:
        return 0.0

    fac = np.tanh(intrinsic_angular_frequency / np.sqrt(grav * depth))
    # == Initial Estimate is a smooth sum of the shallow water and deep water wave number ==
    wavenumber_estimate = (
        intrinsic_angular_frequency**2 / grav * (1 - fac)
        + intrinsic_angular_frequency / np.sqrt(grav * depth) * fac
    )

    # If the initial estimate turns out to be in the capillary range- use the pure capillary dispersion relation for
    # the initial estimate.
    if wavenumber_estimate**3 * kinematic_surface_tension / grav > 5:
        wavenumber_estimate = (
            intrinsic_angular_frequency**2 / kinematic_surface_tension
        ) ** (1 / 3)

    # == Newton Iteration ==
    for ii in range(0, maximum_number_of_iterations):
        error = (
            _intrinsic_dispersion_relation_intermediate(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            - intrinsic_angular_frequency
        ) + innerproduct * wavenumber_estimate

        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / intrinsic_angular_frequency) < relative_tolerance
        ):
            break

        # Calculate the derivative of the error function with respect to wavenumber.
        error_derivative_to_wavenumber = _intrinsic_group_speed_intermediate(
            wavenumber_estimate, depth, kinematic_surface_tension, grav
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        ) + innerproduct

    return wavenumber_estimate


@vectorize(
    _inverse_intrinsic_relation_signatures,
    **numba_default_vectorize,
)
def _inverse_intrinsic_dispersion_relation_deep(
    intrinsic_angular_frequency,
    depth,
    innerproduct,
    kinematic_surface_tension: float,
    grav: float,
    maximum_number_of_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    """
    ** Internal function. Call inverse_intrinsic_dispersion_relation instead. **
    """

    # == Input Validation ==

    # Since we can try to solve for the negative intrinsic frequency branch of the dispersion relation, we allow
    # negative intrinsic frequencies. In this case the wavenumber magnitude is the same as for the positive branch.
    intrinsic_angular_frequency = np.abs(intrinsic_angular_frequency)

    # For zero or negative depths the solution is undefined.
    if depth <= 0:
        return np.nan

    # For zero intrinsic frequency the wavenumber is zero
    if intrinsic_angular_frequency == 0:
        return 0.0

    # == Initial Estimate ==
    wavenumber_estimate = intrinsic_angular_frequency**2 / grav

    # If the initial estimate turns out to be in the capillary range- use the pure capillary dispersion relation for
    # the initial estimate.
    if wavenumber_estimate**3 * kinematic_surface_tension / grav > 5:
        wavenumber_estimate = (
            intrinsic_angular_frequency**2 / kinematic_surface_tension
        ) ** (1 / 3)

    # == Newton Iteration ==
    for ii in range(0, maximum_number_of_iterations):
        error = (
            _intrinsic_dispersion_relation_deep(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            + innerproduct * wavenumber_estimate
            - intrinsic_angular_frequency
        )

        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / intrinsic_angular_frequency) < relative_tolerance
        ):
            break

        # Calculate the derivative of the error function with respect to wavenumber.
        error_derivative_to_wavenumber = (
            _intrinsic_group_speed_deep(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            + innerproduct
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        )

    return wavenumber_estimate
