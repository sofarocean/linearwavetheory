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
    if not np.isfinite(kd) or kd > 350:
        # if depth is infinite the division is undefined; for large (but finite) kd, sinh(2*kd) overflows to inf
        # while the correction term is already numerically negligible, so we use the same deep-water limit.
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
        error_derivative_to_wavenumber = (
            _intrinsic_group_speed_shallow(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            + innerproduct
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        )

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
        error_derivative_to_wavenumber = (
            _intrinsic_group_speed_intermediate(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            + innerproduct
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        )

    return wavenumber_estimate


_group_speed_inversion_signatures = [
    float64(float64, float64, float64, float64, int64, float64, float64),
    float32(float32, float32, float32, float32, int32, float32, float32),
    float32(float32, float32, float64, float64, int64, float64, float64),
]


@vectorize(_group_speed_inversion_signatures, **numba_default_vectorize)
def _inverse_group_speed_intermediate(
    target_group_speed,
    depth,
    kinematic_surface_tension,
    grav,
    maximum_number_of_iterations,
    relative_tolerance,
    absolute_tolerance,
):
    """
    Solve for the wavenumber magnitude k at which the intrinsic group speed Cg(k,d) equals a target speed, using
    bisection. For pure gravity waves Cg is a strictly monotonically decreasing function of k (from sqrt(g*d) at k=0
    down to the deep-water asymptote as k -> infinity), so bisection is unconditionally convergent and no closed-form
    inverse is needed. Used to locate the wave-current "blocking" fold point (c.f. Peregrine, 1976) for
    inverse_encounter_dispersion_relation.

    NOTE: this assumes negligible surface tension; a nonzero kinematic_surface_tension can make Cg non-monotonic at
    capillary wavenumbers, which is outside the scope of the fold-point analysis this helper supports.

    ** Internal function. **
    """
    if depth <= 0:
        return np.nan

    shallow_limit = np.sqrt(grav * depth)
    if target_group_speed <= 0.0 or target_group_speed >= shallow_limit:
        # No interior solution: Cg(k) spans (0, shallow_limit) for k in (0, infinity).
        return np.nan

    lower = 0.0
    # Deep water estimate: Cg = 0.5 * sqrt(g / k)  =>  k = g / (4 * Cg**2)
    upper = 0.25 * grav / target_group_speed**2

    while (
        _intrinsic_group_speed_intermediate(
            upper, depth, kinematic_surface_tension, grav
        )
        >= target_group_speed
    ):
        upper *= 2

    wavenumber_estimate = 0.5 * (lower + upper)
    for _ in range(0, maximum_number_of_iterations):
        wavenumber_estimate = 0.5 * (lower + upper)
        speed = _intrinsic_group_speed_intermediate(
            wavenumber_estimate, depth, kinematic_surface_tension, grav
        )
        error = speed - target_group_speed
        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / target_group_speed) < relative_tolerance
        ):
            break

        if speed > target_group_speed:
            lower = wavenumber_estimate
        else:
            upper = wavenumber_estimate

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


_inverse_encounter_relation_signatures = [
    float64(float64, float64, float64, float64, int64),
    float32(float32, float32, float32, float32, int32),
    float32(float32, float32, float64, float64, int64),
]


@vectorize(_inverse_encounter_relation_signatures, **numba_default_vectorize)
def _inverse_encounter_dispersion_relation_deep(omega, depth, delta, grav, principal):
    """
    Closed-form inverse encounter dispersion relation in deep water for a moving observer/current, following the
    derivation in the "Estimating wavenumber from encounter frequency" design note. Solves

        omega = s * sqrt(g*k) + k * delta          (s=+1 principal root, s=-1 secondary root)

    for the wavenumber magnitude k, where delta = V * cos(theta_p - theta_V) is the projection of the relative
    current velocity onto the wave direction, and omega >= 0 is the encounter angular frequency.

    Both roots reduce to the same quadratic in sqrt(k), with discriminant `1 + 4*omega*delta/g`. This single
    expression covers both delta>=0 (design doc "Option 1") and delta<0 (design doc "Option 2") uniformly: for
    delta<0 the discriminant becomes `1 - 4*omega*|delta|/g`, matching the "wave-blocking" quadratic of Option 2.
    A negative discriminant means the current is strong enough to sweep away all waves of this frequency in this
    direction (the deep-water closed form of the finite-depth "wave-blocking" fold point, c.f. Peregrine, 1976).

    NOTE: valid for pure gravity waves only -- surface tension is not included in this closed-form solution
    (unlike the rest of the module's deep-water functions). Use wave_regime='intermediate' for gravity-capillary
    waves in the capillary range.

    ** Internal function. Call inverse_encounter_dispersion_relation instead. **
    """
    if depth <= 0:
        return np.nan

    if principal:
        if omega == 0.0:
            return 0.0
    else:
        if delta == 0.0:
            # Secondary root is undefined without a current projection onto the wave direction.
            return np.nan

        if omega == 0.0:
            # The closed-form expression below is a genuine 0/0 at omega=0 for the secondary root; taking the
            # limit gives the nonzero "stationary crest" wavenumber where Omega(k) = k * delta exactly.
            return grav / delta**2

    discriminant = 1.0 + 4.0 * omega * delta / grav
    if discriminant < 0.0:
        return np.nan

    root = np.sqrt(discriminant)
    denominator = (1.0 + root) if principal else (root - 1.0)

    return 4.0 * omega**2 / (grav * denominator**2)


@vectorize(_inverse_encounter_relation_signatures, **numba_default_vectorize)
def _inverse_encounter_dispersion_relation_shallow(
    omega, depth, delta, grav, principal
):
    """
    Closed-form inverse encounter dispersion relation in shallow water for a moving observer/current, following the
    derivation in the "Estimating wavenumber from encounter frequency" design note. Solves

        omega = s * k * c_sw + k * delta          (s=+1 principal root, s=-1 secondary root)

    with c_sw = sqrt(g*d) the (non-dispersive) shallow-water phase speed, so k = omega / (s*c_sw + delta).

    Since the shallow-water dispersion relation is linear in k (unlike deep/intermediate water), this single
    formula -- with the existence guard `denominator > 0` -- reproduces every case in the design doc without a
    separate branch on the sign of delta:
    - principal (s=+1): denominator = delta + c_sw, which is exactly the design doc's Option 1 (delta>=0) formula
      and, algebraically, its Option 2 (delta<0) formula `c_sw - |delta|` too (both equal delta + c_sw). The root
      only fails to exist if the current retreats at least as fast as the shallow-water wave speed.
    - secondary (s=-1): denominator = delta - c_sw, matching Option 1's secondary root, which only exists when
      supercritical (delta > c_sw) -- note the design doc's prose describing this condition is inverted relative
      to its own formula; implemented per the formula. For delta<0 this denominator is always negative, which
      naturally reproduces the design doc's separate statement that the secondary root never exists in that case.

    NOTE: valid for pure gravity waves only -- surface tension is not included (unlike the rest of the module's
    shallow-water functions).

    ** Internal function. Call inverse_encounter_dispersion_relation instead. **
    """
    if depth <= 0:
        return np.nan

    shallow_water_speed = np.sqrt(grav * depth)
    denominator = (
        (delta + shallow_water_speed) if principal else (delta - shallow_water_speed)
    )

    if denominator <= 0.0:
        return np.nan

    return omega / denominator


_inverse_encounter_relation_intermediate_signatures = [
    float64(
        float64, float64, float64, int64, float64, float64, int64, float64, float64
    ),
    float32(
        float32, float32, float32, int32, float32, float32, int32, float32, float32
    ),
    float32(
        float32, float32, float64, int64, float64, float64, int64, float64, float64
    ),
]


@vectorize(
    _inverse_encounter_relation_intermediate_signatures, **numba_default_vectorize
)
def _inverse_encounter_dispersion_relation_intermediate_negative_delta(
    omega,
    depth,
    delta,
    principal,
    kinematic_surface_tension,
    grav,
    maximum_number_of_iterations,
    relative_tolerance,
    absolute_tolerance,
):
    """
    Solves omega = Omega(k,d) + k*delta for delta < 0 in intermediate depth, i.e. Omega(k,d) - k*|delta| = omega.
    This function rises from 0 at k=0 to a maximum omega_max at the "blocking" fold point k_max (where
    Cg(k_max,d)=|delta|), then falls monotonically -- so there are two roots (principal, k<k_max; secondary,
    k>k_max) whenever omega < omega_max, a degenerate one at omega==omega_max, and none for omega > omega_max
    (c.f. Peregrine, 1976, on wave blocking by an opposing current). If |delta| exceeds the shallow-water speed
    there is no interior maximum at all (the function is monotonically decreasing everywhere), so only omega=0
    (k=0, principal branch) has a solution.

    Uses bracketed Newton iteration with a bisection fallback (Newton is poorly conditioned near the fold, where
    the two roots merge and the derivative vanishes).

    ** Internal function. Call inverse_encounter_dispersion_relation instead. **
    """
    if depth <= 0:
        return np.nan

    abs_delta = -delta

    if omega == 0.0:
        return 0.0 if principal else np.nan

    k_max = _inverse_group_speed_intermediate(
        abs_delta,
        depth,
        kinematic_surface_tension,
        grav,
        maximum_number_of_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    if np.isnan(k_max):
        # |delta| is at or beyond the shallow-water limit: no interior maximum, no positive-frequency solution.
        return np.nan

    omega_max = (
        _intrinsic_dispersion_relation_intermediate(
            k_max, depth, kinematic_surface_tension, grav
        )
        - k_max * abs_delta
    )

    if omega_max <= 0.0:
        return np.nan

    excess = omega - omega_max
    if excess > absolute_tolerance or excess > relative_tolerance * omega_max:
        return np.nan

    if (
        np.abs(excess) < absolute_tolerance
        and np.abs(excess) < relative_tolerance * omega_max
    ):
        return k_max

    if principal:
        lower_bound = 0.0
        upper_bound = k_max
    else:
        lower_bound = k_max
        upper_bound = k_max * 2.0
        while (
            _intrinsic_dispersion_relation_intermediate(
                upper_bound, depth, kinematic_surface_tension, grav
            )
            - upper_bound * abs_delta
            > omega
        ):
            upper_bound *= 2.0

    # Seed with the deep-water closed-form estimate (same discriminant as the deep-water kernel, using the
    # signed delta) rather than the bare bracket midpoint -- the bracket alone can be many orders of magnitude
    # wider than the true root, and bisection needs ~log2(bracket/tolerance) iterations to close that gap.
    deep_water_discriminant = 1.0 + 4.0 * omega * delta / grav
    wavenumber_estimate = np.nan
    if deep_water_discriminant >= 0.0:
        deep_water_root = np.sqrt(deep_water_discriminant)
        if principal:
            deep_water_estimate = 4.0 * omega**2 / (grav * (1.0 + deep_water_root) ** 2)
        elif deep_water_root > 1.0:
            deep_water_estimate = 4.0 * omega**2 / (grav * (deep_water_root - 1.0) ** 2)
        else:
            deep_water_estimate = np.nan

        if lower_bound < deep_water_estimate < upper_bound:
            wavenumber_estimate = deep_water_estimate

    if not (lower_bound < wavenumber_estimate < upper_bound):
        wavenumber_estimate = 0.5 * (lower_bound + upper_bound)
    for _ in range(0, maximum_number_of_iterations):
        value = (
            _intrinsic_dispersion_relation_intermediate(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            - wavenumber_estimate * abs_delta
        )
        error = value - omega
        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / omega) < relative_tolerance
        ):
            break

        derivative = (
            _intrinsic_group_speed_intermediate(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
            - abs_delta
        )
        newton_step = (
            wavenumber_estimate - error / derivative if derivative != 0.0 else np.nan
        )

        if principal:
            # value increases with k on [0, k_max]
            if value > omega:
                upper_bound = wavenumber_estimate
            else:
                lower_bound = wavenumber_estimate
        else:
            # value decreases with k on [k_max, upper_bound]
            if value > omega:
                lower_bound = wavenumber_estimate
            else:
                upper_bound = wavenumber_estimate

        if lower_bound < newton_step < upper_bound:
            wavenumber_estimate = newton_step
        else:
            wavenumber_estimate = 0.5 * (lower_bound + upper_bound)

    return wavenumber_estimate


@vectorize(_inverse_intrinsic_relation_signatures, **numba_default_vectorize)
def _inverse_encounter_dispersion_relation_intermediate_positive_delta_secondary(
    omega,
    depth,
    delta,
    kinematic_surface_tension,
    grav,
    maximum_number_of_iterations,
    relative_tolerance,
    absolute_tolerance,
):
    """
    Secondary-root solve for delta > 0 in intermediate depth: k*delta - Omega(k,d) = omega. This is unimodal, with
    an interior minimum at k_min where Cg(k_min,d)=delta (only present if delta is below the shallow-water speed
    limit), strictly increasing and unbounded thereafter -- so there is always exactly one root for any omega>=0.

    Uses bracketed Newton iteration with a bisection fallback, seeded from the deep-water closed-form secondary
    root estimate.

    ** Internal function. Call inverse_encounter_dispersion_relation instead. **
    """
    if depth <= 0:
        return np.nan

    if delta <= 0.0:
        return np.nan

    k_min = _inverse_group_speed_intermediate(
        delta,
        depth,
        kinematic_surface_tension,
        grav,
        maximum_number_of_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    lower_bound = 0.0 if np.isnan(k_min) else k_min

    # Deep-water closed-form secondary-root estimate, used as an initial upper-bound seed.
    omega_prime = omega * delta / grav
    root = np.sqrt(1.0 + 4.0 * omega_prime)
    if root > 1.0:
        deep_estimate = 4.0 * omega**2 / (grav * (root - 1.0) ** 2)
    else:
        # omega == 0: closed form is a genuine 0/0; the limit is the "stationary crest" wavenumber grav/delta**2.
        deep_estimate = grav / delta**2

    upper_bound = deep_estimate if deep_estimate > lower_bound else lower_bound + 1.0
    while (
        upper_bound * delta
        - _intrinsic_dispersion_relation_intermediate(
            upper_bound, depth, kinematic_surface_tension, grav
        )
        < omega
    ):
        upper_bound *= 2.0

    wavenumber_estimate = 0.5 * (lower_bound + upper_bound)
    for _ in range(0, maximum_number_of_iterations):
        value = (
            wavenumber_estimate * delta
            - _intrinsic_dispersion_relation_intermediate(
                wavenumber_estimate, depth, kinematic_surface_tension, grav
            )
        )
        error = value - omega
        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / max(omega, 1e-300)) < relative_tolerance
        ):
            break

        derivative = delta - _intrinsic_group_speed_intermediate(
            wavenumber_estimate, depth, kinematic_surface_tension, grav
        )
        newton_step = (
            wavenumber_estimate - error / derivative if derivative != 0.0 else np.nan
        )

        # value increases with k on [lower_bound, upper_bound]
        if value > omega:
            upper_bound = wavenumber_estimate
        else:
            lower_bound = wavenumber_estimate

        if lower_bound < newton_step < upper_bound:
            wavenumber_estimate = newton_step
        else:
            wavenumber_estimate = 0.5 * (lower_bound + upper_bound)

    return wavenumber_estimate


@vectorize(
    _inverse_encounter_relation_intermediate_signatures, **numba_default_vectorize
)
def _inverse_encounter_dispersion_relation_intermediate(
    omega,
    depth,
    delta,
    principal,
    kinematic_surface_tension,
    grav,
    maximum_number_of_iterations,
    relative_tolerance,
    absolute_tolerance,
):
    """
    Numerical inverse encounter dispersion relation at general (intermediate) depth for a moving observer/current,
    following the derivation in the "Estimating wavenumber from encounter frequency" design note. Solves

        omega = s * Omega(k,d) + k * delta          (s=+1 principal root, s=-1 secondary root)

    - delta >= 0, principal: Omega(k,d)+k*delta is strictly monotone increasing, so this reuses the (corrected)
      Newton solver `_inverse_intrinsic_dispersion_relation_intermediate` directly, with `delta` passed as its
      current-projection term.
    - delta >= 0, secondary: delegates to
      `_inverse_encounter_dispersion_relation_intermediate_positive_delta_secondary`.
    - delta < 0 (both roots): delegates to `_inverse_encounter_dispersion_relation_intermediate_negative_delta`,
      which locates the wave-blocking fold point first.

    ** Internal function. Call inverse_encounter_dispersion_relation instead. **
    """
    if depth <= 0:
        return np.nan

    if delta >= 0.0:
        if principal:
            return _inverse_intrinsic_dispersion_relation_intermediate(
                omega,
                depth,
                delta,
                kinematic_surface_tension,
                grav,
                maximum_number_of_iterations,
                relative_tolerance,
                absolute_tolerance,
            )
        else:
            if delta == 0.0:
                return np.nan
            return _inverse_encounter_dispersion_relation_intermediate_positive_delta_secondary(
                omega,
                depth,
                delta,
                kinematic_surface_tension,
                grav,
                maximum_number_of_iterations,
                relative_tolerance,
                absolute_tolerance,
            )
    else:
        return _inverse_encounter_dispersion_relation_intermediate_negative_delta(
            omega,
            depth,
            delta,
            principal,
            kinematic_surface_tension,
            grav,
            maximum_number_of_iterations,
            relative_tolerance,
            absolute_tolerance,
        )
