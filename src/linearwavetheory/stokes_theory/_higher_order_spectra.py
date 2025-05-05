from ._nonlinear_dispersion import _pointwise_estimate_nonlinear_dispersion
from ._third_order_coeficients import (
    third_order_amplitude_correction,
    third_order_amplitude_correction_sigma_coordinates,
)
from ._second_order_coeficients import (
    _second_order_surface_elevation,
    _second_order_lagrangian_surface_elevation,
)
from linearwavetheory import inverse_intrinsic_dispersion_relation
import numpy as np
from linearwavetheory._numba_settings import numba_default
from numba import jit, prange
from linearwavetheory._array_shape_preprocessing import atleast_1d
from linearwavetheory._utils import _direction_bin
from typing import Union
from ..settings import PhysicsOptions, _parse_options, StokesTheoryOptions


@jit(**numba_default)
def nonlinear_wave_spectra_1d(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
    progress_bar=None,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_angular_frequency: intrinsic frequency of the variance density spectrum (rad Hz)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2 s/rad/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    dims = variance_density.shape

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    nspec = int(np.prod(np.array(dims[:-2])))
    nfreq = len(intrinsic_angular_frequency)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    spectra = np.zeros(
        (nspec, nfreq),
    )
    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

        spectra[i, :] = _point_bound_surface_wave_spectrum_1d(
            intrinsic_angular_frequency,
            angle_degrees,
            variance_density[i, :, :],
            depth[i],
            output,
            nonlinear_options,
            physics_options,
        )

    return spectra.reshape(dims[:-1])


@jit(**numba_default)
def _point_bound_surface_wave_spectrum_1d(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: float = np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_angular_frequency: intrinsic frequency of the variance density spectrum (rad/s)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2 s/rad/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    number_of_frequencties = len(intrinsic_angular_frequency)
    nonlinear_wave_spectrum = np.zeros(
        (number_of_frequencties,), dtype=variance_density.dtype
    )

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    if nonlinear_options.include_bound_waves:
        if (
            nonlinear_options.include_sum_interactions
            and nonlinear_options.include_difference_interactions
        ):
            sign_indices = np.array((1, -1))
        elif nonlinear_options.include_sum_interactions:
            sign_indices = np.array((1,))
        elif nonlinear_options.include_difference_interactions:
            sign_indices = np.array((-1,))
        else:
            sign_indices = np.array((), dtype=np.int64)

        for i in range(number_of_frequencties):
            for sign_index in sign_indices:
                nonlinear_wave_spectrum[i] += estimate_bound_contribution(
                    intrinsic_angular_frequency[i],
                    intrinsic_angular_frequency,
                    angle_degrees,
                    variance_density,
                    sign_index,
                    depth,
                    output,
                    nonlinear_options,
                    physics_options,
                )

    if nonlinear_options.include_nonlinear_amplitude_correction:
        for i in range(number_of_frequencties):
            nonlinear_wave_spectrum[i] += estimate_nonlinear_amplitude_correction_1d(
                intrinsic_angular_frequency[i],
                intrinsic_angular_frequency,
                angle_degrees,
                variance_density,
                depth,
                output,
                nonlinear_options,
                physics_options,
            )

    if nonlinear_options.include_nonlinear_dispersion:
        stokes_correction = stokes_dispersive_correction(
            intrinsic_angular_frequency,
            angle_degrees,
            variance_density,
            depth,
            output,
            nonlinear_options,
            physics_options,
        )
        number_of_directions = len(angle_degrees)
        direction_bin = _direction_bin(angle_degrees, wrap=360)
        for i in range(number_of_frequencties):
            for j in range(number_of_directions):
                nonlinear_wave_spectrum[i] += stokes_correction[i, j] * direction_bin[j]

    return nonlinear_wave_spectrum


@jit(**numba_default)
def estimate_nonlinear_amplitude_correction_1d(
    angular_frequency_target,
    angular_frequency,
    angles_degrees_coordinates,
    variance_density,
    depth=np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
) -> float:
    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.
    _, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    # Component 1
    # ----------------
    wavenumbers = inverse_intrinsic_dispersion_relation(angular_frequency, depth)

    # precalculate the trigonometric functions
    angles_rad_coordinates = np.deg2rad(angles_degrees_coordinates)
    _cos = np.cos(angles_rad_coordinates)
    _sin = np.sin(angles_rad_coordinates)

    # Set up the integration
    # ----------------
    # Get angle stepsizes for integration. We use the midpoint rule for the angles.
    delta_angle = _direction_bin(angles_degrees_coordinates, wrap=360)
    sum = 0.0

    # Integrate
    # ----------------
    nfreq = len(angular_frequency)
    nangle = len(angles_degrees_coordinates)

    ifreq0 = np.searchsorted(angular_frequency, angular_frequency_target)

    # Compoment 1 wavenumber, components and angular frequency
    w1 = angular_frequency[ifreq0]
    k1 = wavenumbers[ifreq0]

    if nonlinear_options.reference_frame == "eulerian":
        lagrangian = False
    elif nonlinear_options.reference_frame == "lagrangian":
        lagrangian = True
    else:
        raise Exception("Unknown kind must be one of 'eulerian', 'lagrangian'")

    for ifreq in range(nfreq):
        # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints and
        # df inbetween.
        iu = min(ifreq + 1, nfreq - 1)
        il = max(ifreq - 1, 0)
        delta_freq = (angular_frequency[iu] - angular_frequency[il]) / 2.0
        kx1 = k1 * _cos
        ky1 = k1 * _sin

        # Compoment 2 wavenumber, components and angular frequency. The sign index is used to determine if we are
        # calculating the sum (sign_index=1) or difference (sign_index=-1) interactions.
        k2 = wavenumbers[ifreq]
        w2 = angular_frequency[ifreq]
        kx2 = k2 * _cos
        ky2 = k2 * _sin

        e1 = variance_density[ifreq0, :]

        for iangle in range(nangle):
            e2 = variance_density[ifreq, iangle]

            if nonlinear_options.use_s_theory:
                interaction_coef = third_order_amplitude_correction_sigma_coordinates(
                    w1,
                    k1,
                    kx1[iangle],
                    ky1[iangle],
                    w2,
                    k2,
                    kx2,
                    ky2,
                    depth,
                    physics_options.grav,
                    nonlinear_options.wave_driven_flow_included_in_mean_flow,
                    nonlinear_options.wave_driven_setup_included_in_mean_depth,
                    lagrangian,
                )
            else:
                interaction_coef = third_order_amplitude_correction(
                    w1,
                    k1,
                    kx1[iangle],
                    ky1[iangle],
                    w2,
                    k2,
                    kx2,
                    ky2,
                    depth,
                    physics_options.grav,
                    nonlinear_options.wave_driven_flow_included_in_mean_flow,
                    nonlinear_options.wave_driven_setup_included_in_mean_depth,
                    lagrangian,
                )

            # Factor 2 because positive/negative frequencies. /4 because one-sided densities.
            sum += (
                2
                * (
                    np.sum(e1 * e2 * interaction_coef * delta_angle)
                    * delta_freq
                    * delta_angle[iangle]
                )
                / 4
            )

    return sum


@jit(**numba_default)
def estimate_nonlinear_amplitude_correction_2d(
    angular_frequency_target,
    angle_degrees_target,
    angular_frequency,
    angles_degrees_coordinates,
    variance_density,
    depth=np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
) -> float:
    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.
    _, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    angle_radians_target = np.deg2rad(angle_degrees_target)
    # Component 1
    # ----------------
    wavenumbers = inverse_intrinsic_dispersion_relation(angular_frequency, depth)

    # precalculate the trigonometric functions
    angles_rad_coordinates = np.deg2rad(angles_degrees_coordinates)
    _cos = np.cos(angles_rad_coordinates)
    _sin = np.sin(angles_rad_coordinates)

    # Set up the integration
    # ----------------
    # Get angle stepsizes for integration. We use the midpoint rule for the angles.
    delta_angle = _direction_bin(angles_degrees_coordinates, wrap=360)
    sum = 0.0

    # Integrate
    # ----------------
    nfreq = len(angular_frequency)
    nangle = len(angles_degrees_coordinates)

    ifreq0 = np.searchsorted(angular_frequency, angular_frequency_target)
    iangle = np.searchsorted(angles_degrees_coordinates, angle_degrees_target)

    # Compoment 1 wavenumber, components and angular frequency
    w1 = angular_frequency[ifreq0]
    k1 = wavenumbers[ifreq0]

    if nonlinear_options.reference_frame == "eulerian":
        lagrangian = False
    elif nonlinear_options.reference_frame == "lagrangian":
        lagrangian = True
    else:
        raise Exception("Unknown kind must be one of 'eulerian', 'lagrangian'")

    for ifreq in range(nfreq):
        # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints and
        # df inbetween.
        iu = min(ifreq + 1, nfreq - 1)
        il = max(ifreq - 1, 0)
        delta_freq = (angular_frequency[iu] - angular_frequency[il]) / 2.0
        kx1 = k1 * _cos[iangle]
        ky1 = k1 * _sin[iangle]

        # Compoment 2 wavenumber, components and angular frequency. The sign index is used to determine if we are
        # calculating the sum (sign_index=1) or difference (sign_index=-1) interactions.
        k2 = wavenumbers[ifreq]
        w2 = angular_frequency[ifreq]
        kx2 = k2 * _cos
        ky2 = k2 * _sin

        e1 = variance_density[ifreq0, :]
        e2 = variance_density[ifreq, iangle]

        if nonlinear_options.use_s_theory:
            interaction_coef = third_order_amplitude_correction_sigma_coordinates(
                w1,
                k1,
                kx1,
                ky1,
                w2,
                k2,
                kx2,
                ky2,
                depth,
                physics_options.grav,
                nonlinear_options.wave_driven_flow_included_in_mean_flow,
                nonlinear_options.wave_driven_setup_included_in_mean_depth,
                lagrangian,
            )
        else:
            interaction_coef = third_order_amplitude_correction(
                w1,
                k1,
                kx1,
                ky1,
                w2,
                k2,
                kx2,
                ky2,
                depth,
                physics_options.grav,
                nonlinear_options.wave_driven_flow_included_in_mean_flow,
                nonlinear_options.wave_driven_setup_included_in_mean_depth,
                lagrangian,
            )

        # Factor 2 because positive/negative frequencies. /4 because one-sided densities.
        sum += 2 * (np.sum(e1 * e2 * interaction_coef * delta_angle) * delta_freq) / 4

    return sum


@jit(**numba_default)
def stokes_dispersive_correction(
    angular_frequency,
    angles_degrees_coordinates,
    variance_density,
    depth=np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
) -> float:
    _, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    nl_dispersion = _pointwise_estimate_nonlinear_dispersion(
        angular_frequency,
        angles_degrees_coordinates,
        variance_density,
        depth=depth,
        nonlinear_options=nonlinear_options,
        physics_options=physics_options,
    )

    func = nl_dispersion * variance_density
    grad = np.zeros_like(func)

    nfreq = len(angular_frequency)
    for ifreq in range(nfreq):
        id = max(ifreq - 1, 0)
        iu = min(ifreq + 1, nfreq - 1)
        df = angular_frequency[iu] - angular_frequency[id]
        grad[ifreq, :] = -(func[iu, :] - func[id, :]) / df

    return grad


@jit(**numba_default)
def estimate_bound_contribution(
    angular_frequency_target,
    angular_frequency,
    angles_degrees_coordinates,
    variance_density,
    sign_index,
    depth=np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
) -> float:
    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.

    _, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    if sign_index == 1:
        # For sum frequencies the limit is given by frequency_target - fmax >= 0
        fmax = angular_frequency_target / 2
        factor = 2
    else:
        # For difference frequencies the limit is given by frequency_target + fmax <= max(frequency)
        fmax = np.max(angular_frequency) - angular_frequency_target
        factor = 2

    # Component 1
    # ----------------
    angular_freq_component1 = np.arange(0, fmax, np.min(np.diff(angular_frequency)))
    df = np.min(np.diff(angular_frequency))
    angular_freq_component1 = np.linspace(0, fmax, int(fmax / df) + 1)
    variance_density_component1 = bilinear_interpolation(
        angular_frequency,
        angles_degrees_coordinates,
        variance_density,
        angular_freq_component1,
        angles_degrees_coordinates,
    )

    wavenumber1_component1 = inverse_intrinsic_dispersion_relation(
        angular_freq_component1, depth
    )

    # Component 2
    # ----------------
    angular_freq_component2 = (
        angular_frequency_target - sign_index * angular_freq_component1
    )
    variance_density_component2 = bilinear_interpolation(
        angular_frequency,
        angles_degrees_coordinates,
        variance_density,
        angular_freq_component2,
        angles_degrees_coordinates,
    )
    wavenumber_component2 = inverse_intrinsic_dispersion_relation(
        angular_freq_component2, depth
    )

    # precalculate the trigonometric functions
    angles_rad_coordinates = np.deg2rad(angles_degrees_coordinates)
    _cos = np.cos(angles_rad_coordinates)
    _sin = np.sin(angles_rad_coordinates)

    # Set up the integration
    # ----------------
    # Get angle stepsizes for integration. We use the midpoint rule for the angles.
    delta_angle = _direction_bin(angles_degrees_coordinates, wrap=360)
    sum = 0.0

    # Integrate
    # ----------------
    nfreq = len(angular_freq_component1)
    nangle = len(angles_degrees_coordinates)
    for ifreq in range(nfreq):
        # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints and
        # df inbetween.
        iu = min(ifreq + 1, nfreq - 1)
        il = max(ifreq - 1, 0)
        delta_freq = (angular_freq_component1[iu] - angular_freq_component1[il]) / 2.0

        # Compoment 1 wavenumber, components and angular frequency
        w1 = angular_freq_component1[ifreq]
        k1 = wavenumber1_component1[ifreq]
        kx1 = k1 * _cos
        ky1 = k1 * _sin

        # Compoment 2 wavenumber, components and angular frequency. The sign index is used to determine if we are
        # calculating the sum (sign_index=1) or difference (sign_index=-1) interactions.
        k2 = wavenumber_component2[ifreq]
        w2 = sign_index * angular_freq_component2[ifreq]
        kx2 = sign_index * k2 * _cos
        ky2 = sign_index * k2 * _sin

        e2 = variance_density_component2[ifreq, :]
        for iangle in range(nangle):
            e1 = variance_density_component1[ifreq, iangle]

            # Calculate the interaction coefficients. Use interaction coefficient appropriate for Eulerian or Lagrangian
            # observations.
            if nonlinear_options.reference_frame == "eulerian":
                interaction_coef = _second_order_surface_elevation(
                    w1,
                    k1,
                    kx1[iangle],
                    ky1[iangle],
                    w2,
                    k2,
                    kx2,
                    ky2,
                    depth,
                    physics_options.grav,
                )
            elif nonlinear_options.reference_frame == "lagrangian":
                interaction_coef = _second_order_lagrangian_surface_elevation(
                    w1,
                    k1,
                    kx1[iangle],
                    ky1[iangle],
                    w2,
                    k2,
                    kx2,
                    ky2,
                    depth,
                    physics_options.grav,
                )
            else:
                raise Exception("Unknown kind must be one of 'eulerian', 'lagrangian'")

            sum += (
                np.sum(e1 * e2 * interaction_coef**2 * delta_angle)
                * delta_freq
                * delta_angle[iangle]
            )
    return factor * sum


@jit(**numba_default)
def bilinear_interpolation(
    freq: np.ndarray,
    angle: np.ndarray,
    func: np.ndarray,
    freq_int: np.ndarray,
    angle_int: np.ndarray,
) -> np.ndarray:
    """
    Simple bilinear interpolation of a 2D function. Supports wrapping of the angle axis.
    :param freq:
    :param angle:
    :param func:
    :param freq_int:
    :param angle_int:
    :return:
    """
    freq_int = atleast_1d(freq_int)
    angle_int = atleast_1d(angle_int)

    nfreq = len(freq)
    nangle = len(angle)
    nfreq_int = len(freq_int)
    nangle_int = len(angle_int)

    angle = angle % (360)
    angle_int = angle_int % (360)

    func_int = np.zeros((nfreq_int, nangle_int), dtype=func.dtype)

    delta_angle = _direction_bin(angle, wrap=360, kind="forward")

    angle_indices = np.zeros((2, nangle_int), dtype=np.int32)

    angle_indices[1, :] = np.searchsorted(angle, angle_int, side="right") % nangle
    angle_indices[0, :] = (angle_indices[1, :] - 1) % nangle

    # Calulate the weights for the angle interpolation, ensure it works with wrapping

    angle_diff = (angle_int - angle[angle_indices[0, :]] + 180) % (360) - 180
    angle_weights = angle_diff / delta_angle[angle_indices[0, :]]

    freq_indices = np.searchsorted(freq, freq_int, side="right")

    for i in range(nfreq_int):
        ifu = freq_indices[i]
        if ifu == 0 or ifu == nfreq:
            continue

        ifd = ifu - 1
        delta_freq = freq[ifu] - freq[ifd]
        freq_weight_lower = (freq[ifu] - freq_int[i]) / delta_freq
        freq_weight_upper = 1 - freq_weight_lower

        for j in range(nangle_int):
            jau = angle_indices[1, j]
            jad = angle_indices[0, j]

            angle_weights_upper = angle_weights[j]
            angle_weights_lower = 1 - angle_weights_upper

            func_int[i, j] = (
                freq_weight_upper * angle_weights_upper * func[ifu, jau]
                + freq_weight_upper * angle_weights_lower * func[ifu, jad]
                + freq_weight_lower * angle_weights_upper * func[ifd, jau]
                + freq_weight_lower * angle_weights_lower * func[ifd, jad]
            )

    return func_int


@jit(**numba_default)
def _angle_bounds(k1, k2, theta):
    # Note here we require that k2 > k1
    # print(k2,k1)

    # maxval = min(1, k2 / k1)
    maxval = 1
    out = np.array(
        (
            (theta - np.arcsin(maxval), theta + np.arcsin(maxval)),
            (theta - np.arcsin(maxval) + np.pi, theta + np.arcsin(maxval) + np.pi),
        )
    )

    return out


@jit(**numba_default)
def integration_grids_and_delta_function_scaling(
    k1, k2, theta0, angle_step_radians, sign_index, nonlinear_options
):
    n = int(np.pi * 2 / angle_step_radians) + 1
    angle_step_radians = 2 * np.pi / n
    bin_size = np.rad2deg(angle_step_radians)

    if nonlinear_options.angle_integration_convention == "janssen":
        theta1 = np.linspace(0, 2 * np.pi - angle_step_radians, n)
        theta2 = theta0 - theta1
        delta_function_rescaling = np.ones_like(theta1)
    elif nonlinear_options.angle_integration_convention == "symmetric":
        theta = np.linspace(-np.pi, np.pi - angle_step_radians, n)
        theta1 = theta0 + theta / 2
        theta2 = theta0 - theta / 2
        delta_function_rescaling = np.ones_like(theta1)
    else:
        theta1 = np.linspace(0, 2 * np.pi - angle_step_radians, n)

        delta = (theta1 - theta0 + np.pi) % (2 * np.pi) - np.pi
        theta2 = -sign_index * np.asin(k1 / k2 * np.sin(delta)) + theta0

        if sign_index == 1:
            delta = theta1 - theta2
        else:
            delta = theta1 - theta2 - np.pi

        k0 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(delta))

        delta_function_rescaling = np.abs(k0**2 / (k2**2 + k1 * k2 * np.cos(delta)))
        delta_function_rescaling[np.isnan(delta_function_rescaling)] = 0.0

    return theta1, theta2, bin_size, delta_function_rescaling


@jit(**numba_default)
def estimate_bound_contribution_2d(
    angular_frequency_target,
    angle_degrees_target,
    angular_frequency,
    angles_degrees_coordinates,
    variance_density,
    sign_index,
    depth=np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
) -> float:
    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.

    _, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    if sign_index == 1:
        # For sum frequencies the limit is given by frequency_target - fmax >= 0
        fmax = angular_frequency_target / 2
    else:
        # For difference frequencies the limit is given by frequency_target + fmax <= max(frequency)
        fmax = np.max(angular_frequency) - angular_frequency_target

    # Component 1
    # ----------------
    # angular_freq_component1 = np.arange(0, fmax, np.min(np.diff(angular_frequency)))
    df = np.min(np.diff(angular_frequency))
    angular_freq_component1 = np.linspace(0, fmax, int(fmax / df) + 1)

    wavenumber1_component1 = inverse_intrinsic_dispersion_relation(
        angular_freq_component1, depth
    )

    # Component 2
    # ----------------
    angular_freq_component2 = (
        angular_frequency_target - sign_index * angular_freq_component1
    )
    wavenumber_component2 = inverse_intrinsic_dispersion_relation(
        angular_freq_component2, depth
    )

    angle_radians_target = np.deg2rad(angle_degrees_target)
    angle_bin_degrees = np.min(_direction_bin(angles_degrees_coordinates, wrap=360))
    angle_bin_radians = np.deg2rad(angle_bin_degrees)
    angle_radians_coordinates = np.deg2rad(angles_degrees_coordinates)

    # Integrate
    # ----------------
    nfreq = len(angular_freq_component1)
    sum = 0.0
    for ifreq in range(nfreq):
        # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints and
        # df inbetween.
        iu = min(ifreq + 1, nfreq - 1)
        il = max(ifreq - 1, 0)
        delta_freq = (angular_freq_component1[iu] - angular_freq_component1[il]) / 2.0

        # Compoment 1 wavenumber, components and angular frequency
        w1 = angular_freq_component1[ifreq]
        k1 = wavenumber1_component1[ifreq]
        k2 = wavenumber_component2[ifreq]
        w2 = sign_index * angular_freq_component2[ifreq]

        (
            theta1,
            theta2,
            intergration_bin,
            delta_function_scaling,
        ) = integration_grids_and_delta_function_scaling(
            k1,
            k2,
            angle_radians_target,
            angle_bin_radians,
            sign_index,
            nonlinear_options,
        )

        kx1 = k1 * np.cos(theta1)
        ky1 = k1 * np.sin(theta1)

        # Compoment 2 wavenumber, components and angular frequency. The sign index is used to determine if we are
        # calculating the sum (sign_index=1) or difference (sign_index=-1) interactions.
        kx2 = sign_index * k2 * np.cos(theta2)
        ky2 = sign_index * k2 * np.sin(theta2)
        e2 = bilinear_interpolation(
            angular_frequency,
            angles_degrees_coordinates,
            variance_density,
            np.abs(w2),
            np.rad2deg(theta2),
        )

        e1 = bilinear_interpolation(
            angular_frequency,
            angles_degrees_coordinates,
            variance_density,
            np.abs(w1),
            np.rad2deg(theta1),
        )

        if nonlinear_options.reference_frame == "eulerian":
            interaction_coef = _second_order_surface_elevation(
                w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, physics_options.grav
            )
        elif nonlinear_options.reference_frame == "lagrangian":
            interaction_coef = _second_order_lagrangian_surface_elevation(
                w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, physics_options.grav
            )
        else:
            raise Exception("Unknown kind must be one of 'eulerian', 'lagrangian'")

        sum += (
            np.sum(
                e1
                * e2
                * interaction_coef**2
                * intergration_bin
                * delta_function_scaling
            )
            * delta_freq
        )
    return 2 * sum


@jit(**numba_default)
def _point_bound_surface_wave_spectrum_2d(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: float = np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_angular_frequency: intrinsic frequency of the variance density spectrum (rad/s)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2 s/rad/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    number_of_frequencties = len(intrinsic_angular_frequency)
    number_of_directions = len(angle_degrees)
    nonlinear_wave_spectrum = np.zeros(
        (number_of_frequencties, number_of_directions), dtype=variance_density.dtype
    )

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    if nonlinear_options.include_bound_waves:
        if (
            nonlinear_options.include_sum_interactions
            and nonlinear_options.include_difference_interactions
        ):
            sign_indices = np.array((1, -1))
        elif nonlinear_options.include_sum_interactions:
            sign_indices = np.array((1,))
        elif nonlinear_options.include_difference_interactions:
            sign_indices = np.array((-1,))
        else:
            sign_indices = np.array((), dtype=np.int64)

        for i in range(number_of_frequencties):
            for j in range(number_of_directions):
                for sign_index in sign_indices:
                    nonlinear_wave_spectrum[i, j] += estimate_bound_contribution_2d(
                        intrinsic_angular_frequency[i],
                        angle_degrees[j],
                        intrinsic_angular_frequency,
                        angle_degrees,
                        variance_density,
                        sign_index,
                        depth,
                        output,
                        nonlinear_options,
                        physics_options,
                    )

    if nonlinear_options.include_nonlinear_amplitude_correction:
        for i in range(number_of_frequencties):
            for j in range(number_of_directions):
                nonlinear_wave_spectrum[
                    i, j
                ] += estimate_nonlinear_amplitude_correction_2d(
                    intrinsic_angular_frequency[i],
                    angle_degrees[j],
                    intrinsic_angular_frequency,
                    angle_degrees,
                    variance_density,
                    depth,
                    output,
                    nonlinear_options,
                    physics_options,
                )

    if nonlinear_options.include_nonlinear_dispersion:
        stokes_correction = stokes_dispersive_correction(
            intrinsic_angular_frequency,
            angle_degrees,
            variance_density,
            depth,
            output,
            nonlinear_options,
            physics_options,
        )
        for i in range(number_of_frequencties):
            for j in range(number_of_directions):
                nonlinear_wave_spectrum[i, j] += stokes_correction[i, j]

    return nonlinear_wave_spectrum


@jit(**numba_default)
def nonlinear_wave_spectra_2d(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
    progress_bar=None,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_angular_frequency: intrinsic frequency of the variance density spectrum (rad Hz)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2 s/rad/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    dims = variance_density.shape

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    nspec = int(np.prod(np.array(dims[:-2])))
    nfreq = len(intrinsic_angular_frequency)
    ndir = len(angle_degrees)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    spectra = np.zeros(
        (nspec, nfreq, ndir),
    )
    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

        spectra[i, :, :] = _point_bound_surface_wave_spectrum_2d(
            intrinsic_angular_frequency,
            angle_degrees,
            variance_density[i, :, :],
            depth[i],
            output,
            nonlinear_options,
            physics_options,
        )

    return spectra.reshape(dims)
