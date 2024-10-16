from ._perturbation_theory_coeficients import (
    _second_order_surface_elevation,
    _second_order_lagrangian_surface_elevation,
)
from linearwavetheory import inverse_intrinsic_dispersion_relation
import numpy as np
from linearwavetheory._numba_settings import numba_default, numba_default_parallel
from numba import jit, prange
from linearwavetheory._array_shape_preprocessing import atleast_1d
from linearwavetheory._utils import _direction_bin


@jit(**numba_default_parallel)
def bound_wave_spectra_1d(
    intrinsic_frequency_hz: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: float = np.inf,
    kind: str = "eulerian",
    contributions: str = "all",
    grav=9.81,
    progress_bar=None,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_frequency_hz: intrinsic frequency of the variance density spectrum (Hz)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2/Hz/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    dims = variance_density.shape

    nspec = int(np.prod(np.array(dims[:-2])))
    nfreq = len(intrinsic_frequency_hz)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    output = np.zeros(
        (nspec, nfreq),
    )
    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

        output[i] = bound_wave_spectrum_1d(
            intrinsic_frequency_hz,
            angle_degrees,
            variance_density[i, :, :],
            depth=depth[i],
            kind=kind,
            contributions=contributions,
            grav=grav,
        )

    return output.reshape(dims[:-1])


@jit(**numba_default)
def bound_wave_spectrum_1d(
    intrinsic_frequency_hz: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: float = np.inf,
    kind: str = "eulerian",
    contributions: str = "all",
    grav=9.81,
):
    """
    Calculate the bound wave spectrum for a given variance density.
    :param intrinsic_frequency_hz: intrinsic frequency of the variance density spectrum (Hz)
    :param angle_degrees: angles of the variance density spectrum (degrees)
    :param variance_density: variance density spectrum as a function of frequencies and direction (m^2/Hz/deg).
        Directions are assumed to be the trailing axis.
    :param depth: depth of the water (m)
    :param kind: kind of observation, either 'eulerian' or 'lagrangian'
    :param contributions: what contributions to calculate, either 'all', 'sum' or 'difference'
    :return: 1d bound wave spectrum as a function of the intrinsic frequencies (m^2/Hz)
    """
    number_of_frequencties = len(intrinsic_frequency_hz)
    bound_wave_spectrum = np.zeros(
        (number_of_frequencties,), dtype=variance_density.dtype
    )

    if contributions == "all":
        sign_indices = np.array((1, -1))
    elif contributions == "sum":
        sign_indices = np.array((1,))
    elif contributions == "difference":
        sign_indices = np.array((-1,))
    else:
        raise Exception(
            f"Unknown contributions {contributions}, must be one of 'all', 'sum', 'difference'"
        )

    for sign_index in sign_indices:
        for i in range(number_of_frequencties):
            bound_wave_spectrum[i] += estimate_bound_contribution_1d(
                intrinsic_frequency_hz[i],
                intrinsic_frequency_hz,
                angle_degrees,
                variance_density,
                sign_index,
                depth=depth,
                kind=kind,
                grav=grav,
            )
    return bound_wave_spectrum


@jit(**numba_default)
def estimate_bound_contribution_1d(
    frequency_target,
    frequency_hz_coordinates,
    angles_degrees_coordinates,
    variance_density,
    sign_index,
    depth=np.inf,
    kind="eulerian",
    grav=9.81,
) -> float:

    # Preliminaries
    # ----------------
    # Calculate the integration frequencies. Note that depending on if we are calculating the sum or difference
    # interactions the frequency range is different.
    if sign_index == 1:
        # For sum frequencies the limit is given by frequency_target - fmax >= 0
        fmax = frequency_target
        factor = 1
    else:
        # For difference frequencies the limit is given by frequency_target + fmax <= max(frequency)
        fmax = np.max(frequency_hz_coordinates) - frequency_target
        factor = 2

    # Component 1
    # ----------------
    freq_component1 = np.arange(0, fmax, np.min(np.diff(frequency_hz_coordinates)))
    variance_density_component1 = bilinear_interpolation(
        frequency_hz_coordinates,
        angles_degrees_coordinates,
        variance_density,
        freq_component1,
        angles_degrees_coordinates,
    )
    angular_freq_component1 = freq_component1 * 2.0 * np.pi
    wavenumber1_component1 = inverse_intrinsic_dispersion_relation(
        angular_freq_component1, depth
    )

    # Component 2
    # ----------------
    freq_component2 = frequency_target - sign_index * freq_component1
    variance_density_component2 = bilinear_interpolation(
        frequency_hz_coordinates,
        angles_degrees_coordinates,
        variance_density,
        freq_component2,
        angles_degrees_coordinates,
    )
    angular_freq_component2 = freq_component2 * 2.0 * np.pi
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
    nfreq = len(freq_component1)
    nangle = len(angles_degrees_coordinates)
    for ifreq in range(nfreq):
        # Use trapezoindal rule to integrate over the frequency. This expression reduces to 0.5*df at endpoints and
        # df inbetween.
        iu = min(ifreq + 1, nfreq - 1)
        il = max(ifreq - 1, 0)
        delta_freq = (freq_component1[iu] - freq_component1[il]) / 2.0

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
            if kind == "eulerian":
                interaction_coef = _second_order_surface_elevation(
                    w1, k1, kx1[iangle], ky1[iangle], w2, k2, kx2, ky2, depth, grav
                )
            elif kind == "lagrangian":
                interaction_coef = _second_order_lagrangian_surface_elevation(
                    w1, k1, kx1[iangle], ky1[iangle], w2, k2, kx2, ky2, depth, grav
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
