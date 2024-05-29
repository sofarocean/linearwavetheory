from numba import vectorize, float64, float32
from linearwavetheory._numba_settings import numba_default_vectorize
import numpy as np

_interaction_signatures = [
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    float32(
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
    ),
    float32(
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float64,
    ),
    float32(
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float64,
        float64,
    ),
]


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_potential(w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, grav):
    """
    Calculate the second order potential interaction coefficient for two wave components with wavenumber magnitude
    and direction k_mag1, k_dir1 and k_mag2, k_dir2 respectively. The interaction coefficient is calculated for a
    given water depth.

    See Smit et al. (2017), Nonlinear Wave Kinematics near the Ocean Surface, J. Phys. Oceanogr.

    Parameters
    ----------
    :param w1: frequency of the first wave component in radians per second
    :param k1: wavenumber magnitude of the first wave component in radians per meter
    :param kx1: x-component of the wavenumber direction of the first wave component
    :param ky1: y-component of the wavenumber direction of the first wave component
    :param w2: frequency of the second wave component in radians per second
    :param k2: wavenumber magnitude of the second wave component in radians per meter
    :param kx2: x-component of the wavenumber direction of the second wave component
    :param ky2: y-component of the wavenumber direction of the second wave component
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order potential interaction coefficient

    """
    inner_product = kx1 * kx2 + ky1 * ky2

    wsum = w1 + w2
    ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)

    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w12 = np.sqrt(grav * ksum)
    else:
        w12 = np.sqrt(grav * ksum * np.tanh(ksum * depth))

    if w12 == wsum:
        return 0.0

    elif wsum == 0:
        return (
            grav**2
            / (w12**2 - wsum**2)
            * (+(k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2))
        )

    else:
        return (
            grav**2
            * wsum
            / (w12**2 - wsum**2)
            * (
                inner_product / w1 / w2
                - (w1 * w2 + w1**2 + w2**2) / (2 * grav**2)
                + (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2 * (w1 + w2))
            )
        )


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_surface_elevation(w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, grav):
    """
    Calculate the second order Eulerian surface elevation interaction coefficient for two wave components. For details,
    see e.g. See Smit et al. (2017), Nonlinear Wave Kinematics near the Ocean Surface, J. Phys. Oceanogr.

    :param w1: frequency of the first wave component in radians per second
    :param k1: wavenumber magnitude of the first wave component in radians per meter
    :param kx1: x-component of the wavenumber direction of the first wave component
    :param ky1: y-component of the wavenumber direction of the first wave component
    :param w2: frequency of the second wave component in radians per second
    :param k2: wavenumber magnitude of the second wave component in radians per meter
    :param kx2: x-component of the wavenumber direction of the second wave component
    :param ky2: y-component of the wavenumber direction of the second wave component
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order Eulerian surface elevation interaction coefficient
    """

    inner_product = kx1 * kx2 + ky1 * ky2

    wsum = w1 + w2
    ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)
    if ksum == 0:
        w12 = 0
    else:
        w12 = np.sqrt(grav * ksum * np.tanh(ksum * depth))

    if w12 == wsum:
        resonance_factor = 0.0
    else:
        resonance_factor = wsum / (w12**2 - wsum**2)

    term1 = -grav * (wsum * resonance_factor + 0.5) * inner_product / w1 / w2
    term2 = (1 + wsum * resonance_factor) * (w1 * w2 + w1**2 + w2**2) / (2 * grav)
    term3 = -grav * resonance_factor * (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2)
    return term1 + term2 + term3


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_lagrangian_surface_elevation(
    w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, grav
):
    """
    Calculate the second order Lagrangian surface elevation interaction coefficient for two wave components. For
    details, see eqs 7 and 8 in:

    Herbers, T. H. C., & Janssen, T. T. (2016). Lagrangian surface wave motion and Stokes drift fluctuations.
    Journal of Physical Oceanography, 46(4), 1009-1021.

    :param w1: frequency of the first wave component in radians per second
    :param k1: wavenumber magnitude of the first wave component in radians per meter
    :param kx1: x-component of the wavenumber direction of the first wave component
    :param ky1: y-component of the wavenumber direction of the first wave component
    :param w2: frequency of the second wave component in radians per second
    :param k2: wavenumber magnitude of the second wave component in radians per meter
    :param kx2: x-component of the wavenumber direction of the second wave component
    :param ky2: y-component of the wavenumber direction of the second wave component
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order Lagrangian surface elevation interaction coefficient
    """
    inner_product = kx1 * kx2 + ky1 * ky2

    coef_second_order_eulerian_surface_elevation = _second_order_surface_elevation(
        w1, k1, kx1, kx2, w2, k2, kx2, ky2, depth, grav
    )

    return (
        coef_second_order_eulerian_surface_elevation
        - grav * inner_product / w1**2 / w2**2 / 2 * (w1**2 + w2**2)
    )
