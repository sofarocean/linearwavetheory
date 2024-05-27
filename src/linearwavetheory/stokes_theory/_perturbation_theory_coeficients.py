from numba import vectorize, float64, float32
from linearwavetheory._numba_settings import numba_default_vectorize
import numpy as np

_interaction_signatures = [
    float64(float64, float64, float64, float64, float64, float64, float64, float64),
    float32(float32, float32, float32, float32, float32, float32, float32, float32),
    float32(float32, float32, float32, float32, float32, float32, float32, float64),
    float32(float32, float32, float32, float32, float32, float32, float64, float64),
]


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_potential(
    k_mag1, k_dir1, sign_index1, k_mag2, k_dir2, sign_index2, depth, grav
):
    """
    Calculate the second order potential interaction coefficient for two wave components with wavenumber magnitude
    and direction k_mag1, k_dir1 and k_mag2, k_dir2 respectively. The interaction coefficient is calculated for a
    given water depth.

    See Smit et al. (2017), Nonlinear Wave Kinematics near the Ocean Surface, J. Phys. Oceanogr.

    Parameters
    ----------
    k_mag1 : float
        Wavenumber magnitude of the first wave component in radian per meter
    k_dir1 : float
        Wavenumber direction of the first wave component in radian
    k_mag2 : float
        Wavenumber magnitude of the second wave component in radian per meter
    k_dir2 : float
        Wavenumber direction of the second wave component in radian
    depth : float
        Water depth
    grav : float
        Gravitational acceleration

    Returns
    -------
    float
        The second order potential interaction coefficient
    """
    inner_product = (
        sign_index1 * sign_index2 * k_mag1 * k_mag2 * np.cos(k_dir1 - k_dir2)
    )

    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w1 = np.sqrt(grav * k_mag1) * sign_index1
        w2 = np.sqrt(grav * k_mag2) * sign_index2
    else:
        w1 = np.sqrt(grav * k_mag1 * np.tanh(k_mag1 * depth)) * sign_index1
        w2 = np.sqrt(grav * k_mag2 * np.tanh(k_mag2 * depth)) * sign_index2

    wsum = w1 + w2
    ksum = np.sqrt(
        (k_mag1 * np.cos(k_dir1) * sign_index1 + k_mag2 * np.cos(k_dir2) * sign_index2)
        ** 2
        + (
            k_mag1 * np.sin(k_dir1) * sign_index1
            + k_mag2 * np.sin(k_dir2) * sign_index2
        )
        ** 2
    )

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
            * (+(k_mag1**2 * w2 + k_mag2**2 * w1) / (2 * w1 * w2))
        )

    else:
        return (
            grav**2
            * wsum
            / (w12**2 - wsum**2)
            * (
                inner_product / w1 / w2
                - (w1 * w2 + w1**2 + w2**2) / (2 * grav**2)
                + (k_mag1**2 * w2 + k_mag2**2 * w1) / (2 * w1 * w2 * (w1 + w2))
            )
        )


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_surface_elevation(
    k_mag1, k_dir1, sign_index1, k_mag2, k_dir2, sign_index2, depth, grav
):
    """
    Calculate the second order Eulerian surface elevation interaction coefficient for two wave components. For details,
    see e.g. See Smit et al. (2017), Nonlinear Wave Kinematics near the Ocean Surface, J. Phys. Oceanogr.

    :param k_mag1: wavenumber magnitude of the first wave component in radian per meter
    :param k_dir1: wavenumber direction of the first wave component in radian
    :param k_mag2: wavenumber magnitude of the second wave component in radian per meter
    :param k_dir2: wavenumber direction of the second wave component in radian
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return:
    """

    inner_product = (
        k_mag1 * k_mag2 * np.cos(k_dir1 - k_dir2) * sign_index1 * sign_index2
    )
    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w1 = np.sqrt(grav * k_mag1) * sign_index1
        w2 = np.sqrt(grav * k_mag2) * sign_index2
    else:
        w1 = np.sqrt(grav * k_mag1 * np.tanh(k_mag1 * depth)) * sign_index1
        w2 = np.sqrt(grav * k_mag2 * np.tanh(k_mag2 * depth)) * sign_index2

    coef_second_order_potential = _second_order_potential(
        k_mag1, k_dir1, sign_index1, k_mag2, k_dir2, sign_index2, depth, grav
    )
    return (
        -(w1 + w2) / grav * coef_second_order_potential
        + (w1 * w2 + w1**2 + w2**2) / (2 * grav)
        - grav * inner_product / w1 / w2 / 2
    )


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_lagrangian_surface_elevation(
    k_mag1, k_dir1, sign_index1, k_mag2, k_dir2, sign_index2, depth, grav
):
    """
    Calculate the second order Lagrangian surface elevation interaction coefficient for two wave components. For
    details, see eqs 7 and 8 in:

    Herbers, T. H. C., & Janssen, T. T. (2016). Lagrangian surface wave motion and Stokes drift fluctuations.
    Journal of Physical Oceanography, 46(4), 1009-1021.

    :param k_mag1:
    :param k_dir1:
    :param k_mag2:
    :param k_dir2:
    :param depth:
    :param grav:
    :return:
    """
    inner_product = (
        k_mag1 * k_mag2 * np.cos(k_dir1 - k_dir2) * sign_index1 * sign_index2
    )
    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w1 = np.sqrt(grav * k_mag1) * sign_index1
        w2 = np.sqrt(grav * k_mag2) * sign_index2
    else:
        w1 = np.sqrt(grav * k_mag1 * np.tanh(k_mag1 * depth)) * sign_index1
        w2 = np.sqrt(grav * k_mag2 * np.tanh(k_mag2 * depth)) * sign_index2

    coef_second_order_eulerian_surface_elevation = _second_order_surface_elevation(
        k_mag1, k_dir1, sign_index1, k_mag2, k_dir2, sign_index2, depth, grav
    )

    return (
        coef_second_order_eulerian_surface_elevation
        - grav * inner_product / w1**2 / w2**2 / 2 * (w1**2 + w2**2)
    )
