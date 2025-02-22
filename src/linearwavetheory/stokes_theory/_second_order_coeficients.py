import numpy as np
from numba import float64, float32, vectorize

from linearwavetheory._numba_settings import numba_default_vectorize

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

_interaction_signatures_velocity = [
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
        w12 = grav * ksum * np.tanh(ksum * depth)

    if w1 * w2 < 0.0 and wsum == 0.0 and ksum == 0.0:
        # Self - interaction. Undefined for "naked" potential as it has pole here
        coef = 0.0
    else:
        coef = (
            grav**2
            / ((w12 - wsum**2))
            * (
                wsum * inner_product / w1 / w2
                - wsum * (w1 * w2 + w1**2 + w2**2) / (2 * grav**2)
                + (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2)
            )
        )
    return coef


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

    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    inner_product = kx1 * kx2 + ky1 * ky2

    wsum = w1 + w2
    ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)

    if w1 * w2 < 0.0 and wsum == 0.0 and ksum == 0.0:
        # self_interaction = True. Calculate the setdown wave for a self-interaction as limit of vanishinign mismatch
        if np.isinf(depth):
            coef = 0.0
        else:
            cg = np.abs(w1) / k1 * (0.5 + k1 * depth / np.sinh(2.0 * k1 * depth))
            coef = (
                -grav
                * cg**2
                / (grav * depth - cg**2)
                * (
                    np.abs(k1) / cg / np.abs(w1)
                    - k1**2 / 2 / w1**2
                    + k1**2 / w1**2
                    - w1**2 / 2 / grav**2
                )
                - grav * k1**2 / w1**2 / 2
                + (w1**2) / (2 * grav)
            )
    else:
        # self_interaction = False. Use the full expression
        w12 = np.sqrt(grav * ksum * np.tanh(ksum * depth))
        resonance_factor = wsum / (w12**2 - wsum**2)

        term1 = -grav * (wsum * resonance_factor + 0.5) * inner_product / w1 / w2
        term2 = (
            (1 + wsum * resonance_factor) * (w1 * w2 + w1**2 + w2**2) / (2 * grav)
        )
        term3 = -grav * resonance_factor * (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2)

        coef = term1 + term2 + term3

    return coef


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
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    inner_product = kx1 * kx2 + ky1 * ky2

    coef_second_order_eulerian_surface_elevation = _second_order_surface_elevation(
        w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, grav
    )
    return (
        coef_second_order_eulerian_surface_elevation
        - grav * inner_product / w1**2 / w2**2 / 2 * (w1**2 + w2**2)
    )


@vectorize(_interaction_signatures_velocity, **numba_default_vectorize)
def _second_order_horizontal_velocity(
    w1, k1, kx1, ky1, w2, k2, kx2, ky2, direction, depth, grav
):
    """
    Calculate the spatial derivative of the second order potential interaction coefficient for two wave components with
    wavenumber magnitude and direction k_mag1, k_dir1 and k_mag2, k_dir2 respectively.
    The interaction coefficient is calculated for a given water depth. To note; this function is usefull to ensure
    the proper limit of the potential interaction coefficient in the limit of vanishing mismatch
    (the mean flow response).

    The function will return the interaction coeficient for Ux or Uy - i.e. the x or y component of the mean flow
    response - depending on the direction parameter (Ux for direction > 0. or Uy for direction < 0.)

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
    :param direction: x or y direction (x > 0. or y < 0.)
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order potential interaction coefficient

    """
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    inner_product = kx1 * kx2 + ky1 * ky2

    wsum = w1 + w2
    ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)

    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w12 = np.sqrt(grav * ksum)
    else:
        w12 = grav * ksum * np.tanh(ksum * depth)

    resonant_mismatch = w12 - wsum**2

    if w1 * w2 < 0.0 and wsum == 0.0 and ksum == 0.0:
        # Check for a self-difference interaction limit

        if np.isinf(depth):
            coef = 0.0
        else:
            # The mean flow will be "directed" along the direction of the wave vector of the positive component
            if direction >= 0.0:
                if w1 > 0.0:
                    kdir = kx1 / k1
                else:
                    kdir = kx2 / k2
            else:
                if w1 > 0.0:
                    kdir = ky1 / k1
                else:
                    kdir = ky2 / k2

            # In the limit of vanishing mismatch - we need to account for the proper limiting behavior
            cg = np.abs(w1) / k1 * (0.5 + k1 * depth / np.sinh(2.0 * k1 * depth))
            coef = -kdir * (
                grav**2
                * cg
                / (grav * depth - cg**2)
                * (
                    k1**2 / w1**2
                    - (w1**2) / (2 * grav**2)
                    - (k1**2 - 2 * k1 * np.abs(w1) / cg) / (2 * w1**2)
                )
            )
    else:

        if direction >= 0.0:
            kdir = kx1 + kx2
        else:
            kdir = ky1 + ky2

        coef = (
            -(grav**2)
            * kdir
            / (resonant_mismatch)
            * (
                wsum * inner_product / w1 / w2
                - wsum * (w1 * w2 + w1**2 + w2**2) / (2 * grav**2)
                + (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2)
            )
        )
    return coef


@vectorize(_interaction_signatures, **numba_default_vectorize)
def _second_order_vertical_velocity(w1, k1, kx1, ky1, w2, k2, kx2, ky2, depth, grav):
    """
    Calculate the vertical derivative of the second order potential interaction coefficient for two wave components with
    wavenumber magnitude and direction k_mag1, k_dir1 and k_mag2, k_dir2 respectively.
    The interaction coefficient is calculated for a given water depth. To note; this function is usefull to ensure
    the proper limit of the potential interaction coefficient in the limit of vanishing mismatch    .

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
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    inner_product = kx1 * kx2 + ky1 * ky2

    wsum = w1 + w2
    ksum = np.sqrt((kx1 + kx2) ** 2 + (ky1 + ky2) ** 2)

    if np.isinf(depth):
        # Note if depth is infinite and ksum == 0, then the tanh in undefined.
        w12 = np.sqrt(grav * ksum)
    else:
        w12 = grav * ksum * np.tanh(ksum * depth)

    if w1 * w2 < 0.0 and wsum == 0.0 and ksum == 0.0:
        # Check for a self-difference interaction limit
        coef = 0.0
    else:
        coef = (
            grav**2
            * ksum
            * np.tanh(ksum * depth)
            / (w12 - wsum**2)
            * (
                wsum * inner_product / w1 / w2
                - wsum * (w1 * w2 + w1**2 + w2**2) / (2 * grav**2)
                + (k1**2 * w2 + k2**2 * w1) / (2 * w1 * w2)
            )
        )
    return coef


@vectorize(_interaction_signatures_velocity, **numba_default_vectorize)
def _second_order_horizontal_lagrangian_surface_perturbation(
    w1, k1, kx1, ky1, w2, k2, kx2, ky2, direction, depth, grav
):
    """
    This is the second-order particle displacement in a frame of reference following the mean Lagrangian flow. As a
    conseqence, the mean contributions (w1+w2) to the flow are absorbed into the phase function and are excluded.

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
    :param direction: x or y direction (x > 0. or y < 0.)
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order potential interaction coefficient

    """
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    inner_product = kx1 * kx2 + ky1 * ky2

    if direction >= 0.0:
        k_dir_1 = kx1
        k_dir_2 = kx2
    else:
        k_dir_1 = ky1
        k_dir_2 = ky2

    if np.abs(w1 + w2) > 0.0:
        second_order_horizontal_surface_displacement = (
            -(grav**2)
            * inner_product
            * (w2 * k_dir_2 + w1 * k_dir_1)
            / 2
            / w1**2
            / w2**2
            + w2 * k_dir_2 / 2
            + w1 * k_dir_1 / 2
            + _second_order_horizontal_velocity(
                w1, k1, kx1, ky1, w2, k2, kx2, ky2, direction, depth, grav
            )
        ) / (w1 + w2)

    else:
        second_order_horizontal_surface_displacement = 0.0

    return second_order_horizontal_surface_displacement


@vectorize(_interaction_signatures_velocity, **numba_default_vectorize)
def _second_order_lagrangian_horizontal_velocity(
    w1, k1, kx1, ky1, w2, k2, kx2, ky2, direction, depth, grav
):
    """

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
    :param direction: x or y direction (x > 0. or y < 0.)
    :param depth: depth of the water in meters
    :param grav: gravitational acceleration in meters per second squared
    :return: the second order potential interaction coefficient

    """
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    eulerian = _second_order_horizontal_velocity(
        w1, k1, kx1, ky1, w2, k2, kx2, ky2, direction, depth, grav
    )

    inner_product = kx1 * kx2 + ky1 * ky2

    if direction >= 0.0:
        k_dir_1 = kx1
        k_dir_2 = kx2
    else:
        k_dir_1 = ky1
        k_dir_2 = ky2

    return (
        eulerian
        - (grav**2)
        * inner_product
        * (w2 * k_dir_2 + w1 * k_dir_1)
        / 2
        / w1**2
        / w2**2
        + w2 * k_dir_2 / 2
        + w1 * k_dir_1 / 2
    )
