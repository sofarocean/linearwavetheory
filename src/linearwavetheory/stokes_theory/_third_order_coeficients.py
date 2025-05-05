try:
    from numba import vectorize, float64, bool
except ImportError:
    from numba import vectorize, float64, boolean

    bool = boolean

from linearwavetheory._numba_settings import numba_default_vectorize
import numpy as np

from linearwavetheory.stokes_theory._second_order_coeficients import (
    _second_order_potential,
    _second_order_surface_elevation,
    _second_order_horizontal_velocity,
    _second_order_vertical_velocity,
    _second_order_horizontal_lagrangian_surface_perturbation,
)

_interaction_signatures_non_symmetric = [
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
        float64,
        float64,
        float64,
        bool,
        bool,
        bool,
    ),
]

_interaction_signatures_symmetric = [
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
        bool,
        bool,
        bool,
    ),
]


@vectorize(_interaction_signatures_non_symmetric, **numba_default_vectorize)
def _third_order_dispersion_correction_non_symmetric(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    w3,
    k3,
    kx3,
    ky3,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):

    # read this as "inner product between k1 and k2 plus k3)
    inner_product_12p3 = kx1 * (kx2 + kx3) + ky1 * (ky2 + ky3)

    inner_product_k23 = kx2 * kx3 + ky2 * ky3

    second_order_surface_elevation_23 = _second_order_surface_elevation(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    second_order_potential_23 = _second_order_potential(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    second_order_w_23 = _second_order_vertical_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    ux_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, 1.0, depth, grav
    )

    uy_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, -1.0, depth, grav
    )

    k23 = np.sqrt((kx2 + kx3) ** 2 + (ky2 + ky3) ** 2)
    if w2 * w3 < 0.0 and w2 + w3 == 0 and k23 == 0:
        if wave_driven_setup_included_in_mean_depth:
            second_order_surface_elevation_23 = 0.0

        if wave_driven_flow_included_in_mean_flow:
            ux_23 = 0.0
            uy_23 = 0.0

    term_w = second_order_w_23 * (w1 * (w1 + w2 + w3) + (w2 + w3) ** 2 + w1 * (w2 + w3))

    # SW
    term_wz = -grav * k23**2 * second_order_potential_23

    # SW
    term_u = grav * (kx1 * ux_23 + ky1 * uy_23) * (1.0 + (w1 + w2 + w3) / w1)

    term_c = second_order_surface_elevation_23 * (
        grav**2 * inner_product_12p3 / w1
        - w1**3
        + grav**2 * k1**2 / w1
        - w1**2 * (w2 + w3)
    )

    term_tb = (
        grav
        / 2
        * (
            inner_product_k23
            * ((w2 + w3) + (w1 + w2 + w3) * (w3**2 + w2**2) / w2 / w3)
            - (w1 + w2 + w3) * (w3**2 * k2**2 + w2**2 * k3**2) / (w2 * w3)
            - w2 * k3**2
            - w3 * k2**2
        )
    )

    return term_c + term_u + term_w + term_wz + term_tb


@vectorize(_interaction_signatures_non_symmetric, **numba_default_vectorize)
def _third_order_dispersion_correction_non_symmetric_sigma_coordinates(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    w3,
    k3,
    kx3,
    ky3,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):

    # read this as "inner product between k1 and k2 plus k3)
    inner_product_12p3 = kx1 * (kx2 + kx3) + ky1 * (ky2 + ky3)

    inner_product_k23 = kx2 * kx3 + ky2 * ky3
    inner_product_k12 = kx1 * kx2 + ky1 * ky2
    inner_product_k13 = kx1 * kx3 + ky1 * ky3

    second_order_surface_elevation_23 = _second_order_surface_elevation(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    second_order_potential_23 = _second_order_potential(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    ux_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, 1.0, depth, grav
    )

    uy_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, -1.0, depth, grav
    )

    k23 = np.sqrt((kx2 + kx3) ** 2 + (ky2 + ky3) ** 2)
    if w2 * w3 < 0.0 and w2 + w3 == 0 and k23 == 0:
        if wave_driven_setup_included_in_mean_depth:
            second_order_surface_elevation_23 = 0.0

        if wave_driven_flow_included_in_mean_flow:
            ux_23 = 0.0
            uy_23 = 0.0

    # SW
    term_wz = (
        -(
            grav * k23**2
            + (w2 + w3) / depth
            - (w2 + w3) ** 2 * (w1 + w2 + w3) ** 2 / grav
        )
        * second_order_potential_23
    )

    # SW
    term_u = grav * (kx1 * ux_23 + ky1 * uy_23) * (1.0 + (w1 + w2 + w3) / w1)

    term_c = second_order_surface_elevation_23 * (
        grav**2 * inner_product_12p3 / w1
        + grav**2 * k1**2 / w1
        - w1**2 * (w1 + w2 + w3)
        - grav / depth * (w2 + w3)
    )

    term_tb = (
        grav
        / 2
        * (
            (inner_product_k12 * w2 + inner_product_k13 * w3)
            * ((w1 + w2 + w3) / w1 + 1)
            + k1**2 * w1
            - k1**2 * (w1 + w2 + w3) ** 2 / w1
        )
    )

    term_tbd = (
        grav**2 / depth * ((w2 + w3) * inner_product_k23 / w2 / w3 / 2 + k1**2 / w1)
    )

    return term_c + term_u + term_wz + term_tb + term_tbd


@vectorize(_interaction_signatures_non_symmetric, **numba_default_vectorize)
def _third_order_amplitude_correction_non_symmetric(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    w3,
    k3,
    kx3,
    ky3,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):
    # read this as "inner product between k1 and k2 plus k3)
    inner_product_12p3 = kx1 * (kx2 + kx3) + ky1 * (ky2 + ky3)

    second_order_potential_23 = _second_order_potential(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    inner_product_k23 = kx2 * kx3 + ky2 * ky3

    second_order_surface_elevation_23 = _second_order_surface_elevation(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    second_order_w_23 = _second_order_vertical_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    ux_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, 1.0, depth, grav
    )

    uy_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, -1.0, depth, grav
    )

    k23 = np.sqrt((kx2 + kx3) ** 2 + (ky2 + ky3) ** 2)
    if w2 * w3 < 0.0 and w2 + w3 == 0 and k23 == 0:
        if wave_driven_setup_included_in_mean_depth:
            second_order_surface_elevation_23 = 0.0

        if wave_driven_flow_included_in_mean_flow:
            ux_23 = 0.0
            uy_23 = 0.0

    # DISP
    term_w = second_order_w_23 * (
        (w1 * (w1 + w2 + w3) + (w2 + w3) ** 2 + w1 * (w2 + w3))
        / 2
        / grav
        / (w1 + w2 + w3)
        - (w1 + w2 + w3) / grav
    )

    # SW
    term_wz = -grav * k23**2 * second_order_potential_23 / 2 / grav / (w1 + w2 + w3)

    # SW
    term_u = (
        grav
        * (kx1 * ux_23 + ky1 * uy_23)
        * (1.0 + (w1 + w2 + w3) / w1)
        / 2
        / grav
        / (w1 + w2 + w3)
        - (kx1 * ux_23 + ky1 * uy_23) / w1
    )

    term_c = (
        second_order_surface_elevation_23
        * (
            grav**2 * inner_product_12p3 / w1
            - w1**3
            + grav**2 * k1**2 / w1
            - w1**2 * (w2 + w3)
        )
        / 2
        / grav
        / (w1 + w2 + w3)
    ) + second_order_surface_elevation_23 * w1**2 / grav

    term_tb = (
        (
            grav
            / 2
            * (
                inner_product_k23
                * ((w2 + w3) + (w1 + w2 + w3) * (w3**2 + w2**2) / w2 / w3)
                - (w1 + w2 + w3) * (w3**2 * k2**2 + w2**2 * k3**2) / (w2 * w3)
                - w2 * k3**2
                - w3 * k2**2
            )
        )
        / 2
        / grav
        / (w1 + w2 + w3)
        - inner_product_k23 / 2 / w2 / w3 * (w2**2 + w3**2)
        + (w2**2 * k3**2 + w3**2 * k2**2) / 2 / w2 / w3
        + k1**2 / 2
    )

    if lagrangian:
        second_order_horizontal_surface_displacement_x = (
            _second_order_horizontal_lagrangian_surface_perturbation(
                w3, k3, kx3, ky3, w2, k2, kx2, ky2, 1.0, depth, grav
            )
        )

        second_order_horizontal_surface_displacement_y = (
            _second_order_horizontal_lagrangian_surface_perturbation(
                w3, k3, kx3, ky3, w2, k2, kx2, ky2, -1.0, depth, grav
            )
        )

        term_c += (
            -grav * inner_product_12p3 * second_order_surface_elevation_23 / w1**2
        )
        term_x = (
            -kx1 * second_order_horizontal_surface_displacement_x
            - ky1 * second_order_horizontal_surface_displacement_y
        )
        term_r = (
            grav**2
            / 2
            / w2**2
            / w3**2
            * (
                kx1**2 * kx2 * kx3
                + ky1**2 * ky2 * ky3
                + kx1 * ky1 * kx2 * ky3
                + kx1 * ky1 * kx3 * ky2
            )
        )
    else:
        term_x = 0.0
        term_r = 0.0

    return term_c + term_u + term_w + term_wz + term_tb + term_x + term_r


@vectorize(_interaction_signatures_non_symmetric, **numba_default_vectorize)
def _third_order_amplitude_correction_non_symmetric_sigma_coordinates(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    w3,
    k3,
    kx3,
    ky3,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):
    second_order_potential_23 = _second_order_potential(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    second_order_surface_elevation_23 = _second_order_surface_elevation(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    ux_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, 1.0, depth, grav
    )

    uy_23 = _second_order_horizontal_velocity(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, -1.0, depth, grav
    )

    k23 = np.sqrt((kx2 + kx3) ** 2 + (ky2 + ky3) ** 2)
    if w2 * w3 < 0.0 and w2 + w3 == 0 and k23 == 0:
        if wave_driven_setup_included_in_mean_depth:
            second_order_surface_elevation_23 = 0.0

        if wave_driven_flow_included_in_mean_flow:
            ux_23 = 0.0
            uy_23 = 0.0

    dispersion = (
        _third_order_dispersion_correction_non_symmetric_sigma_coordinates(
            w1,
            k1,
            kx1,
            ky1,
            w2,
            k2,
            kx2,
            ky2,
            w3,
            k3,
            kx3,
            ky3,
            depth,
            grav,
            wave_driven_setup_included_in_mean_depth,
            wave_driven_flow_included_in_mean_flow,
            lagrangian,
        )
        / (w1 + w2 + w3)
        / grav
        / 2
    )
    # To note the /grav/2 is because we add that factor in the symmetric implementation

    k1p2 = kx1 * kx2 + ky1 * ky2
    k1p3 = kx1 * kx3 + ky1 * ky3
    bernouilli = (
        -(kx1 * ux_23 + ky1 * uy_23) / w1
        - (k1p2 * w2 + k1p3 * w3) / 2 / w1
        - w1 * (w2 + w3) * second_order_surface_elevation_23 / grav
    )
    particular = (
        +second_order_surface_elevation_23 * (w1 + w2 + w3) * w1 / grav
        - second_order_potential_23 * (w2 + w3) ** 2 * (w1 + w2 + w3) / grav**2
        + k1**2 * (w1 + w2 + w3) / 2 / w1
    )
    if lagrangian:
        raise Exception(
            "third_order_amplitude_correction_sigma_coordinates is not implemented for the Lagrangian case"
        )

    return dispersion + bernouilli + particular


@vectorize(_interaction_signatures_symmetric, **numba_default_vectorize)
def third_order_amplitude_correction(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    w = (w1, w2, -w2)
    k = (k1, k2, k2)
    kx = (kx1, kx2, -kx2)
    ky = (ky1, ky2, -ky2)

    coef = 0.0
    for jj in range(3):
        ii1 = jj % 3
        ii2 = (jj + 1) % 3
        ii3 = (jj + 2) % 3

        coef += _third_order_amplitude_correction_non_symmetric(
            w[ii1],
            k[ii1],
            kx[ii1],
            ky[ii1],
            w[ii2],
            k[ii2],
            kx[ii2],
            ky[ii2],
            w[ii3],
            k[ii3],
            kx[ii3],
            ky[ii3],
            depth,
            grav,
            wave_driven_setup_included_in_mean_depth,
            wave_driven_flow_included_in_mean_flow,
            lagrangian,
        )
        # print(coef,            w[ii1],
        #     k[ii1],
        #     kx[ii1])
    return coef


@vectorize(_interaction_signatures_symmetric, **numba_default_vectorize)
def third_order_amplitude_correction_sigma_coordinates(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    w = (w1, w2, -w2)
    k = (k1, k2, k2)
    kx = (kx1, kx2, -kx2)
    ky = (ky1, ky2, -ky2)

    coef = 0.0
    for jj in range(3):
        ii1 = jj % 3
        ii2 = (jj + 1) % 3
        ii3 = (jj + 2) % 3

        coef += _third_order_amplitude_correction_non_symmetric_sigma_coordinates(
            w[ii1],
            k[ii1],
            kx[ii1],
            ky[ii1],
            w[ii2],
            k[ii2],
            kx[ii2],
            ky[ii2],
            w[ii3],
            k[ii3],
            kx[ii3],
            ky[ii3],
            depth,
            grav,
            wave_driven_setup_included_in_mean_depth,
            wave_driven_flow_included_in_mean_flow,
            lagrangian,
        )
    return coef


@vectorize(_interaction_signatures_symmetric, **numba_default_vectorize)
def third_order_dispersion_correction(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    if lagrangian:
        inner_product_k12 = kx1 * kx2 + ky1 * ky2
        coef = -inner_product_k12 * (1 + grav**2 * k2**2 / w2**4) * w2
    else:
        coef = 0.0

    w = (w1, w2, -w2)
    k = (k1, k2, k2)
    kx = (kx1, kx2, -kx2)
    ky = (ky1, ky2, -ky2)

    for jj in range(3):
        ii1 = jj % 3
        ii2 = (jj + 1) % 3
        ii3 = (jj + 2) % 3

        coef += (
            _third_order_dispersion_correction_non_symmetric(
                w[ii1],
                k[ii1],
                kx[ii1],
                ky[ii1],
                w[ii2],
                k[ii2],
                kx[ii2],
                ky[ii2],
                w[ii3],
                k[ii3],
                kx[ii3],
                ky[ii3],
                depth,
                grav,
                wave_driven_setup_included_in_mean_depth,
                wave_driven_flow_included_in_mean_flow,
                lagrangian,
            )
            / grav
            / 2
        )
    return coef


@vectorize(_interaction_signatures_symmetric, **numba_default_vectorize)
def third_order_dispersion_correction_sigma_coordinates(
    w1,
    k1,
    kx1,
    ky1,
    w2,
    k2,
    kx2,
    ky2,
    depth,
    grav,
    wave_driven_setup_included_in_mean_depth,
    wave_driven_flow_included_in_mean_flow,
    lagrangian,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    if lagrangian:
        inner_product_k12 = kx1 * kx2 + ky1 * ky2
        coef = -inner_product_k12 * (1 + grav**2 * k2**2 / w2**4) * w2
    else:
        coef = 0.0

    w = (w1, w2, -w2)
    k = (k1, k2, k2)
    kx = (kx1, kx2, -kx2)
    ky = (ky1, ky2, -ky2)

    for jj in range(3):
        ii1 = jj % 3
        ii2 = (jj + 1) % 3
        ii3 = (jj + 2) % 3

        coef += (
            _third_order_dispersion_correction_non_symmetric_sigma_coordinates(
                w[ii1],
                k[ii1],
                kx[ii1],
                ky[ii1],
                w[ii2],
                k[ii2],
                kx[ii2],
                ky[ii2],
                w[ii3],
                k[ii3],
                kx[ii3],
                ky[ii3],
                depth,
                grav,
                wave_driven_setup_included_in_mean_depth,
                wave_driven_flow_included_in_mean_flow,
                lagrangian,
            )
            / grav
            / 2
        )
    return coef
