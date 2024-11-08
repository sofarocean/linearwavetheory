from numba import vectorize, float64, float32, bool, jit
from linearwavetheory._numba_settings import numba_default_vectorize, numba_default
import numpy as np

from linearwavetheory.stokes_theory._second_order_coeficients import (
    _second_order_potential,
    _second_order_surface_elevation,
    _second_order_horizontal_velocity,
    _second_order_vertical_velocity,
    _second_order_horizontal_lagrangian_surface_perturbation,
)

_interaction_signatures_to = [
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
        float32,
        float32,
        float32,
        bool,
        bool,
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
        float32,
        float32,
        float64,
        bool,
        bool,
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
        float32,
        float64,
        float64,
        bool,
        bool,
    ),
]


_interaction_signatures_to_reduced = [
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
        bool,
        bool,
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
        bool,
        bool,
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
        bool,
        bool,
    ),
]

_interaction_signatures_to_reduced_disp = [
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
        bool,
        bool,
        bool,
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
        bool,
        bool,
        bool,
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
        bool,
        bool,
        bool,
    ),
]


@vectorize(_interaction_signatures_to, **numba_default_vectorize)
def _third_order_coef_dispersion_non_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
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
        if not include_mean_setdown:
            second_order_surface_elevation_23 = 0.0

        if not include_mean_flow:
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


@vectorize(_interaction_signatures_to_reduced_disp, **numba_default_vectorize)
def _third_order_coef_dispersion_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
    include_stokes_drift=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    if include_stokes_drift:
        inner_product_k12 = kx1 * kx2 + ky1 * ky2
        coef = inner_product_k12 * (1 + grav**2 * k2**2 / w2**4) * w2
    else:
        coef = 0.0

    return (
        coef
        + (
            _third_order_coef_dispersion_non_symmetric(
                w1,
                k1,
                kx1,
                ky1,
                -w2,
                k2,
                -kx2,
                -ky2,
                w2,
                k2,
                kx2,
                ky2,
                depth,
                grav,
                include_mean_setdown,
                include_mean_flow,
            )
            + _third_order_coef_dispersion_non_symmetric(
                w2,
                k2,
                kx2,
                ky2,
                w1,
                k1,
                kx1,
                ky1,
                -w2,
                k2,
                -kx2,
                -ky2,
                depth,
                grav,
                include_mean_setdown,
                include_mean_flow,
            )
            + _third_order_coef_dispersion_non_symmetric(
                -w2,
                k2,
                -kx2,
                -ky2,
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
                include_mean_setdown,
                include_mean_flow,
            )
        )
        / grav
        / 2
    )


@vectorize(_interaction_signatures_to, **numba_default_vectorize)
def _third_order_coef_stokes_amplitude_non_symmetric_reference(
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
    include_mean_setdown=False,
    include_mean_flow=False,
):

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
        if not include_mean_setdown:
            second_order_surface_elevation_23 = 0.0

        if not include_mean_flow:
            ux_23 = 0.0
            uy_23 = 0.0

    term_c = second_order_surface_elevation_23 * w1**2 / grav
    term_u = -(kx1 * ux_23 + ky1 * uy_23) / w1
    term_w = -(w1 + w2 + w3) * second_order_w_23 / grav
    term_bb = (
        -inner_product_k23 / 2 / w2 / w3 * (w2**2 + w3**2)
        + (w2**2 * k3**2 + w3**2 * k2**2) / 2 / w2 / w3
        + k1**2 / 2
    )

    return term_c + term_u + term_w + term_bb


@vectorize(_interaction_signatures_to_reduced, **numba_default_vectorize)
def _third_order_coef_stokes_amplitude_symmetric_reference(
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
    include_mean_setdown=False,
    include_mean_flow=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    disp = (
        _third_order_coef_dispersion_symmetric(
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
            include_mean_setdown,
            include_mean_flow,
            False,
        )
        / w1
    )

    return +disp + (
        _third_order_coef_stokes_amplitude_non_symmetric_reference(
            w1,
            k1,
            kx1,
            ky1,
            w2,
            k2,
            kx2,
            ky2,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_non_symmetric_reference(
            w2,
            k2,
            kx2,
            ky2,
            w1,
            k1,
            kx1,
            ky1,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_non_symmetric_reference(
            -w2,
            k2,
            -kx2,
            -ky2,
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
            include_mean_setdown,
            include_mean_flow,
        )
    )


@jit(**numba_default)
def _third_order_coef_stokes_amplitude_non_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
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
        if not include_mean_setdown:
            second_order_surface_elevation_23 = 0.0

        if not include_mean_flow:
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

    return term_c + term_u + term_w + term_wz + term_tb


@vectorize(_interaction_signatures_to_reduced, **numba_default_vectorize)
def _third_order_coef_stokes_amplitude_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    return (
        _third_order_coef_stokes_amplitude_non_symmetric(
            w1,
            k1,
            kx1,
            ky1,
            w2,
            k2,
            kx2,
            ky2,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_non_symmetric(
            w2,
            k2,
            kx2,
            ky2,
            w1,
            k1,
            kx1,
            ky1,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_non_symmetric(
            -w2,
            k2,
            -kx2,
            -ky2,
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
            include_mean_setdown,
            include_mean_flow,
        )
    )


@vectorize(_interaction_signatures_to, **numba_default_vectorize)
def _third_order_coef_stokes_amplitude_lagrangian_non_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
):
    # read this as "inner product between k1 and k2 plus k3)
    inner_product_12p3 = kx1 * (kx2 + kx3) + ky1 * (ky2 + ky3)

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

    second_order_surface_elevation_23 = _second_order_surface_elevation(
        w3, k3, kx3, ky3, w2, k2, kx2, ky2, depth, grav
    )

    eulerian_contribution = _third_order_coef_stokes_amplitude_non_symmetric(
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
        include_mean_setdown,
        include_mean_flow,
    )

    term_c = -grav * inner_product_12p3 * second_order_surface_elevation_23 / w1**2
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
    return term_c + term_x + term_r + eulerian_contribution


@vectorize(_interaction_signatures_to_reduced, **numba_default_vectorize)
def _third_order_coef_stokes_amplitude_lagrangian_symmetric(
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
    include_mean_setdown=False,
    include_mean_flow=False,
):
    if w1 == 0.0 or w2 == 0.0:
        return 0.0

    return (
        _third_order_coef_stokes_amplitude_lagrangian_non_symmetric(
            w1,
            k1,
            kx1,
            ky1,
            w2,
            k2,
            kx2,
            ky2,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_lagrangian_non_symmetric(
            w2,
            k2,
            kx2,
            ky2,
            w1,
            k1,
            kx1,
            ky1,
            -w2,
            k2,
            -kx2,
            -ky2,
            depth,
            grav,
            include_mean_setdown,
            include_mean_flow,
        )
        + _third_order_coef_stokes_amplitude_lagrangian_non_symmetric(
            -w2,
            k2,
            -kx2,
            -ky2,
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
            include_mean_setdown,
            include_mean_flow,
        )
    )
