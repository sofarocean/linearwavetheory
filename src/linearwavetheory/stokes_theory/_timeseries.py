from ._second_order_coeficients import _second_order_surface_elevation
from linearwavetheory import inverse_intrinsic_dispersion_relation
from linearwavetheory.settings import _parse_options
import numpy as np


def surface_time_series(
    primary_wave_amplitudes,
    primary_wave_frequencies,
    primary_wave_directions_degree,
    depth,
    time,
    x=0,
    y=0,
    **kwargs
):

    include_mean_setdown = kwargs.get("include_mean_setdown", False)
    nonlinear = kwargs.get("nonlinear", True)
    linear = kwargs.get("linear", True)
    self_interactions = kwargs.get("self_interactions", True)
    cross_interactions = kwargs.get("cross_interactions", True)
    physics_options = kwargs.get("physics_options", None)
    _, physics_options, _ = _parse_options(None, physics_options, None)
    grav = physics_options.grav

    wavenumber = inverse_intrinsic_dispersion_relation(
        2 * np.pi * primary_wave_frequencies, depth
    )
    angular_frequency = 2 * np.pi * primary_wave_frequencies

    kx = wavenumber * np.cos(np.deg2rad(primary_wave_directions_degree))
    ky = wavenumber * np.sin(np.deg2rad(primary_wave_directions_degree))

    first_order_surface_elevation = np.zeros_like(time)
    second_order_surface_elevation = np.zeros_like(time)

    if linear:
        for a1, w1, k1, kx1, ky1 in zip(
            primary_wave_amplitudes, angular_frequency, wavenumber, kx, ky
        ):
            first_order_surface_elevation += np.real(
                a1 * np.exp(1j * (kx1 * x + ky1 * y - w1 * time))
            )

    if nonlinear:
        for sign in [-1, 1]:
            wave_component1_index = -1
            for a1, w1, k1, kx1, ky1 in zip(
                primary_wave_amplitudes, angular_frequency, wavenumber, kx, ky
            ):
                wave_component1_index += 1
                wave_component2_index = -1
                for a2, w2, k2, kx2, ky2 in zip(
                    primary_wave_amplitudes, angular_frequency, wavenumber, kx, ky
                ):
                    wave_component2_index += 1
                    interaction = _second_order_surface_elevation(
                        w1,
                        k1,
                        kx1,
                        ky1,
                        sign * w2,
                        k2,
                        sign * kx2,
                        sign * ky2,
                        depth,
                        grav,
                    )

                    wsum = w1 + sign * w2
                    kx_sum = kx1 + sign * kx2
                    ky_sum = ky1 + sign * ky2
                    if sign == -1:
                        a2 = np.conj(a2)

                        if (
                            wave_component1_index == wave_component2_index
                        ) and not include_mean_setdown:
                            continue

                    if not self_interactions:
                        if wave_component1_index == wave_component2_index:
                            continue

                    if not cross_interactions:
                        if wave_component1_index != wave_component2_index:
                            continue

                    second_order_surface_elevation += interaction * np.real(
                        a1 * a2 * np.exp(1j * (kx_sum * x + ky_sum * y - wsum * time))
                    )

    surface_elevation = 2 * (
        first_order_surface_elevation + second_order_surface_elevation
    )

    return surface_elevation
