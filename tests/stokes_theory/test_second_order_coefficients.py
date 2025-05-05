from linearwavetheory.stokes_theory._second_order_coeficients import (
    _second_order_surface_elevation,
    _second_order_lagrangian_surface_elevation,
    _second_order_horizontal_velocity,
    _second_order_vertical_velocity,
)
import numpy as np
from linearwavetheory.settings import _GRAV


def stokes_sum_amplitude(k, d):
    w = np.tanh(k * d)
    return k * (3 - w**2) / 4 / w**3


def stokes_setdown(wavenumber, depth):
    angular_frequency = np.sqrt(_GRAV * wavenumber * np.tanh(wavenumber * depth))
    sh = np.sinh(wavenumber * depth)
    sh2 = np.sinh(2.0 * wavenumber * depth)
    sh4 = np.sinh(4.0 * wavenumber * depth)

    cg = (
        np.abs(angular_frequency)
        / wavenumber
        * (0.5 + wavenumber * depth / np.sinh(2.0 * wavenumber * depth))
    )
    K = (
        angular_frequency**2
        / sh**2
        / 4
        * (sh4 + 3 * sh2 + 2 * wavenumber * depth)
        / (sh2 + 2 * wavenumber * depth)
    )

    return (
        -1
        / _GRAV
        * (
            cg**2 * K / (_GRAV * depth - cg**2)
            + angular_frequency**2 / 4 / sh**2
        )
    )


def stokes_mean_flow(wavenumber, depth):
    angular_frequency = np.sqrt(_GRAV * wavenumber * np.tanh(wavenumber * depth))
    sh = np.sinh(wavenumber * depth)
    sh2 = np.sinh(2.0 * wavenumber * depth)
    sh4 = np.sinh(4.0 * wavenumber * depth)

    cg = (
        np.abs(angular_frequency)
        / wavenumber
        * (0.5 + wavenumber * depth / np.sinh(2.0 * wavenumber * depth))
    )
    K = (
        angular_frequency**2
        / sh**2
        / 4
        * (sh4 + 3 * sh2 + 2 * wavenumber * depth)
        / (sh2 + 2 * wavenumber * depth)
    )

    return -cg * K / (_GRAV * depth - cg**2)


def stokes_second_order_potential(wavenumber, depth):
    # Taken from Dingemans, p. 171 (the thesis)
    angular_frequency = np.sqrt(_GRAV * wavenumber * np.tanh(wavenumber * depth))

    return (
        angular_frequency
        / np.sinh(wavenumber * depth)
        * (3 / 8 / np.sinh(wavenumber * depth) ** 3 * np.cosh(2 * wavenumber * depth))
    )


def stokes_second_order_horizontal_flow_amplitude(wavenumber, depth):
    # Taken from Dingemans, p. 171 (the thesis)
    return 2 * wavenumber * stokes_second_order_potential(wavenumber, depth)


def stokes_second_order_vertical_flow_amplitude(wavenumber, depth):
    # Taken from Dingemans, p. 171 (the thesis). To note the minus sign here is because we compare to complex
    # exponential as the base function (not a sine)
    return (
        -2
        * wavenumber
        * stokes_second_order_potential(wavenumber, depth)
        * np.tanh(2 * wavenumber * depth)
    )


def stokes_wave(amplitude, wavenumber, depth, phase):
    z0 = amplitude**2 * stokes_setdown(wavenumber, depth)
    z1 = amplitude
    z2 = amplitude**2 * stokes_sum_amplitude(wavenumber, depth)

    return z0 + z1 * np.cos(phase) + z2 * np.cos(2 * phase), z0, z1, z2


def stokes_skewness(amplitude, wavenumber, depth, include_set_down=False):
    phase = np.linspace(0, 2 * np.pi, 100000)
    surface_elevation, z0, z1, z2 = stokes_wave(amplitude, wavenumber, depth, phase)
    # We only consider the skewness calculated up to O(4). I.e. we ignore contributions due to
    # z2*z2*z0 etc. Note that there are 3 terms in the skewness calculation, and that 0.5 is the mean of cos^2(x)
    # and 0.25 is the mean of cos^2(x) cos(2x)
    if include_set_down:
        return 1.5 * z1 * z1 * z0 + z1 * z1 * z2 * 3 / 4
    else:
        return z1 * z1 * z2 * 3 / 4


def test_coef_second_order_surface_elevation():
    depth = 1
    grav = _GRAV
    for kd in np.linspace(0.1, 10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))
        c12 = _second_order_surface_elevation(w, k, k, 0.0, w, k, k, 0.0, depth, grav)
        assert np.isclose(c12, 2 * stokes_sum_amplitude(k, depth))

        _setdown = _second_order_surface_elevation(
            w, k, k, 0.0, -w, k, -k, 0.0, depth, grav
        )
        # We have a factor 2 here because in stokes theory A^2 is used - whereas we use complex amplitudes so that
        # A^2 = a*conj(a) *2
        _stokes_setdown = 2 * stokes_setdown(k, depth)

        assert np.isclose(_setdown, _stokes_setdown, rtol=1e-5, atol=1e-5)

    # Test it works for infinite depth
    depth = np.inf
    k = 1
    w = np.sqrt(grav * k)
    c12 = _second_order_surface_elevation(w, k, k, 0.0, w, k, k, 0.0, depth, grav)
    assert np.isfinite(c12)

    c12 = _second_order_surface_elevation(w, k, k, 0.0, -w, k, -k, 0.0, depth, grav)
    assert np.isfinite(c12)


def test_coef_second_order_horizontal_velocity():
    #
    # Do we correctly reproduce the mean flow associated with a single wavetrain? Classic LHS result used to check.
    #
    depth = 1
    grav = _GRAV
    for kd in np.linspace(0.1, 10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))

        velocity_amplitude = stokes_second_order_horizontal_flow_amplitude(k, depth) * 2
        amplitude = _second_order_horizontal_velocity(
            w, k, k, 0.0, w, k, k, 0.0, 1, depth, grav
        )
        assert np.isclose(amplitude, velocity_amplitude, rtol=1e-5, atol=1e-5)

        _mean_flow = _second_order_horizontal_velocity(
            w, k, k, 0.0, -w, k, -k, 0.0, 1, depth, grav
        )
        # We have a factor 2 here because in stokes theory A^2 is used - whereas we use complex amplitudes so that
        # A^2 = a*conj(a) *2
        _stokes_mean_flow = 2 * stokes_mean_flow(k, depth)
        assert np.isclose(_mean_flow, _stokes_mean_flow, rtol=1e-5, atol=1e-5)

        # Check with direction
        kx = k * np.cos(1.75 * np.pi)
        ky = k * np.sin(1.75 * np.pi)
        _mean_flow_x = _second_order_horizontal_velocity(
            w, k, kx, ky, -w, k, -kx, -ky, 1, depth, grav
        )

        _mean_flow_y = _second_order_horizontal_velocity(
            w, k, kx, ky, -w, k, -kx, -ky, -1, depth, grav
        )

        assert _mean_flow_x < 0
        assert _mean_flow_y > 0

        # Note the negative sign here, the flow is opposite to the wave number direction
        _mean_flow = -np.sqrt(_mean_flow_x**2 + _mean_flow_y**2)

        assert np.isclose(_mean_flow, _stokes_mean_flow, rtol=1e-5, atol=1e-5)

    # Test it works for infinite depth
    depth = np.inf
    k = 1
    w = np.sqrt(grav * k)
    c12 = _second_order_horizontal_velocity(
        w, k, k, 0.0, w, k, k, 0.0, 1.0, depth, grav
    )
    assert np.isfinite(c12)

    c12 = _second_order_horizontal_velocity(
        w, k, k, 0.0, -w, k, -k, 0.0, 1, depth, grav
    )
    assert np.isfinite(c12)


def test_coef_second_order_lagrangian_surface_elevation():

    # Deep water sum interactions must vanish
    depth = 1
    grav = 9.81
    for kd in range(10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))

        k2 = 1.5 * k
        w2 = np.sqrt(grav * k2 * np.tanh(k2 * depth))
        c12 = _second_order_lagrangian_surface_elevation(
            w, k, k, 0.0, w2, k2, k2, 0.0, depth, grav
        )
        assert np.isclose(c12, 0, rtol=1e-5, atol=1e-5)


def test_coef_second_order_vertical_velocity():
    #
    # Do we correctly reproduce the vertical flow associated with a single wavetrain?
    #
    depth = 1
    grav = _GRAV
    for kd in np.linspace(0.1, 10, 100):
        k = kd / depth
        w = np.sqrt(grav * k * np.tanh(k * depth))

        velocity_amplitude = stokes_second_order_vertical_flow_amplitude(k, depth) * 2
        amplitude = _second_order_vertical_velocity(
            w, k, k, 0.0, w, k, k, 0.0, depth, grav
        )
        assert np.isclose(amplitude, velocity_amplitude, rtol=1e-5, atol=1e-5)

        _mean_flow = _second_order_vertical_velocity(
            w, k, k, 0.0, -w, k, -k, 0.0, depth, grav
        )
        # We have a factor 2 here because in stokes theory A^2 is used - whereas we use complex amplitudes so that
        # A^2 = a*conj(a) *2

        assert np.isclose(_mean_flow, 0.0, rtol=1e-5, atol=1e-5)

    # Test it works for infinite depth
    depth = np.inf
    k = 1
    w = np.sqrt(grav * k)
    c12 = _second_order_vertical_velocity(w, k, k, 0.0, w, k, k, 0.0, depth, grav)
    assert np.isfinite(c12)

    c12 = _second_order_vertical_velocity(w, k, k, 0.0, -w, k, -k, 0.0, depth, grav)
    assert np.isfinite(c12)
