import src.linearwavetheory.dispersion as ld
import numpy as np
from numpy.testing import assert_allclose
from src.linearwavetheory import settings
from src.linearwavetheory.settings import physics_options


def test_inverse_intrinsic_dispersion_relation():
    """
    Test if the inverse intrinsic dispersion relation is correct.
    """

    # For vector input (depth and frequency)
    kd = 10 ** np.linspace(-5, 5, 10001)
    depth = np.linspace(1, 100, 10001)
    wavenumber = kd / depth

    # calculate the intrinsic frequency from the wavenumber and depth
    intrinsic_frequency = ld.intrinsic_dispersion_relation(wavenumber, depth)
    # estimate the wavenumber based on the intrinsic frequency and depth
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, depth)
    solution_freq = ld.intrinsic_dispersion_relation(solution, depth)

    # check if the solution is similar to the initial wavenumber within the given tolerance
    assert_allclose(
        intrinsic_frequency, solution_freq, atol=0, rtol=settings._RELATIVE_TOLERANCE
    )

    # For scalar depth
    depth = 10
    wavenumber = kd / depth
    intrinsic_frequency = ld.intrinsic_dispersion_relation(wavenumber, 10)
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, 10)
    solution_freq = ld.intrinsic_dispersion_relation(solution, 10)
    assert_allclose(
        intrinsic_frequency, solution_freq, atol=0, rtol=settings._RELATIVE_TOLERANCE
    )

    # errors for wrong depth
    try:
        _ = ld.inverse_intrinsic_dispersion_relation(
            intrinsic_frequency, np.array([11, 12])
        )
        assert False
    except ValueError:
        pass

    # For scalar input (depth and frequency)
    intrinsic_frequency = ld.intrinsic_dispersion_relation(0.1, 10)
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, 10)
    solution_freq = ld.intrinsic_dispersion_relation(solution, 10)
    assert_allclose(
        solution_freq, intrinsic_frequency, atol=0, rtol=settings._RELATIVE_TOLERANCE
    )

    # For negative depth we get a nan
    intrinsic_frequency = ld.intrinsic_dispersion_relation(0.1, 10)
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, -10)
    assert np.isnan(solution[0])

    # For 0 frequency we get a 0
    solution = ld.inverse_intrinsic_dispersion_relation(0.0, 10)
    assert solution[0] == 0.0


def test_intrinsic_dispersion_relation():
    """
    Test if the intrinsic dispersion relation is correct.
    """

    # Shallow water
    k = np.linspace(0, 0.01, 100)
    depth = np.full(100, 1.0)

    shallow_water_omega = k * np.sqrt(settings._GRAV * depth)
    omega = ld.intrinsic_dispersion_relation(k, depth)

    assert_allclose(
        omega, shallow_water_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE
    )

    # Deep water
    k = np.linspace(0.1, 1, 100)
    depth = np.full(100, 1000.0)
    deep_water_omega = np.sqrt(settings._GRAV * k)
    omega = ld.intrinsic_dispersion_relation(k, depth)
    assert_allclose(omega, deep_water_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # Capillary waves
    k = np.linspace(10000, 100000, 10)
    depth = np.full(10, 1000.0)
    capillary_omega = np.sqrt(settings._KINEMATIC_SURFACE_TENSION * k**3)
    omega = ld.intrinsic_dispersion_relation(k, depth)
    assert_allclose(
        omega, capillary_omega, atol=np.inf, rtol=settings._RELATIVE_TOLERANCE
    )

    # Test input verification

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_dispersion_relation(-1.0, 10)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_dispersion_relation(-1.0, np.array([10, 10]))
        assert False
    except ValueError:
        pass

    # nan for neg depth
    assert np.isnan(ld.intrinsic_dispersion_relation(1.0, -10)[0])


def test_encounter_dispersion_relation():
    # deep water
    kvec = np.array((0.1, 0.2))
    k = np.linalg.norm(kvec)
    d = np.inf

    # no flow - should be the same as intrinsic
    assert_allclose(
        ld.encounter_dispersion_relation(kvec, d),
        ld.intrinsic_dispersion_relation(k, d),
        atol=0,
        rtol=settings._RELATIVE_TOLERANCE,
    )

    # flow perpendicular to wave vector, should be the same as intrinsic
    kvec = np.array((0.1, 0.0))
    u = np.array((0.0, 1.0))
    k = np.linalg.norm(kvec)
    d = np.inf
    assert_allclose(
        ld.encounter_dispersion_relation(kvec, d, u),
        ld.intrinsic_dispersion_relation(k, d),
        atol=0,
        rtol=settings._RELATIVE_TOLERANCE,
    )

    # flow parallel to wave vector, should be the same as intrinsic + doppler
    kvec = np.array((0.1, 0.0))
    u = np.array((1.0, 0.0))
    k = np.linalg.norm(kvec)
    d = np.inf
    assert_allclose(
        ld.encounter_dispersion_relation(kvec, d, u),
        ld.intrinsic_dispersion_relation(k, d) + kvec[0] * u[0],
        atol=0,
        rtol=settings._RELATIVE_TOLERANCE,
    )

    # Check if broadcasting works
    kvec = np.array(((0.1, 0.0), (0.1, 0.0)))
    u = np.array((1.0, 0.0))
    k = np.linalg.norm(kvec, axis=-1)
    d = np.inf
    assert_allclose(
        ld.encounter_dispersion_relation(kvec, d, u),
        ld.intrinsic_dispersion_relation(k, d) + kvec[0, 0] * u[0],
        atol=0,
        rtol=settings._RELATIVE_TOLERANCE,
    )

    # for a negative current we can get negative frequencies
    kvec = np.array(((0.1, 0.0), (0.1, 0.0)))
    u = np.array((-10.0, 0.0))
    k = np.linalg.norm(kvec, axis=-1)
    d = np.inf
    assert np.all(ld.encounter_dispersion_relation(kvec, d, u) < 0)


def test_intrinsic_phase_speed():
    """
    Test if the intrinsic dispersion relation is correct.
    """

    # Shallow water
    k = np.linspace(0, 0.01, 100)
    depth = np.full(100, 1.0)

    shallow_water_c = np.sqrt(settings._GRAV * depth)
    c = ld.intrinsic_phase_speed(k, depth)

    assert_allclose(c, shallow_water_c, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # Deep water
    k = np.linspace(0.1, 1, 100)
    depth = np.full(100, 1000.0)
    deep_water_omega = np.sqrt(settings._GRAV / k)
    omega = ld.intrinsic_phase_speed(k, depth)
    assert_allclose(omega, deep_water_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # Capillary waves
    k = np.linspace(10000, 100000, 10)
    depth = np.full(10, 1000.0)
    capillary_omega = np.sqrt(
        settings._GRAV / k + settings._KINEMATIC_SURFACE_TENSION * k
    )
    omega = ld.intrinsic_phase_speed(k, depth)
    assert_allclose(omega, capillary_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_phase_speed(-1.0, 10)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_phase_speed(-1.0, np.array([10, 10]))
        assert False
    except ValueError:
        pass

    # nan for neg depth
    assert np.isnan(ld.intrinsic_phase_speed(1.0, -10)[0])


def test_intrinsic_group_speed():
    """
    Test if the intrinsic dispersion relation is correct.
    """

    # Shallow water
    k = np.linspace(0, 0.01, 100)
    depth = np.full(100, 1.0)

    shallow_water_c = np.sqrt(settings._GRAV * depth)
    c = ld.intrinsic_group_speed(k, depth)
    assert_allclose(c, shallow_water_c, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # test with scalar depth
    k = np.linspace(0, 0.01, 100)
    depth = 1

    shallow_water_c = np.sqrt(settings._GRAV * depth) * np.ones(100)
    c = ld.intrinsic_group_speed(k, depth)
    assert_allclose(c, shallow_water_c, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # Deep water
    k = np.linspace(0.1, 1, 100)
    depth = np.full(100, 1000.0)
    deep_water_omega = np.sqrt(settings._GRAV / k) / 2
    omega = ld.intrinsic_group_speed(k, depth)
    assert_allclose(omega, deep_water_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    # Capillary waves
    k = np.linspace(100000, 1000000, 10)
    depth = np.full(10, 1000.0)
    capillary_omega = 3 * np.sqrt(settings._KINEMATIC_SURFACE_TENSION * k) / 2
    omega = ld.intrinsic_group_speed(k, depth)
    assert_allclose(omega, capillary_omega, atol=0, rtol=settings._RELATIVE_TOLERANCE)

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_group_speed(-1.0, 10)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        _ = ld.intrinsic_phase_speed(-1.0, np.array([10, 10]))
        assert False
    except ValueError:
        pass

    # nan for neg depth
    assert np.isnan(ld.intrinsic_phase_speed(1.0, -10)[0])


def test_encounter_group_phase_velocity_speed():
    def _helper(k, d, u=(0, 0)):
        cg = ld.encounter_group_velocity(k, d, u)
        c = ld.encounter_phase_velocity(k, d, u)
        _ = ld.encounter_group_speed(k, d, u)
        _ = ld.encounter_group_velocity(k, d, u)
        return cg, c

    # test call sequences
    # --------------------
    # scalars
    try:
        cg, c = _helper(0.1, 10)
        assert False
    except ValueError:
        pass

    # 1d vec, scalar depth
    cg, c = _helper((0.1, 0), 10)

    # 2d vec, scalar depth
    cg, c = _helper(((0.1, 0), (0.2, 0)), 10)

    # 2d vec, depth vec
    cg, c = _helper(((0.1, 0), (0.2, 0)), (10, 11))

    # 1d vec, scalar depth, scalar u, scalar o
    try:
        cg, c = _helper((0.1, 0), 10, 1, 1)
        assert False
    except Exception:
        pass

    # 1d vec, scalar depth, vec  u, vec o
    cg, c = _helper((0.1, 0), 10, (1, 0))

    # 2d vec, depth vec, vec  u, vec o
    cg, c = _helper(((0.1, 0), (0.2, 0)), (10, 11), (1, 0))

    # 2d vec, depth vec, 2d vec  u, 2d vec o
    cg, c = _helper(((0.1, 0), (0.2, 0)), (10, 11), ((0.1, 0), (0.2, 0)))


def test_inverse_encounter_dispersion_relation_zero_current():
    """
    With no current, the principal root reduces to the intrinsic inversion, and the secondary root does not exist.
    """
    freq = np.array([0.3, 0.6, 1.0])
    direction = np.zeros_like(freq)

    for regime in ["shallow", "intermediate", "deep"]:
        opts = physics_options(wave_regime=regime)
        depth = 5.0 if regime == "shallow" else (5000.0 if regime == "deep" else 20.0)

        k_principal = ld.inverse_encounter_dispersion_relation(
            freq, direction, depth=depth, physics_options=opts
        )
        k_reference = ld.inverse_intrinsic_dispersion_relation(
            freq, depth, physics_options=opts
        )
        assert_allclose(
            k_principal, k_reference, atol=0, rtol=settings._RELATIVE_TOLERANCE
        )

        k_secondary = ld.inverse_encounter_dispersion_relation(
            freq, direction, depth=depth, physics_options=opts, root="secondary"
        )
        assert np.all(np.isnan(k_secondary))


def test_inverse_encounter_dispersion_relation_roundtrip_principal():
    """
    For a range of wavenumbers, currents and depths, the forward encounter dispersion relation followed by its
    inverse should recover the original wavenumber for the principal root.
    """
    for regime, depth in [("deep", 5000.0), ("intermediate", 20.0), ("shallow", 5.0)]:
        opts = physics_options(wave_regime=regime)
        for current_speed, current_direction in [
            (0.0, 0.0),
            (0.5, 0.0),
            (2.0, 0.3),
            (-0.3, 0.0),
        ]:
            for k in [0.001, 0.01, 0.05]:
                direction = 0.0
                delta = current_speed * np.cos(direction - current_direction)

                omega = ld.encounter_dispersion_relation(
                    np.array((k, 0.0)),
                    depth,
                    ambient_current_velocity=(
                        current_speed * np.cos(current_direction),
                        current_speed * np.sin(current_direction),
                    ),
                    physics_options=opts,
                )[0]
                if omega < 0:
                    continue

                k_est = ld.inverse_encounter_dispersion_relation(
                    omega,
                    direction,
                    relative_current_speed=current_speed,
                    relative_current_direction=current_direction,
                    depth=depth,
                    physics_options=opts,
                )
                assert_allclose(
                    k_est,
                    k,
                    atol=0,
                    rtol=100 * settings._RELATIVE_TOLERANCE,
                    err_msg=f"regime={regime} k={k} delta={delta}",
                )


def test_inverse_encounter_dispersion_relation_roundtrip_secondary():
    """
    Round-trip check for the secondary root, in the regimes/current combinations where it is expected to exist:
    deep/intermediate water with a following current (delta > 0), and shallow water with a supercritical current.
    """
    direction = 0.0

    # deep and intermediate water, delta > 0
    for regime, depth in [("deep", 5000.0), ("intermediate", 20.0)]:
        opts = physics_options(wave_regime=regime)
        for current_speed in [0.5, 2.0]:
            for k in [0.01, 0.1, 1.0]:
                intrinsic_omega = ld.intrinsic_dispersion_relation(
                    k, depth, physics_options=opts
                )[0]
                omega = k * current_speed - intrinsic_omega
                if omega < 0:
                    continue

                k_est = ld.inverse_encounter_dispersion_relation(
                    omega,
                    direction,
                    relative_current_speed=current_speed,
                    relative_current_direction=0.0,
                    depth=depth,
                    physics_options=opts,
                    root="secondary",
                )
                assert_allclose(
                    k_est,
                    k,
                    atol=0,
                    rtol=100 * settings._RELATIVE_TOLERANCE,
                    err_msg=f"regime={regime} k={k} current_speed={current_speed}",
                )

    # shallow water, supercritical current
    opts = physics_options(wave_regime="shallow")
    depth = 5.0
    c_sw = np.sqrt(settings._GRAV * depth)
    for current_speed in [c_sw * 1.5, c_sw * 3.0]:
        for k in [0.001, 0.005]:
            omega = k * current_speed - k * c_sw
            k_est = ld.inverse_encounter_dispersion_relation(
                omega,
                direction,
                relative_current_speed=current_speed,
                relative_current_direction=0.0,
                depth=depth,
                physics_options=opts,
                root="secondary",
            )
            assert_allclose(k_est, k, atol=0, rtol=settings._RELATIVE_TOLERANCE)


def test_inverse_encounter_dispersion_relation_nonexistence():
    """
    Check that roots that do not exist return NaN rather than raising, across regimes.
    """
    direction = 0.0

    # Delta == 0 -> secondary always NaN, regardless of regime.
    for regime, depth in [("deep", 5000.0), ("intermediate", 20.0), ("shallow", 5.0)]:
        opts = physics_options(wave_regime=regime)
        k_sec = ld.inverse_encounter_dispersion_relation(
            1.0,
            direction,
            relative_current_speed=0.0,
            depth=depth,
            physics_options=opts,
            root="secondary",
        )
        assert np.isnan(k_sec[0])

    # deep/intermediate water: opposing current strong enough to block all waves in this direction.
    for regime, depth in [("deep", 5000.0), ("intermediate", 20.0)]:
        opts = physics_options(wave_regime=regime)
        current_speed = -5.0
        k_p = ld.inverse_encounter_dispersion_relation(
            50.0,
            direction,
            relative_current_speed=current_speed,
            depth=depth,
            physics_options=opts,
        )
        k_s = ld.inverse_encounter_dispersion_relation(
            50.0,
            direction,
            relative_current_speed=current_speed,
            depth=depth,
            physics_options=opts,
            root="secondary",
        )
        assert np.isnan(k_p[0])
        assert np.isnan(k_s[0])

    # shallow water: retreating current faster than the shallow-water wave speed blocks the principal root too.
    opts = physics_options(wave_regime="shallow")
    depth = 5.0
    c_sw = np.sqrt(settings._GRAV * depth)
    k_p = ld.inverse_encounter_dispersion_relation(
        1.0,
        direction,
        relative_current_speed=-2 * c_sw,
        depth=depth,
        physics_options=opts,
    )
    assert np.isnan(k_p[0])

    # shallow water: subcritical current -> secondary never exists.
    k_s = ld.inverse_encounter_dispersion_relation(
        1.0,
        direction,
        relative_current_speed=0.5 * c_sw,
        depth=depth,
        physics_options=opts,
        root="secondary",
    )
    assert np.isnan(k_s[0])

    # Negative frequency raises.
    try:
        ld.inverse_encounter_dispersion_relation(-1.0, 0.0)
        assert False
    except ValueError:
        pass

    # Invalid root raises.
    try:
        ld.inverse_encounter_dispersion_relation(1.0, 0.0, root="bogus")
        assert False
    except ValueError:
        pass


def test_inverse_encounter_dispersion_relation_closed_form_cross_check():
    """
    In the regimes where the closed-form shallow/deep formulas should coincide with the general numerical solver,
    check that they agree (catches inconsistencies between the two implementations of the same physics).
    """
    default_opts = physics_options(wave_regime="intermediate")

    # Deep-water-like: very large depth, moderate k.
    depth = 5000.0
    deep_opts = physics_options(wave_regime="deep")
    for delta_current, direction in [(0.0, 0.0), (1.0, 0.0), (-0.5, 0.0)]:
        omega = 1.0
        k_general = ld.inverse_encounter_dispersion_relation(
            omega,
            direction,
            relative_current_speed=delta_current,
            depth=depth,
            physics_options=default_opts,
        )
        k_deep = ld.inverse_encounter_dispersion_relation(
            omega,
            direction,
            relative_current_speed=delta_current,
            depth=depth,
            physics_options=deep_opts,
        )
        assert_allclose(k_general, k_deep, atol=0, rtol=1e-3)

    # Shallow-water-like: small depth, supercritical current so both roots are in the linear regime.
    depth = 5.0
    shallow_opts = physics_options(wave_regime="shallow")
    c_sw = np.sqrt(settings._GRAV * depth)
    omega = 0.01
    for root in ["principal", "secondary"]:
        current_speed = c_sw * 2.0
        k_general = ld.inverse_encounter_dispersion_relation(
            omega,
            0.0,
            relative_current_speed=current_speed,
            depth=depth,
            physics_options=default_opts,
            root=root,
        )
        k_shallow = ld.inverse_encounter_dispersion_relation(
            omega,
            0.0,
            relative_current_speed=current_speed,
            depth=depth,
            physics_options=shallow_opts,
            root=root,
        )
        assert_allclose(k_general, k_shallow, atol=0, rtol=1e-2)


def test_inverse_encounter_dispersion_relation_broadcasting():
    """
    Check that array frequency/direction broadcast correctly against a scalar current and both scalar and array
    depth.
    """
    freq = np.array([0.3, 0.5, 0.8])
    direction = np.array([0.0, 0.5, 1.0])

    k_scalar_depth = ld.inverse_encounter_dispersion_relation(
        freq,
        direction,
        relative_current_speed=0.3,
        relative_current_direction=0.2,
        depth=20.0,
    )
    assert k_scalar_depth.shape == freq.shape

    depths = np.array([20.0, 30.0, 50.0])
    k_array_depth = ld.inverse_encounter_dispersion_relation(
        freq,
        direction,
        relative_current_speed=0.3,
        relative_current_direction=0.2,
        depth=depths,
    )
    assert k_array_depth.shape == freq.shape
    assert np.all(np.isfinite(k_array_depth))
