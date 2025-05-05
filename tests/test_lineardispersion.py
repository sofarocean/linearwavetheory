import src.linearwavetheory.dispersion as ld
import numpy as np
from numpy.testing import assert_allclose
from src.linearwavetheory import settings


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
