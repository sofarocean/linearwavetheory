import linearwavetheory.dispersion as ld
import numpy as np
from numpy.testing import assert_allclose


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
        intrinsic_frequency, solution_freq, atol=0, rtol=ld.RELATIVE_TOLERANCE
    )

    # For scalar depth
    depth = 10
    wavenumber = kd / depth
    intrinsic_frequency = ld.intrinsic_dispersion_relation(wavenumber, 10)
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, 10)
    solution_freq = ld.intrinsic_dispersion_relation(solution, 10)
    assert_allclose(
        intrinsic_frequency, solution_freq, atol=0, rtol=ld.RELATIVE_TOLERANCE
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
        solution_freq, intrinsic_frequency, atol=0, rtol=ld.RELATIVE_TOLERANCE
    )

    # For negative depth we get a nan
    intrinsic_frequency = ld.intrinsic_dispersion_relation(0.1, 10)
    solution = ld.inverse_intrinsic_dispersion_relation(intrinsic_frequency, -10)
    assert np.isnan(solution[0])

    # For 0 frequency we get a 0
    solution = ld.inverse_intrinsic_dispersion_relation(0.0, 10)
    assert solution[0] == 0.0

    # Test input verification
    try:
        solution = ld.inverse_intrinsic_dispersion_relation(
            1.0, 10, kinematic_surface_tension=-1
        )
        assert False
    except ValueError:
        pass

    try:
        solution = ld.inverse_intrinsic_dispersion_relation(1.0, 10, grav=-1)
        assert False
    except ValueError:
        pass

    try:
        solution = ld.inverse_intrinsic_dispersion_relation(
            1.0, 10, maximum_number_of_iterations=-1
        )
        assert False
    except ValueError:
        pass

    try:
        solution = ld.inverse_intrinsic_dispersion_relation(
            1.0, 10, absolute_tolerance=-1
        )
        assert False
    except ValueError:
        pass

    try:
        solution = ld.inverse_intrinsic_dispersion_relation(
            1.0, 10, relative_tolerance=-1
        )
        assert False
    except ValueError:
        pass


def test_intrinsic_dispersion_relation():
    """
    Test if the intrinsic dispersion relation is correct.
    """

    # Shallow water
    k = np.linspace(0, 0.01, 100)
    depth = np.full(100, 1.0)

    shallow_water_omega = k * np.sqrt(ld.GRAV * depth)
    omega = ld.intrinsic_dispersion_relation(k, depth)

    assert_allclose(omega, shallow_water_omega, atol=0, rtol=ld.RELATIVE_TOLERANCE)

    # Deep water
    k = np.linspace(0.1, 1, 100)
    depth = np.full(100, 1000.0)
    deep_water_omega = np.sqrt(ld.GRAV * k)
    omega = ld.intrinsic_dispersion_relation(k, depth)
    assert_allclose(omega, deep_water_omega, atol=0, rtol=ld.RELATIVE_TOLERANCE)

    # Capillary waves
    k = np.linspace(10000, 100000, 10)
    depth = np.full(10, 1000.0)
    capillary_omega = np.sqrt(ld.KINEMATIC_SURFACE_TENSION * k**3)
    omega = ld.intrinsic_dispersion_relation(k, depth)
    assert_allclose(omega, capillary_omega, atol=np.inf, rtol=ld.RELATIVE_TOLERANCE)

    # Test input verification
    try:
        solution = ld.intrinsic_dispersion_relation(
            1.0, 10, kinematic_surface_tension=-1
        )
        assert False
    except ValueError:
        pass

    try:
        solution = ld.intrinsic_dispersion_relation(1.0, 10, grav=-1)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        solution = ld.intrinsic_dispersion_relation(-1.0, 10)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        solution = ld.intrinsic_dispersion_relation(-1.0, np.array([10, 10]))
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
        rtol=ld.RELATIVE_TOLERANCE,
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
        rtol=ld.RELATIVE_TOLERANCE,
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
        rtol=ld.RELATIVE_TOLERANCE,
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
        rtol=ld.RELATIVE_TOLERANCE,
    )

    # For observer moving with the flow we retrerive the intrinsic dispersion relation
    kvec = np.array(((0.1, 0.0), (0.1, 0.0)))
    u = np.array((1.0, 0.0))
    obs = u.copy()
    k = np.linalg.norm(kvec, axis=-1)
    d = np.inf
    assert_allclose(
        ld.encounter_dispersion_relation(kvec, d, u, obs),
        ld.intrinsic_dispersion_relation(k, d),
        atol=0,
        rtol=ld.RELATIVE_TOLERANCE,
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

    shallow_water_c = np.sqrt(ld.GRAV * depth)
    c = ld.intrinsic_phase_speed(k, depth)

    assert_allclose(c, shallow_water_c, atol=0, rtol=ld.RELATIVE_TOLERANCE)

    # Deep water
    k = np.linspace(0.1, 1, 100)
    depth = np.full(100, 1000.0)
    deep_water_omega = np.sqrt(ld.GRAV / k)
    omega = ld.intrinsic_phase_speed(k, depth)
    assert_allclose(omega, deep_water_omega, atol=0, rtol=ld.RELATIVE_TOLERANCE)

    # Capillary waves
    k = np.linspace(10000, 100000, 10)
    depth = np.full(10, 1000.0)
    capillary_omega = np.sqrt(ld.KINEMATIC_SURFACE_TENSION * k)
    omega = ld.intrinsic_phase_speed(k, depth)
    assert_allclose(omega, capillary_omega, atol=np.inf, rtol=ld.RELATIVE_TOLERANCE)

    # Test input verification
    try:
        solution = ld.intrinsic_phase_speed(1.0, 10, kinematic_surface_tension=-1)
        assert False
    except ValueError:
        pass

    try:
        solution = ld.intrinsic_phase_speed(1.0, 10, grav=-1)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        solution = ld.intrinsic_phase_speed(-1.0, 10)
        assert False
    except ValueError:
        pass

    try:
        # error on negative wavenumber magnitude
        solution = ld.intrinsic_phase_speed(-1.0, np.array([10, 10]))
        assert False
    except ValueError:
        pass

    # nan for neg depth
    assert np.isnan(ld.intrinsic_phase_speed(1.0, -10)[0])
