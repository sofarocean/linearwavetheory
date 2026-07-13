import src.linearwavetheory.encounter_spectra as es
import src.linearwavetheory.dispersion as ld
import numpy as np
from numpy import trapezoid
from numpy.testing import assert_allclose
from src.linearwavetheory import settings
from src.linearwavetheory.settings import physics_options
from tests._utils import spectrum2D


def test_encounter_spectrum_to_intrinsic_1d_roundtrip():
    """
    Doppler-shift a known intrinsic spectrum forward to encounter frequency, then invert it back; the recovered
    intrinsic frequencies should match the original grid, across regimes and both signs of Delta.
    """
    for regime, depth in [("deep", 5000.0), ("intermediate", 20.0), ("shallow", 5.0)]:
        opts = physics_options(wave_regime=regime)
        for current_speed in [0.0, 0.5, -0.3]:
            k = np.linspace(0.001, 0.05, 20)
            intrinsic_frequency = ld.intrinsic_dispersion_relation(
                k, depth, physics_options=opts
            )
            kvec = np.column_stack([k, np.zeros_like(k)])
            encounter_frequency = ld.encounter_dispersion_relation(
                kvec,
                depth,
                ambient_current_velocity=(current_speed, 0.0),
                physics_options=opts,
            )
            valid = encounter_frequency >= 0
            variance_density = np.ones_like(k)

            sigma, density = es.encounter_spectrum_to_intrinsic_1d(
                encounter_frequency[valid],
                variance_density[valid],
                current_speed,
                0.0,
                0.0,
                depth=depth,
                physics_options=opts,
            )
            assert_allclose(
                sigma,
                intrinsic_frequency[valid],
                atol=0,
                rtol=100 * settings._RELATIVE_TOLERANCE,
                err_msg=f"regime={regime} current_speed={current_speed}",
            )


def test_encounter_spectrum_to_intrinsic_1d_zero_current():
    """
    With no current, the spectrum should pass through unchanged (Jacobian is exactly 1).
    """
    freq = np.linspace(0.2, 1.5, 10)
    density = np.linspace(1.0, 2.0, 10)

    sigma, out_density = es.encounter_spectrum_to_intrinsic_1d(
        freq, density, 0.0, 0.0, 0.0, depth=50.0
    )
    reference_k = ld.inverse_intrinsic_dispersion_relation(freq, 50.0)
    reference_sigma = ld.intrinsic_dispersion_relation(reference_k, 50.0)

    assert_allclose(sigma, reference_sigma, atol=0, rtol=settings._RELATIVE_TOLERANCE)
    assert_allclose(out_density, density, atol=0, rtol=settings._RELATIVE_TOLERANCE)


def test_encounter_spectrum_to_intrinsic_1d_energy_conservation():
    """
    Away from any blocking region, the density transform should conserve total variance (energy) between the
    encounter and intrinsic frames.
    """
    depth = 50.0
    current_speed = 0.5

    k = np.linspace(0.01, 0.5, 500)
    kvec = np.column_stack([k, np.zeros_like(k)])
    encounter_frequency = ld.encounter_dispersion_relation(
        kvec, depth, ambient_current_velocity=(current_speed, 0.0)
    )
    density = np.exp(-0.5 * ((encounter_frequency - 1.0) / 0.2) ** 2)

    sigma, out_density = es.encounter_spectrum_to_intrinsic_1d(
        encounter_frequency, density, current_speed, 0.0, 0.0, depth=depth
    )

    e_omega = trapezoid(density, encounter_frequency)
    e_sigma = trapezoid(out_density, sigma)
    assert_allclose(e_sigma, e_omega, atol=0, rtol=1e-3)


def test_encounter_spectrum_to_intrinsic_1d_nan_propagation():
    """
    Frequencies for which no root exists (blocked by an opposing current) should return NaN in both outputs,
    rather than being dropped or silently zeroed.
    """
    freq = np.linspace(0.1, 50.0, 10)
    density = np.ones_like(freq)

    sigma, out_density = es.encounter_spectrum_to_intrinsic_1d(
        freq, density, -5.0, 0.0, 0.0, depth=20.0
    )
    assert sigma.shape == freq.shape
    assert np.any(np.isnan(sigma))
    assert np.array_equal(np.isnan(sigma), np.isnan(out_density))
    # the low-frequency end should still exist
    assert np.isfinite(sigma[0])


def test_regrid_1d_spectrum():
    """
    The private regridding helper should handle unsorted input, NaN entries, and out-of-range target points.
    """
    sigma = np.array([3.0, 1.0, np.nan, 2.0])
    density = np.array([30.0, 10.0, 999.0, 20.0])
    target = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

    result = es._regrid_1d_spectrum(sigma, density, target)

    assert np.isnan(result[0])  # below range
    assert_allclose(result[1], 10.0)
    assert_allclose(result[2], 15.0)
    assert_allclose(result[3], 20.0)
    assert_allclose(result[4], 30.0)
    assert np.isnan(result[5])  # above range

    # all-NaN sigma -> all-NaN output
    all_nan_result = es._regrid_1d_spectrum(np.full(3, np.nan), np.zeros(3), target)
    assert np.all(np.isnan(all_nan_result))


def test_encounter_spectrum_to_intrinsic_2d_matches_1d_single_bin():
    """
    A 2D spectrum with all its energy concentrated in a single direction bin should, after regridding, match
    calling the 1D conversion directly for that bin's direction.
    """
    depth = 50.0
    current_speed = 0.5
    current_direction_degrees = 10.0

    freq = np.linspace(0.1, 2.0, 40)
    directions = np.array([0.0, 45.0, 90.0, 180.0])

    variance_density = np.zeros((len(freq), len(directions)))
    variance_density[:, 1] = np.exp(-0.5 * ((freq - 1.0) / 0.3) ** 2)

    result_2d = es.encounter_spectrum_to_intrinsic_2d(
        freq,
        directions,
        variance_density,
        current_speed,
        current_direction_degrees,
        depth=depth,
    )

    sigma_1d, density_1d = es.encounter_spectrum_to_intrinsic_1d(
        freq,
        variance_density[:, 1],
        current_speed,
        current_direction_degrees,
        directions[1],
        depth=depth,
    )
    expected = es._regrid_1d_spectrum(sigma_1d, density_1d, freq)

    assert_allclose(result_2d[:, 1], expected, atol=0, rtol=1e-10, equal_nan=True)

    # the bins with no input energy should carry no energy in the output (finite entries are exactly zero)
    other_columns = result_2d[:, [0, 2, 3]]
    assert np.all(np.isnan(other_columns) | (other_columns == 0.0))


def test_encounter_spectrum_to_intrinsic_2d_realistic_spectrum():
    """
    Smoke test on a realistic (Pierson-Moskowitz x raised-cosine) 2D spectrum: check output shape and that no
    unexpected all-NaN result is produced for a modest current.
    """
    frequencies = np.linspace(0.03, 0.3, 30)
    directions = np.linspace(0, 360, 36, endpoint=False)
    variance_density = spectrum2D(2.0, 90.0, 8.0, 30.0, frequencies, directions)

    angular_frequency = 2 * np.pi * frequencies
    # Jacobian from Hz to rad/s for the input density itself is not needed here -- we only check plumbing.
    result = es.encounter_spectrum_to_intrinsic_2d(
        angular_frequency,
        directions,
        variance_density,
        relative_current_speed=0.3,
        relative_current_direction_degrees=90.0,
        depth=50.0,
    )
    assert result.shape == (len(angular_frequency), len(directions))
    assert np.any(np.isfinite(result))
