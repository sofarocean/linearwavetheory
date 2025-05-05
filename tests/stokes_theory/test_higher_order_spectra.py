from linearwavetheory.stokes_theory._higher_order_spectra import (
    bilinear_interpolation,
    nonlinear_wave_spectra_1d,
    nonlinear_wave_spectra_2d,
)
from linearwavetheory.settings import stokes_theory_options
from ._utils import get_spectrum
from scipy.integrate import trapezoid

import numpy as np
from linearwavetheory.dispersion import inverse_intrinsic_dispersion_relation
from tests.stokes_theory._utils import integrate, integrate_over_angles


def get_spec(peak_freq=0.1, hm0=1, dirwidth=10, freqwidth=0.01):
    freq = np.linspace(0.01, 1, 100)
    angle = np.linspace(0, 360, 36, endpoint=False)

    m0 = (hm0 / 4) ** 2

    theta0 = 0.0
    freq0 = peak_freq

    dir_func = np.exp(-((angle - theta0) ** 2) / dirwidth**2)
    dir_func[dir_func < 0] = 0

    delta_backward = (np.diff(angle, prepend=angle[-1]) + 180) % (360) - 180
    delta_forward = (np.diff(angle, append=angle[0]) + 180) % (360) - 180
    delta = (delta_backward + delta_forward) / 2

    dir_func = dir_func / np.sum(dir_func * delta)

    freq_func = np.exp(-((freq - freq0) ** 2) / (freqwidth**2))
    freq_func = m0 * freq_func / trapezoid(freq_func, freq)

    return freq, angle, np.outer(freq_func, dir_func)


def stokes_sum_ampitude(k, d):
    w = np.tanh(k * d)
    return k * (3 - w**2) / 4 / w**3


def test_estimate_bound_spectrum():
    peak_freq = 0.15
    options = stokes_theory_options(
        reference_frame="eulerian",
        include_nonlinear_dispersion=False,
        include_nonlinear_amplitude_correction=False,
        include_bound_waves=True,
        wave_driven_flow_included_in_mean_flow=True,
        wave_driven_setup_included_in_mean_depth=True,
        include_sum_interactions=True,
        include_difference_interactions=False,
        include_eulerian_contribution_in_drift=True,
    )

    for depth in np.linspace(10, 1000, 10):

        k = inverse_intrinsic_dispersion_relation(2 * np.pi * peak_freq, depth)
        eps = 0.2
        amp = eps / k

        stokes_amp = stokes_sum_ampitude(k, depth) * amp**2
        stokes_energy = (
            stokes_amp**2
        )  # Note that <a a a a> = 2 <a a>^2 for a Gaussian variable, which is why we loose the
        # factor of 1/2 here.

        hm0 = 4 * np.sqrt(amp**2 / 2)
        freq, angle, spec = get_spec(peak_freq, hm0=hm0)
        angular_freq = freq * np.pi * 2
        bound = nonlinear_wave_spectra_1d(
            angular_freq, angle, spec / 2 / np.pi, depth, nonlinear_options=options
        )

        bound_energy = trapezoid(bound, angular_freq)

        ratio = bound_energy / stokes_energy[0]
        np.testing.assert_allclose(ratio, 1, atol=1e-2, rtol=1e-2)


def test_bilinear_interpolation():
    freq, angle, spec = get_spec(hm0=4)

    freq_int = np.linspace(0.01, 2, 200)
    angle_int = np.linspace(0, 360, 200, endpoint=False)

    intp = bilinear_interpolation(freq, angle, spec, freq_int, angle_int)

    m0 = integrate(angle_int, freq_int, intp)
    np.testing.assert_allclose(m0, 1, atol=1e-2, rtol=1e-2)

    assert intp.shape == (len(freq_int), len(angle_int))
    assert intp.dtype == spec.dtype


def test_consistency_for_angular_integration():
    """
    Test the consistency of the angular spectrum implemantion for bound waves. When integrated over angles it should not
    matter if we use the 1d or first the 2d and then integrate over angles.
    :return:
    """
    steepness = 0.1
    deepness = np.linspace(1, 3, 5)
    peak_frequency = 0.1

    intrinsic_angular_frequency, angle, spec, k, depth = get_spectrum(
        steepness, peak_frequency, deepness, kind="sea"
    )
    options = stokes_theory_options()

    bound_1d = nonlinear_wave_spectra_1d(
        intrinsic_angular_frequency, angle, spec, depth, nonlinear_options=options
    )

    bound_2d = nonlinear_wave_spectra_2d(
        intrinsic_angular_frequency, angle, spec, depth, nonlinear_options=options
    )

    integrated_bound_2d = integrate_over_angles(angle, bound_2d)
    np.testing.assert_allclose(integrated_bound_2d, bound_1d, rtol=1e-2, atol=1e-2)
