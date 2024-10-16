from linearwavetheory.stokes_theory._higher_order_spectra import (
    bilinear_interpolation,
    bound_wave_spectra_1d,
)
import numpy as np
from linearwavetheory.dispersion import inverse_intrinsic_dispersion_relation


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
    freq_func = m0 * freq_func / np.trapz(freq_func, freq)

    return freq, angle, np.outer(freq_func, dir_func)


def integrate(angle, freq, spec):
    delta_backward = (np.diff(angle, prepend=angle[-1]) + 180) % (360) - 180
    delta_forward = (np.diff(angle, append=angle[0]) + 180) % (360) - 180

    delta = (delta_backward + delta_forward) / 2
    m0 = np.trapz(np.sum(spec * delta, axis=1), freq)
    return m0


def stokes_sum_ampitude(k, d):
    w = np.tanh(k * d)
    return k * (3 - w**2) / 4 / w**3


def test_estimate_bound_spectrum():
    peak_freq = 0.15

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

        bound = bound_wave_spectra_1d(
            freq, angle, spec, depth, contributions="sum", kind="eulerian"
        )

        bound_energy = np.trapz(bound, freq)

        ratio = bound_energy / stokes_energy[0]
        print(ratio, depth, bound_energy, stokes_energy[0])
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
