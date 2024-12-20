from linearwavetheory.stokes_theory._nonlinear_dispersion import nonlinear_dispersion
from ._utils import stokes_frequency_correction, get_spectrum
from linearwavetheory.settings import _GRAV
import numpy as np


def test_nonlinear_disp():
    """
    Test the nonlinear dispersion correction for a swell spectrum. This should be very close to the nonlinear correction
    from Stokes theory for a monochromatic wave of similar energy and frequency.
    :return:
    """
    peak_freq = 0.1
    peak_angular_freq = peak_freq * 2 * np.pi
    epsilon = 0.1
    kd = np.linspace(0.5, 3, 26, endpoint=True)

    angular_freq, angle, spec, k, depth = get_spectrum(
        epsilon, peak_freq, kd, kind="swell"
    )

    index_peak_freq = np.argmin(np.abs(angular_freq - peak_angular_freq))
    index_peak_dir = 0

    dispersion = nonlinear_dispersion(angular_freq, angle, spec, depth)[
        :, index_peak_freq, index_peak_dir
    ]

    stokes_estimate = stokes_frequency_correction(k, epsilon / k, depth)

    np.testing.assert_allclose(dispersion, stokes_estimate, rtol=1e-2)
