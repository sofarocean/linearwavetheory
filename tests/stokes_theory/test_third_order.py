from linearwavetheory.stokes_theory._third_order_coeficients import (
    third_order_dispersion_correction,
    third_order_amplitude_correction,
)
from linearwavetheory.dispersion import intrinsic_dispersion_relation
import numpy as np
from linearwavetheory.settings import _GRAV
from tests.stokes_theory._utils import stokes_frequency_correction


def stokes_amplitude_correction(k, amp, d, grav=9.81, include_mean=False):
    sigma = np.tanh(k * d)
    eps = k * amp

    stokes = (3 + 8 * sigma**2 - 9 * sigma**4) / 16 / sigma**4 * eps**2

    return stokes


def test_dispersion_coef():
    depth = 10
    kd = np.linspace(0.1, 10, 301, endpoint=True)
    k = kd / depth
    omega = intrinsic_dispersion_relation(k, depth)
    freqs = omega / 2 / np.pi

    bound = np.zeros_like(freqs)
    stokes = np.zeros_like(freqs)

    # Without setdown
    include_mean_setdown, include_mean_flow, lagrangian = True, True, False

    for ii, freq in enumerate(freqs):
        ww1 = omega[ii]
        kk1 = k[ii]

        stokes[ii] = stokes_frequency_correction(k[ii], 1, depth) * 4
        bound[ii] = third_order_dispersion_correction(
            ww1,
            kk1,
            kk1,
            0.0,
            ww1,
            kk1,
            kk1,
            0.0,
            depth,
            _GRAV,
            include_mean_setdown,
            include_mean_flow,
            lagrangian,
        )
    np.testing.assert_allclose(bound, stokes, rtol=1e-3)


def test_amplitude_coef():
    depth = 10
    kd = np.linspace(1, 6, 301, endpoint=True)
    k = kd / depth
    omega = intrinsic_dispersion_relation(k, depth)
    freqs = omega / 2 / np.pi

    bound = np.zeros_like(freqs)
    stokes = np.zeros_like(freqs)
    include_mean_setdown, include_mean_flow, lagrangian = True, True, False

    for ii, freq in enumerate(freqs):
        ww1 = omega[ii]
        kk1 = k[ii]
        # Mostly note to self as this caused a lot of headaches.
        # To note - we have to multiply by 4 in this comparison. The explanation is convoluted and invoves:
        # - Stokes is measured in terms of total ampitude A squared -> 4 times the amplitude squared of complex
        #   components a
        #
        # - In general we have two solutions aa* and a*a (* denotes complex conjugate). This would reduce the
        #   amplitude by a factor of 2. However, for the self interacting case this factor two vanishes in the
        #   deterministic case. In the stochastic case, the factor two is still there.
        #
        stokes[ii] = 4 * stokes_amplitude_correction(k[ii], 1, depth, _GRAV)

        bound[ii] = third_order_amplitude_correction(
            ww1,
            kk1,
            kk1,
            0.0,
            ww1,
            kk1,
            kk1,
            0.0,
            depth,
            _GRAV,
            include_mean_setdown,
            include_mean_flow,
            lagrangian,
        )
        # exit(-1)

    np.testing.assert_allclose(bound, stokes, rtol=1e-3)
