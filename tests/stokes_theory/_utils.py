import numpy
import numpy as np

from linearwavetheory._parametricspectra import (
    create_dir_shape,
    create_freq_shape,
    parametric_directional_spectrum,
)
from scipy.integrate import trapezoid
from linearwavetheory.settings import _GRAV


def integrate_over_angles(angle, spec):
    delta_backward = (np.diff(angle, prepend=angle[-1]) + 180) % (360) - 180
    delta_forward = (np.diff(angle, append=angle[0]) + 180) % (360) - 180

    delta = (delta_backward + delta_forward) / 2
    m0 = np.sum(spec * delta, axis=-1)
    return m0


def integrate(angle, freq, spec):
    delta_backward = (np.diff(angle, prepend=angle[-1]) + 180) % (360) - 180
    delta_forward = (np.diff(angle, append=angle[0]) + 180) % (360) - 180

    delta = (delta_backward + delta_forward) / 2
    m0 = trapezoid(np.sum(spec * delta, axis=1), freq)
    return m0


def stokes_frequency_correction(
    k, amp, d, grav=9.81, include_mean=False, include_setdown=False
):
    sigma = np.tanh(k * d)
    omega = np.sqrt(grav * k * sigma)
    eps = k * amp

    stokes = (
        0.5 * (9 - 10 * sigma**2 + 9 * sigma**4) / 8 / sigma**4 * eps**2 * omega
    )

    if include_mean:
        c = omega / k
        stokes += stokes_mean_cur(k, d) * k

    return stokes


def stokes_mean_cur(wavenumber, depth):
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

    lhs = -cg * K / (_GRAV * depth - cg**2)

    return lhs


def get_spectrum(steepness, peak_freq, deepness, kind="swell"):
    """
    Generate a spectrum with given steepness (epsilon), peak_frequency and shallownes (kd). The returned spectrum is
    formulated in terms of m^2 rad/s /deg

    :param steepness: how steep is the spectrum (larger is steeper, typically 0.1 is steep)
    :param peak_freq:
    :param deepness: (how deep is the water relatively, kd>>1 is deep, kd<<1 is shallow, kd=1 is intermediate
    :param kind: what type of spectrum (sea/swell)
    :return:
    """
    deepness = np.atleast_1d(deepness)
    w_peak = 2 * np.pi * peak_freq
    wavenumber = w_peak**2 / _GRAV / np.tanh(deepness)
    depth = deepness / wavenumber

    nspec = len(deepness)
    nfreq = 50
    nangle = 36

    spectrum = np.zeros((nspec, nfreq, nangle))

    frequency = np.linspace(0.02, 1, 50, endpoint=True) * peak_freq * 5
    intrinsic_angular_frequency = 2 * np.pi * frequency

    angle_degrees = np.linspace(0, 360, 36, endpoint=False)
    for ii in range(nspec):

        amp = steepness / wavenumber[ii]
        hm0 = 4 * np.sqrt(amp**2 / 2)

        if kind == "swell":
            dir_shape = create_dir_shape(0, 1, "cosN")
            freq_shape = create_freq_shape(
                1 / peak_freq,
                hm0,
                "gaussian",
                standard_deviation_hertz=0.001 * peak_freq,
            )
        else:
            dir_shape = create_dir_shape(0, 30, "cosN")
            freq_shape = create_freq_shape(1 / peak_freq, hm0, "jonswap")

        spectrum[ii, :, :] = parametric_directional_spectrum(
            frequency, angle_degrees, freq_shape, dir_shape
        )

    spectrum = spectrum / 2 / np.pi

    return intrinsic_angular_frequency, angle_degrees, spectrum, wavenumber, depth
