import numpy as np
from linearwavetheory.stokes_theory.regular_waves.eularian_velocity_amplitudes import (
    dimensionless_velocity_amplitude,
)

import linearwavetheory.stokes_theory.regular_waves.nonlinear_dispersion as nd

import linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes as eea
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lda


def phase():
    return np.linspace(0, 2 * np.pi, 1000, endpoint=True)[None, :]


def cos(n):
    return np.cos(n * phase())


def sin(n):
    return np.sin(n * phase())


def ch(n, dimensionless_depth, dimensionless_height):
    return np.cosh(n * (dimensionless_depth + dimensionless_height)) / np.cosh(
        n * dimensionless_depth
    )


def sh(n, dimensionless_depth, dimensionless_height):
    return np.sinh(n * (dimensionless_depth + dimensionless_height)) / np.cosh(
        n * dimensionless_depth
    )


def fourier_amplitude(term, n, kind):

    if n == 0:
        fac = 1
    else:
        fac = 2

    if kind == "cos":
        amp = fac * np.trapezoid(term * cos(n), phase()) / 2 / np.pi
    elif kind == "sin":
        amp = fac * np.trapezoid(term * sin(n), phase()) / 2 / np.pi

    return amp


def mu(dimensionless_depth):
    return np.tanh(dimensionless_depth)


def u11(dimensionless_depth, dimensionless_height):
    return ch(
        1, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 1, order=1)


def w11(dimensionless_depth, dimensionless_height):
    return sh(
        1, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 1, order=1)


def u22(dimensionless_depth, dimensionless_height):
    return ch(
        2, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 2, order=2)


def w22(dimensionless_depth, dimensionless_height):
    return sh(
        2, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 2, order=2)


def u33(dimensionless_depth, dimensionless_height):
    return ch(
        3, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 3, order=3)


def w33(dimensionless_depth, dimensionless_height):
    return sh(
        3, dimensionless_depth, dimensionless_height
    ) * dimensionless_velocity_amplitude(1, dimensionless_depth, 3, order=3)


def a11(dimensionless_depth, dimensionless_height):
    return eea.eta11(dimensionless_depth, dimensionless_height)


def a22(dimensionless_depth, dimensionless_height):
    return eea.eta22(dimensionless_depth, dimensionless_height)


def a33(dimensionless_depth, dimensionless_height):
    return eea.eta33(dimensionless_depth, dimensionless_height)


def al20(dimensionless_depth, dimensionless_height):
    return lda.eta20(dimensionless_depth, dimensionless_height)


def al11(dimensionless_depth, dimensionless_height):
    return lda.eta11(dimensionless_depth, dimensionless_height)


def al22(dimensionless_depth, dimensionless_height):
    return lda.eta22(dimensionless_depth, dimensionless_height)


def al33(dimensionless_depth, dimensionless_height):
    return lda.eta33(dimensionless_depth, dimensionless_height)


def al31(dimensionless_depth, dimensionless_height):
    return lda.eta31(dimensionless_depth, dimensionless_height)


def ul20(dimensionless_depth, dimensionless_height):
    return lda.ul20(dimensionless_depth, dimensionless_height)


def ul40(dimensionless_depth, dimensionless_height):
    return lda.ul40(dimensionless_depth, dimensionless_height)


def a31(dimensionless_depth, dimensionless_height):
    return eea.eta31(dimensionless_depth, dimensionless_height)


def a42(dimensionless_depth, dimensionless_height):
    return eea.eta42(dimensionless_depth, dimensionless_height)


def a44(dimensionless_depth, dimensionless_height):
    return eea.eta44(dimensionless_depth, dimensionless_height)


def al40(dimensionless_depth, dimensionless_height):
    return lda.eta40(dimensionless_depth, dimensionless_height)


def al44(dimensionless_depth, dimensionless_height):
    return lda.eta44(dimensionless_depth, dimensionless_height)


def al42(dimensionless_depth, dimensionless_height):
    return lda.eta42(dimensionless_depth, dimensionless_height)


def dispersion(dimensionless_depth, dimensionless_height):
    return nd.w2l(dimensionless_depth, dimensionless_height)


def x11(dimensionless_depth, dimensionless_height):
    return -ch(1, dimensionless_depth, dimensionless_height) / mu(dimensionless_depth)


def x31(dimensionless_depth, dimensionless_height):
    return lda.x31(dimensionless_depth, dimensionless_height)


def x33(dimensionless_depth, dimensionless_height):
    return lda.x33(dimensionless_depth, dimensionless_height)


def x22(dimensionless_depth, dimensionless_height):
    return lda.x22(dimensionless_depth, dimensionless_height)


def x42(dimensionless_depth, dimensionless_height):
    return lda.x42(dimensionless_depth, dimensionless_height)


def x44(dimensionless_depth, dimensionless_height):
    return lda.x44(dimensionless_depth, dimensionless_height)


def validate(target, new, name, plot=False):
    kd = np.linspace(0.1, 5, 100)[:, None]
    height = [0, -0.5]

    _ok = True
    for z in height:
        ref = np.squeeze(target(kd, z))
        res = np.squeeze(new(kd, z))
        diff = np.max(np.abs(ref - res) / np.abs(ref))
        if diff > 1e-8:
            _ok = False
            print(f"- {name} Height: {z} -> Max diff: {diff}")
            raise AssertionError(
                f"Validation failed for {name} at height {z}: max diff = {diff:.2e}"
            )

        if plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(kd, ref, "ko")
            plt.plot(kd, res)
            plt.title(f"{name}-z={z}")
            plt.show()

    if _ok:
        print(f"{name} OK")
    return _ok
