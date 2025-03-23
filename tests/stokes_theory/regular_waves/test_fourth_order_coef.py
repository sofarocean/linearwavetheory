import tests.stokes_theory.regular_waves.terms as terms
from linearwavetheory.stokes_theory.regular_waves.eularian_velocity_amplitudes import (
    dimensionless_velocity_amplitude,
)


def x4n(dimensionless_depth, dimensionless_height, n):
    c2 = terms.cos(2)
    c3 = terms.cos(3)
    c4 = terms.cos(4)
    c1 = terms.cos(1)
    s2 = terms.sin(2)
    s1 = terms.sin(1)
    s3 = terms.sin(3)

    _al11 = terms.al11(dimensionless_depth, dimensionless_height) * c1
    _al22 = terms.al22(dimensionless_depth, dimensionless_height) * c2
    _al20 = terms.al20(dimensionless_depth, dimensionless_height)
    _al33 = terms.al33(dimensionless_depth, dimensionless_height) * c3
    _al31 = terms.al31(dimensionless_depth, dimensionless_height) * c1

    phi11 = dimensionless_velocity_amplitude(1, dimensionless_depth, 1, order=1)
    phi22 = dimensionless_velocity_amplitude(1, dimensionless_depth, 2, order=2)
    phi33 = dimensionless_velocity_amplitude(1, dimensionless_depth, 3, order=3)
    phi44 = dimensionless_velocity_amplitude(1, dimensionless_depth, 4, order=4)
    phi42 = dimensionless_velocity_amplitude(1, dimensionless_depth, 2, order=4) - phi22

    ch1 = terms.ch(1, dimensionless_depth, dimensionless_height)
    ch2 = terms.ch(2, dimensionless_depth, dimensionless_height)
    ch3 = terms.ch(3, dimensionless_depth, dimensionless_height)
    ch4 = terms.ch(4, dimensionless_depth, dimensionless_height)
    sh1 = terms.sh(1, dimensionless_depth, dimensionless_height)
    sh2 = terms.sh(2, dimensionless_depth, dimensionless_height)
    sh3 = terms.sh(3, dimensionless_depth, dimensionless_height)

    _u44 = phi44 * c4 * ch4
    _u42 = phi42 * c2 * ch2

    d_dx_u11 = -phi11 * s1 * ch1
    d2_dx2_u11 = -phi11 * c1 * ch1
    d3_dx3_u11 = phi11 * s1 * ch1
    d_dx_u22 = -2 * phi22 * s2 * ch2
    d2_dx2_u22 = -4 * phi22 * c2 * ch2
    d_dx_u33 = -3 * phi33 * s3 * ch3

    d_dz_u11 = phi11 * c1 * sh1
    d2_dz2_u11 = phi11 * c1 * ch1
    d3_dz3_u11 = phi11 * c1 * sh1
    d_dz_u22 = 2 * phi22 * c2 * sh2
    d2_dz2_u22 = 4 * phi22 * c2 * ch2
    d_dz_u33 = 3 * phi33 * c3 * sh3

    d2_dzdx_u11 = -phi11 * s1 * sh1
    d2_dzdx_u22 = -4 * phi22 * s2 * sh2
    d3_dzdx2_u11 = -phi11 * c1 * sh1
    d3_dz2dx_u11 = -phi11 * s1 * ch1

    disper = terms.dispersion(dimensionless_depth, dimensionless_height)

    _x11 = terms.x11(dimensionless_depth, dimensionless_height) * s1
    _x22 = terms.x22(dimensionless_depth, dimensionless_height) * s2
    _x22_disp = 2 * terms.x22(dimensionless_depth, dimensionless_height) * c2
    _x31 = terms.x31(dimensionless_depth, dimensionless_height) * s1
    _x33 = terms.x33(dimensionless_depth, dimensionless_height) * s3

    _ul4n = (
        disper * _x22_disp  # T0.
        + _x11 * d_dx_u33  # T1.
        + _x22 * d_dx_u22  # T2.
        + _x31 * d_dx_u11  # T3a.
        + _x33 * d_dx_u11  # T3b.
        + _al11 * d_dz_u33  # T4
        + _al22 * d_dz_u22  # T5a
        + _al20 * d_dz_u22  # T5b
        + _al31 * d_dz_u11  # T6a
        + _al33 * d_dz_u11  # 6b
        + _x11**2 * d2_dx2_u22 / 2  # T7.
        + _x22 * _x11 * d2_dx2_u11  # T8.
        + _al11**2 * d2_dz2_u22 / 2  # T9.
        + _al22 * _al11 * d2_dz2_u11  # T10a.
        + _al20 * _al11 * d2_dz2_u11  # T10b.
        + _al11 * _x11 * d2_dzdx_u22  # T11
        + _al22 * _x11 * d2_dzdx_u11  # T12a
        + _al20 * _x11 * d2_dzdx_u11  # T12b
        + _al11 * _x22 * d2_dzdx_u11  # T13
        + _al11**2 * _x11 * d3_dz2dx_u11 / 2  # T14
        + _al11 * _x11**2 * d3_dzdx2_u11 / 2  # T15
        + _al11**3 / 6 * d3_dz3_u11  # T16
        + _x11**3 / 6 * d3_dx3_u11  # T17
        + _u42  # T18
        + _u44  # T19
    )

    if n > 0:
        _x4n = -terms.fourier_amplitude(_ul4n, n, "cos") / n
    else:
        _x4n = terms.fourier_amplitude(_ul4n, n, "cos")
    return _x4n


def eta4n(dimensionless_depth, dimensionless_height, n):
    c2 = terms.cos(2)
    c4 = terms.cos(4)
    c1 = terms.cos(1)
    s2 = terms.sin(2)
    s1 = terms.sin(1)
    s3 = terms.sin(3)

    a11 = terms.a11(dimensionless_depth, dimensionless_height)
    a22 = terms.a22(dimensionless_depth, dimensionless_height)
    a33 = terms.a33(dimensionless_depth, dimensionless_height)
    a31 = terms.a31(dimensionless_depth, dimensionless_height)
    a44 = terms.a44(dimensionless_depth, dimensionless_height)
    a42 = terms.a42(dimensionless_depth, dimensionless_height)

    _a44 = a44 * c4
    _a42 = a42 * c2

    d_dx_eta11 = -a11 * s1
    d_dx_eta22 = -2 * a22 * s2
    d_dx_eta33 = -3 * a33 * s3
    d_dx_eta31 = -a31 * s1
    d2_dx2_eta11 = -a11 * c1
    d2_dx2_eta22 = -4 * a22 * c2
    d3_dx3_eta11 = +a11 * s1

    _x11 = terms.x11(dimensionless_depth, dimensionless_height) * s1
    _x22 = terms.x22(dimensionless_depth, dimensionless_height) * s2
    _x31 = terms.x31(dimensionless_depth, dimensionless_height) * s1
    _x33 = terms.x33(dimensionless_depth, dimensionless_height) * s3

    _eta4n = (
        _a44  # T1a
        + _a42  # T1b
        + _x11 * d_dx_eta33  # T2a
        + _x11 * d_dx_eta31  # T2b
        + _x22 * d_dx_eta22  # T3
        + _x33 * d_dx_eta11  # T4a
        + _x31 * d_dx_eta11  # T4b
        + _x11**2 * d2_dx2_eta22 / 2  # T5
        + _x11 * _x22 * d2_dx2_eta11  # T6
        + _x11**3 / 6 * d3_dx3_eta11  # T7
    )

    return terms.fourier_amplitude(_eta4n, n, "cos")


def x40(dimensionless_depth, dimensionless_height):
    return x4n(dimensionless_depth[:, None], dimensionless_height, 0)


def x42(dimensionless_depth, dimensionless_height):
    return x4n(dimensionless_depth[:, None], dimensionless_height, 2)


def x44(dimensionless_depth, dimensionless_height):
    return x4n(dimensionless_depth[:, None], dimensionless_height, 4)


def eta40(dimensionless_depth, dimensionless_height):
    return eta4n(dimensionless_depth[:, None], dimensionless_height, 0)


def eta42(dimensionless_depth, dimensionless_height):
    return eta4n(dimensionless_depth[:, None], dimensionless_height, 2)


def eta44(dimensionless_depth, dimensionless_height):
    return eta4n(dimensionless_depth[:, None], dimensionless_height, 4)


def target_eta40(dimensionless_depth, dimensionless_height):
    return terms.al40(dimensionless_depth, dimensionless_height)


def target_eta42(dimensionless_depth, dimensionless_height):
    return terms.al42(dimensionless_depth, dimensionless_height)


def target_eta44(dimensionless_depth, dimensionless_height):
    return terms.al44(dimensionless_depth, dimensionless_height)


def test_coef_eta40():
    terms.validate(target_eta40, eta40, "eta40", False)


def test_coef_eta42():
    terms.validate(target_eta42, eta42, "eta42", False)


def test_coef_eta44():
    terms.validate(target_eta44, eta44, "eta44", False)


def test_coef_ul40():
    terms.validate(terms.ul40, x40, "stokes drift 4th order", False)


def test_coef_x42():
    terms.validate(terms.x42, x42, "x42", False)


def test_coef_x44():
    terms.validate(terms.x44, x44, "x44", False)
