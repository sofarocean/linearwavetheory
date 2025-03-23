import tests.stokes_theory.regular_waves.terms as terms


def x2n(dimensionless_depth, dimensionless_height, n):
    c2 = terms.cos(2)
    c1 = terms.cos(1)
    s1 = terms.sin(1)

    _eta11 = terms.a11(dimensionless_depth, dimensionless_height) * c1
    _x11 = terms.x11(dimensionless_depth, dimensionless_height) * s1
    _d_dx_u11 = -terms.u11(dimensionless_depth, dimensionless_height) * s1
    _d_dz_u11 = terms.w11(dimensionless_depth, dimensionless_height) * c1

    _u22 = terms.u22(dimensionless_depth, dimensionless_height) * c2

    _u22l = _u22 + _eta11 * _d_dz_u11 + _x11 * _d_dx_u11

    amp = terms.fourier_amplitude(_u22l, n, "cos")

    if n == 0:
        return amp
    else:
        return -amp / 2


def x22(dimensionless_depth, dimensionless_height):
    return x2n(dimensionless_depth, dimensionless_height, 2)


def ul20(dimensionless_depth, dimensionless_height):
    return x2n(dimensionless_depth, dimensionless_height, 0)


def target_x22(dimensionless_depth, dimensionless_height):
    return terms.x22(dimensionless_depth, dimensionless_height)


def target_ul20(dimensionless_depth, dimensionless_height):
    return terms.ul20(dimensionless_depth, dimensionless_height)


def test_x22():
    terms.validate(target_x22, x22, "x22", False)


def test_x20():
    terms.validate(target_ul20, ul20, "x20", False)
