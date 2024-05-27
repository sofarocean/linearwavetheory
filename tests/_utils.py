import numpy
import numpy as np


def PearsonMoskowitzITTC1978(frequencies, Hs, Tm01, g=9.81):

    angular_frequencies = numpy.pi * 2 * frequencies

    Af = 173 * Hs**2 / (Tm01**4)
    Bf = 691 / (Tm01**4)
    spec = (
        Af * angular_frequencies ** (-5) * numpy.exp(-Bf * angular_frequencies ** (-4))
    )
    # Return frequency spectrum, multiply with Jacobian
    return spec * 2 * numpy.pi


def raised_cosine(directions, mean_direction, width):
    lookup_table = [
        [1.0, 37.5],
        [2.0, 31.5],
        [3.0, 27.6],
        [4.0, 24.9],
        [5.0, 22.9],
        [6.0, 21.2],
        [7.0, 19.9],
        [8.0, 18.8],
        [9.0, 17.9],
        [10.0, 17.1],
        [15.0, 14.2],
        [20.0, 12.4],
        [30.0, 10.2],
        [40.0, 8.9],
        [50.0, 8.0],
        [60.0, 7.3],
        [70.0, 6.8],
        [80.0, 6.4],
        [90.0, 6.0],
        [100.0, 5.7],
        [200.0, 4.0],
        [400.0, 2.9],
        [800.0, 2.0],
    ]

    lookup_table = numpy.array(lookup_table).transpose()
    lookup_table = numpy.flip(lookup_table, 1)

    if width <= lookup_table[1, 0]:
        power = lookup_table[0, 0]
    elif width >= lookup_table[1, -1]:
        power = lookup_table[0, -1]
    else:
        power = np.interp(width, lookup_table[1, :], lookup_table[0, :])
        # interp = scipy.interpolate.interp1d(lookup_table[1, :], lookup_table[0, :])
        # power = interp(width)

    mutual_angle = (mean_direction - directions + 180) % 360 - 180

    with numpy.errstate(invalid="ignore", divide="ignore"):
        D = numpy.where(
            numpy.abs(mutual_angle) < 90,
            numpy.cos(mutual_angle * numpy.pi / 180.0) ** power,
            0.0,
        )
    delta = directions[2] - directions[1]
    return D / (numpy.sum(D) * delta)


def spectrum2D(waveheight, meandir, tm01, spread, frequencies, dir):
    spec = (
        PearsonMoskowitzITTC1978(frequencies, waveheight, tm01)[:, None]
        * raised_cosine(dir, meandir, spread)[None, :]
    )
    return spec
