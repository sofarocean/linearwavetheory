from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gamma
import numpy
from linearwavetheory.settings import _GRAV
from typing import Literal
from scipy.integrate import trapezoid

# Added for convinience but hidden. May remove at any time. Do not rely on it being present


class DirShape(ABC):
    """
    Abstract base class for the directional shape of the wave spectrum. Note: Never instantiate this class directly,
    but use one of the subclasses.
    """

    def __init__(
        self, mean_direction_degrees: float = 0, width_degrees: float = 28.64, **kwargs
    ):
        """
        Create a directional shape with the given mean direction and width.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees
        """
        self.width_degrees = (
            width_degrees  #: Width of the directional distribution in degrees
        )
        self.mean_direction_degrees = mean_direction_degrees  #: Mean direction of the waves in degrees in the assumed
        # coordinate system.

    @abstractmethod
    def _normalization(self):
        pass

    def values(
        self, direction_degrees: numpy.ndarray, renormalize: bool = False
    ) -> numpy.ndarray:
        """
        Return the value of the directional distribution (in degrees**-1) at the given angles in the assumed coordinate
        system. If renormalize is True, the distribution is renormalized so that the discrete integral (midpoint rule)
        over the distribution over the given directions is 1.

        :param direction_degrees: directions in degrees
        :param renormalize: renormalize the distribution so that the discrete integral over the distribution is 1.
        :return: numpy array with the values of the directional distribution
        """
        data = self._values(direction_degrees)

        try:
            normalization = self._normalization()
            data = data * normalization
        except ValueError:
            normalization = 1
            renormalize = True

        if renormalize:
            # Renormalize so that the discretely integrated distribution is 1. First we need to estimate the bin size
            # for the given direction vector. We do this by calculating the forward and backward difference and taking
            # the average of the two. We need take into account that the direction vector is cyclic, so we need to
            # append the first value to the end and prepend the last value to the beginning and use modulo arithmetic.
            wrap = 360
            forward_diff = (
                numpy.diff(direction_degrees, append=direction_degrees[0]) + wrap / 2
            ) % wrap - wrap / 2
            backward_diff = (
                numpy.diff(direction_degrees, prepend=direction_degrees[-1]) + wrap / 2
            ) % wrap - wrap / 2
            bin_size = (forward_diff + backward_diff) / 2

            # With the binsize known, renormalize to 1
            data = data / numpy.sum(data * bin_size)

        return data

    @abstractmethod
    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        ...


class DirCosineN(DirShape):
    """
    Raised cosine directional shape, $D(\\theta)=A \\cos^n(\\theta)$ where the normalization constant $A$ is chosen such
    that $\\int_0^{2\\pi} D(\\theta) d\\theta = 1$ (see Holthuijsen, 2010, section 6.3.3).
    """

    def __init__(
        self, mean_direction_degrees: float = 0, width_degrees: float = 28.64, **kwargs
    ):
        """
        Create a raised cosine directional shape $D(\\theta)=A \\cos^n(\\theta-\\theta_0)$ with the given mean
        direction $\\theta_0$ and width $\\sigma_\\theta$.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees. The power $n$ in the distribution is
        estimated such that the directional width corresponds to the given width. Default is 28.64 degrees, which
        corresponds to n=2.

        """
        super(DirCosineN, self).__init__(
            mean_direction_degrees, width_degrees, **kwargs
        )

    def _normalization(self):
        power = self.width_degrees_to_power(self.width_degrees)
        values = (
            numpy.pi
            / 180
            * gamma(power / 2 + 1)
            / (gamma(power / 2 + 1 / 2) * numpy.sqrt(numpy.pi))
        )
        if np.isnan(values):
            raise ValueError(
                "Normalization constant is NaN. This is likely due to a very narrow directional spread."
            )
        else:
            return values

    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the directional distribution value at the given angles.
        :param direction_degrees: Direction in degrees in the assumed coordinate system.
        :param renormalize: If True, renormalize so that the discretely integrated distribution is 1. This is useful
        if the direction_degrees array is coarsely sampled, but we want to ensure that the discretely integrated
        distribution is 1.

        :return: Directional distribution value (degree^-1) at the given angles.
        """
        angle = (direction_degrees - self.mean_direction_degrees + 180) % 360 - 180
        power = self.width_degrees_to_power(self.width_degrees)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            data = numpy.where(
                numpy.abs(angle) <= 90,
                numpy.cos(angle * numpy.pi / 180) ** power,
                0,
            )
        return data

    @staticmethod
    def width_degrees_to_power(width_degrees):
        """
        Calculate power that gives the given width of the directional distribution. See Holthuijsen, 2010, section
        6.3.4. Note - this expression is adapted from eq 6.3.26 to use with a raised cosine distribution instead of a
        $\\cos^{2n}(2\\theta)$ distribution.

        :param width_degrees: Width of the directional distribution in degrees.
        """
        return 4 / ((numpy.pi * width_degrees / 90) ** 2) - 2

    @staticmethod
    def power_to_width_degrees(power):
        """
        Calulate width of the directional distribution for the given power. See Holthuijsen, 2010, section 6.3.4.
        Note - this expression is adapted from eq 6.3.26 to use with a raised cosine distribution instead of a
        $\\cos^{2n}(2\\theta)$ distribution.
        """
        return numpy.sqrt(4 / (power + 2)) * 90 / numpy.pi


class DirCosine2N(DirShape):
    """
    Raised cosine directional shape proposed by Longuet-Higgins,  1963, $D(\\theta)=A \\cos^2n(\\theta/2)$ where the
    normalization constant $A$ is chosen such that $\\int_0^{2\\pi} D(\\theta) d\\theta = 1$
    (see Holthuijsen, 2010, section 6.3.4).
    """

    def __init__(
        self, mean_direction_degrees: float = 0, width_degrees: float = 28.64, **kwargs
    ):
        """
        Create a raised cosine directional shape $D(\\theta)=A \\cos^n(\\theta-\\theta_0)$ with the given mean
        direction $\\theta_0$ and width $\\sigma_\\theta$.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees. The power $n$ in the distribution is
        estimated such that the directional width corresponds to the given width. Default is 28.64 degrees, which
        corresponds to n=2.

        """
        super(DirCosine2N, self).__init__(
            mean_direction_degrees, width_degrees, **kwargs
        )

    def _normalization(self):
        # See Holthuijsen, 2010, section 6.3.3
        power = self.width_degrees_to_power(self.width_degrees)
        return (
            numpy.pi
            / 180
            * gamma(power + 1)
            / (gamma(power + 1 / 2) * 2 * numpy.sqrt(numpy.pi))
        )

    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the directional distribution value at the given angles.
        :param direction_degrees: Direction in degrees in the assumed coordinate system.
        :param renormalize: If True, renormalize so that the discretely integrated distribution is 1. This is useful
        if the direction_degrees array is coarsely sampled, but we want to ensure that the discretely integrated
        distribution is 1.

        :return: Directional distribution value (degree^-1) at the given angles.
        """
        angle = (direction_degrees - self.mean_direction_degrees + 180) % 360 - 180
        power = self.width_degrees_to_power(self.width_degrees)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            data = numpy.where(
                numpy.abs(angle) <= 180,
                numpy.cos(angle / 2 * numpy.pi / 180) ** (2 * power),
                0,
            )
        return data

    @staticmethod
    def width_degrees_to_power(width_degrees):
        """
        Calculate power that gives the given width of the directional distribution. See Holthuijsen, 2010, eq 6.3.26.

        :param width_degrees: Width of the directional distribution in degrees.
        """
        return 2 / (width_degrees * numpy.pi / 180) ** 2 - 1

    @staticmethod
    def power_to_width_degrees(power):
        """
        Calulate width of the directional distribution for the given power. See Holthuijsen, 2010, eq 6.3.26.
        """
        return numpy.sqrt(2 / (power + 1)) * 180 / numpy.pi


class FreqShape(ABC):
    """
    Abstract base class for the frequency shape of the wave spectrum. Note: Never instantiate this class directly,
    but use one of the subclasses.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a frequency shape with the given peak frequency and variance density.

        :param peak_frequency_hertz: Peak frequency of the spectrum
        :param significant_waveheight_meter: significant waveheight defined as $4*\\sqrt(m0)$, where m0 is the variance
            of the spectrum.
        """
        self.peak_frequency_hertz = peak_frequency_hertz
        self.significant_waveheight_meter = significant_waveheight_meter

        self.frequency_min = kwargs.get("fmin", 0.5 * peak_frequency_hertz)
        self.frequency_max = kwargs.get("fmax", 3.0 * peak_frequency_hertz)
        self.frequency_bins = kwargs.get("nbins", 26)

    def frequency(self):
        return np.linspace(
            self.frequency_min, self.frequency_max, self.frequency_bins, endpoint=True
        )

    @property
    def m0(self) -> float:
        """
        Variance of the spectrum.
        :return:
        """
        return (self.significant_waveheight_meter / 4) ** 2

    def values(
        self, frequency_hertz: numpy.ndarray, renormalize=False
    ) -> numpy.ndarray:
        """
        Calculate the variance density (in m**2/Hz) at the given frequencies. If renormalize is True, the variance
        density is renormalized so that the discretely integrated spectrum (trapezoidal rule) will yield the specified
        significant waveheight exactly.

        :param frequency_hertz: frequencies in Hertz
        :param renormalize: renormalize so that the discretely integrated spectrum (trapezoidal rule) will yield the
            specified significant waveheight exactly.
        :return: numpy array with variance density (in m**2/Hz) at the given frequencies.
        """
        values = self._values(frequency_hertz)
        if renormalize:
            values = values * self.m0 / trapezoid(values, frequency_hertz)

        return values

    @abstractmethod
    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        ...


class FreqGaussian(FreqShape):
    """
    Gaussian frequency shape. Useful for testing and to model swell (with very narrow standard deviation).
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Gaussian frequency shape with the given peak frequency and significant wave height. The standard
        deviation of the Gaussian is set to 1/10 of the peak frequency (narrow), but may be overridden by the
        standard_deviation_hertz keyword argument.

        :param peak_frequency_hertz: Peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: Significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """

        super(FreqGaussian, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self.standard_deviation_hertz = kwargs.get(
            "standard_deviation_hertz", peak_frequency_hertz / 10
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        return (
            self.m0
            / self.standard_deviation_hertz
            / numpy.sqrt(2 * numpy.pi)
            * numpy.exp(
                -0.5
                * (frequency_hertz - self.peak_frequency_hertz) ** 2
                / self.standard_deviation_hertz**2
            )
        )


class FreqPhillips(FreqShape):
    """
    Phillips frequency shape as proposed by Phillips (1958).
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Phillips frequency shape with the given peak frequency and significant wave height.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqPhillips, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", _GRAV)

    @property
    def _alpha(self):
        """
        Scale parameter of the Phillips spectrum.
        """
        return (
            self.m0
            * 8
            * (numpy.pi) ** 4
            * self.peak_frequency_hertz**4
            / self._g**2
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Phillips variance-density spectrum with frequency in Hz as
        dependent variable.

        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz >= self.peak_frequency_hertz
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
        )
        return values


class FreqPiersonMoskowitz(FreqShape):
    """
    Pierson Moskowitz frequency shape as proposed by Pierson and Moskowitz (1964). Commonly used for wind-generated
    waves in deep water.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Pierson Moskowitz frequency shape with the given peak frequency and significant wave height.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqPiersonMoskowitz, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", _GRAV)

    @property
    def _alpha(self):
        return (
            self.m0 * 5 * (2 * numpy.pi * self.peak_frequency_hertz) ** 4 / self._g**2
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Pierson Moskowitz variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self.peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
        )
        return values


class FreqJonswap(FreqShape):
    """
    JONSWAP frequency shape as proposed by Hasselmann et al. (1973). Commonly used for wind-generated waves in deep
    water.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a JONSWAP frequency shape with the given peak frequency and significant wave height. The peakeness
        parameter $\\gamma$ is set to 3.3 by default, but may be overridden by the gamma keyword argument.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqJonswap, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", _GRAV)
        self.gamma = kwargs.get("gamma", 3.3)

        # Hardcoded JONSWAP parameters until we have a better way of handling alpha
        self._sigma_a = 0.07  # kwargs.get("sigma_a", 0.07)
        self._sigma_b = 0.09  # kwargs.get("sigma_b", 0.09)

    @property
    def _alpha(self):
        # Approximation by Yamaguchi (1984), "Approximate expressions for integral properties of the JONSWAP
        # spectrum" Proc. Japanese Society of Civil Engineers, 345/II-1, 149–152 [in Japanese]. Taken from Holthuijsen
        # "waves in oceanic and coastal waters". Not valid if sigma_a or sigma_b are chanegd from defaults. Otherwise
        # accurate to within 0.25%
        #
        return (
            self.m0
            * (2 * numpy.pi * self.peak_frequency_hertz) ** 4
            / self._g**2
            / (0.06533 * self.gamma**0.8015 + 0.13467)
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Jonswap variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0

        sigma = numpy.where(
            frequency_hertz <= self.peak_frequency_hertz, self._sigma_a, self._sigma_b
        )
        peak_enhancement = self.gamma ** numpy.exp(
            -1 / 2 * ((frequency_hertz / self.peak_frequency_hertz - 1) / sigma) ** 2
        )

        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self.peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
            * peak_enhancement[msk]
        )
        return values


def create_freq_shape(
    period: float,
    waveheight: float,
    shape_name: Literal["jonswap", "pm", "phillips", "gaussian"],
    **kwargs,
) -> FreqShape:
    """
    Create a frequency shape object.
    :param period: Period measure in seconds
    :param waveheight: Wave height
    :param shape_name: frequency shape, one of 'jonswap','pm','phillips','gaussian'
    :return: FreqShape object
    """
    if shape_name == "jonswap":
        return FreqJonswap(
            peak_frequency_hertz=1 / period,
            significant_waveheight_meter=waveheight,
            **kwargs,
        )
    elif shape_name == "pm":
        return FreqPiersonMoskowitz(
            peak_frequency_hertz=1 / period,
            significant_waveheight_meter=waveheight,
            **kwargs,
        )
    elif shape_name == "phillips":
        return FreqPhillips(
            peak_frequency_hertz=1 / period,
            significant_waveheight_meter=waveheight,
            **kwargs,
        )
    elif shape_name == "gaussian":
        return FreqGaussian(
            peak_frequency_hertz=1 / period,
            significant_waveheight_meter=waveheight,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown frequency shape {shape_name}")


def create_dir_shape(
    direction: float, spread: float, shape_name: Literal["cosN", "cos2N"], **kwargs
) -> DirShape:
    """
    Create a directional shape object.
    :param direction: Mean direction
    :param spread: Directional spread
    :param shape_name: Directional shape, one of 'cosN','cos2N'
    :return: DirShape object
    """
    if shape_name == "cosN":
        return DirCosineN(
            mean_direction_degrees=direction, width_degrees=spread, **kwargs
        )
    elif shape_name == "cos2N":
        return DirCosine2N(
            mean_direction_degrees=direction, width_degrees=spread, **kwargs
        )
    else:
        raise ValueError(f"Unknown directional shape {shape_name}.")


class FrequencyDirectionSpectrum:
    def __init__(
        self,
        frequency_shape: "FreqShape",
        direction_shape: "DirShape",
    ):
        self.frequency_shape = frequency_shape
        self.direction_shape = direction_shape

    def values(
        self,
        frequency_hertz: numpy.ndarray,
        direction_degrees: numpy.ndarray,
        renormalize: bool = True,
    ):
        D = self.direction_shape.values(direction_degrees, renormalize=renormalize)
        E = self.frequency_shape.values(frequency_hertz, renormalize=renormalize)
        return E[:, None] * D[None, :]


def parametric_directional_spectrum(
    frequency_hertz: numpy.ndarray,
    direction_degrees: numpy.ndarray,
    frequency_shape: "FreqShape",
    direction_shape: "DirShape",
    renormalize: bool = True,
):
    """
    Create a parametrized directional frequency spectrum according to a given frequency or directional distribution.

    :param frequency_hertz: Frequencies to resolve
    :param peak_frequency_hertz:  Desired peak frequency of the spectrum
    :param frequency_shape:  Frequency shape object, see create_frequency_shape
    :param direction_shape:  Directional shape object, see create_directional_shape

    :return: FrequencyDirectionSpectrum object.
    """

    if frequency_hertz is None:
        fp = frequency_shape.peak_frequency_hertz
        frequency_hertz = numpy.linspace(0.5 * fp, 3 * fp, 26)

    if direction_degrees is None:
        direction_degrees = numpy.linspace(0, 360, 36, endpoint=False)

    D = direction_shape.values(direction_degrees, renormalize=renormalize)
    E = frequency_shape.values(frequency_hertz, renormalize=renormalize)

    return E[:, None] * D[None, :]
