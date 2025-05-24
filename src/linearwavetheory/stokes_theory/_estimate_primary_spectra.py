from ._higher_order_spectra import _point_bound_surface_wave_spectrum_2d
from ..settings import PhysicsOptions, _parse_options, StokesTheoryOptions
from .._array_shape_preprocessing import atleast_1d
import numpy as np
from typing import Union
from numba import prange, jit
from .._numba_settings import numba_default
from linearwavetheory._utils import _direction_bin


@jit(**numba_default)
def estimate_primary_spectrum_from_nonlinear_spectra(
    intrinsic_angular_frequency: np.ndarray,
    angle_degrees: np.ndarray,
    variance_density: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    output="surface_variance",
    nonlinear_options: StokesTheoryOptions = None,
    physics_options: PhysicsOptions = None,
    progress_bar=None,
):

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )

    dims = variance_density.shape

    numerical_option, physics_options, nonlinear_options = _parse_options(
        None, physics_options, nonlinear_options
    )
    nspec = int(np.prod(np.array(dims[:-2])))
    nfreq = len(intrinsic_angular_frequency)
    ndir = len(angle_degrees)
    variance_density = np.reshape(variance_density, (nspec, dims[-2], dims[-1]))
    depth = atleast_1d(depth)

    if len(depth) == 1:
        depth = np.full(nspec, depth[0])

    spectra = np.zeros(
        (nspec, nfreq, ndir),
    )
    spectra[:, :, :] = variance_density[:, :, :]

    delta_angle = _direction_bin(angle_degrees, wrap=360)
    for i in prange(nspec):
        if progress_bar is not None:
            progress_bar.update(1)

        for iter in range(0, 3):
            bound_estimate = _point_bound_surface_wave_spectrum_2d(
                intrinsic_angular_frequency,
                angle_degrees,
                spectra[i, :, :],
                depth[i],
                output,
                nonlinear_options,
                physics_options,
            )
            spectra[i, :, :] = variance_density[i, :, :] - bound_estimate

            # delta = raw_estimate - spectra[i, :, :]
            # spectra[i, :, :] = spectra[i, :, :] + under_relaxation_factor * delta

            for ii in range(nfreq):
                total = 0.0
                target = 0.0
                for jj in range(ndir):
                    target += spectra[i, ii, jj] * delta_angle[jj]
                    if spectra[i, ii, jj] < 0:
                        spectra[i, ii, jj] = 0
                    total += spectra[i, ii, jj] * delta_angle[jj]

                if target < 0.0:
                    target = 0.0

                spectra[i, ii, :] = spectra[i, ii, :] * np.abs((target / total))

    return spectra.reshape(dims)
