"""
Contents: Routines to convert a wave spectrum expressed in encounter frequency (as measured by an observer moving
relative to the water, e.g. a drifting buoy) into the intrinsic frequency frame (as predicted by a model such as
WW3). Built on top of `linearwavetheory.dispersion.inverse_encounter_dispersion_relation`.

Functions:
- `encounter_spectrum_to_intrinsic_1d`, convert a 1D spectrum assuming a single propagation direction for the whole
    spectrum.
- `encounter_spectrum_to_intrinsic_2d`, convert a 2D (frequency, direction) spectrum, using each direction bin's own
    current projection.
"""

import numpy as np

# Relative imports (see the same note in dispersion.py): the package is importable both as
# `linearwavetheory` and as `src.linearwavetheory`, and an absolute import here would bind to
# PhysicsOptions/dispersion functions from whichever of the two module identities happens to load first,
# which can then numba-mismatch against a PhysicsOptions instance a caller constructed via the other path.
from .settings import PhysicsOptions, NumericalOptions
from .dispersion import (
    inverse_encounter_dispersion_relation,
    intrinsic_dispersion_relation,
    intrinsic_group_speed,
)


def encounter_spectrum_to_intrinsic_1d(
    encounter_angular_frequency: np.ndarray,
    variance_density: np.ndarray,
    relative_current_speed: float,
    relative_current_direction_degrees: float,
    mean_direction_degrees: float,
    root: str = "principal",
    depth: float = np.inf,
    physics_options: PhysicsOptions = None,
    numerical_options: NumericalOptions = None,
):
    """
    Convert a 1D encounter-frequency spectrum to the intrinsic frame, assuming every frequency component
    propagates in the same direction `mean_direction_degrees` (e.g. the spectrum's own mean direction). This is
    "Path 1" of the encounter-to-intrinsic conversion: cheap, and adequate whenever the whole spectrum can
    reasonably be treated as propagating in one direction (e.g. a single, locally wind-generated wind sea). See
    `encounter_spectrum_to_intrinsic_2d` for a per-direction-bin conversion that does not require this assumption.

    For each encounter angular frequency $\\omega$ the corresponding wavenumber is found via
    `inverse_encounter_dispersion_relation`, and the density is transformed via the energy-conservation Jacobian

        $$E(\\sigma) = E(\\omega) \\left| \\frac{d\\omega}{d\\sigma} \\right|
                     = E(\\omega) \\left| 1 + \\frac{\\Delta}{C_g(\\sigma)} \\right|$$

    with $\\Delta = V \\cos(\\theta - \\theta_V)$ the current projection onto `mean_direction_degrees`, and $C_g$
    the intrinsic group speed -- valid at any depth.

    NOTE:
        - Frequencies for which the requested root does not exist (see `inverse_encounter_dispersion_relation`)
          return NaN in both outputs; they are not dropped or extrapolated.
        - For `root="secondary"`, the intrinsic direction may actually be `mean_direction_degrees + 180` wherever
          `Delta >= 0` (the "swept away" branch) -- this is not corrected here. `"principal"` is the recommended
          default; `"secondary"` is provided for completeness/testing.

    :param encounter_angular_frequency: Encounter angular frequency (rad/s), non-negative. 1D array.

    :param variance_density: Variance density (m^2 s/rad) at each `encounter_angular_frequency`. Same shape.

    :param relative_current_speed: Magnitude of the relative current velocity (m/s).

    :param relative_current_direction_degrees: Direction (degrees) of the relative current velocity.

    :param mean_direction_degrees: Direction (degrees) assumed for the whole spectrum.

    :param root: Which root to use, `"principal"` (default) or `"secondary"`.

    :param depth: depth in meters.

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values
        are used.

    :param numerical_options: A NumericalOptions object containing the numerical parameters (only used for the
        `intermediate` wave regime). If None, the default values are used.

    :return: A tuple `(intrinsic_frequency, intrinsic_variance_density)`, the same shape as the input, not
        interpolated onto a regular grid -- the caller regrids if needed (see `encounter_spectrum_to_intrinsic_2d`
        for the per-direction-bin regridding approach, which applies identically here with a single bin).
    """
    wavenumber = inverse_encounter_dispersion_relation(
        encounter_angular_frequency,
        mean_direction_degrees,
        relative_current_speed,
        relative_current_direction_degrees,
        root,
        depth,
        physics_options,
        numerical_options,
    )

    intrinsic_frequency = intrinsic_dispersion_relation(
        wavenumber, depth, physics_options
    )
    group_speed = intrinsic_group_speed(wavenumber, depth, physics_options)

    delta = relative_current_speed * np.cos(
        np.deg2rad(mean_direction_degrees - relative_current_direction_degrees)
    )
    domega_dsigma = 1.0 + delta / group_speed

    intrinsic_variance_density = variance_density * np.abs(domega_dsigma)

    return intrinsic_frequency, intrinsic_variance_density


def _regrid_1d_spectrum(
    sigma: np.ndarray, density: np.ndarray, target_frequency: np.ndarray
) -> np.ndarray:
    """
    Interpolate a (possibly unsorted, possibly NaN-containing) `(sigma, density)` pair onto `target_frequency`.
    NaN entries in `sigma` are dropped before interpolating; `target_frequency` points outside the remaining valid
    range of `sigma` are set to NaN rather than extrapolated (numpy's `interp` otherwise clamps to the boundary
    value, which would silently misrepresent out-of-range points).

    ** Internal function. **
    """
    valid = np.isfinite(sigma)
    sigma = sigma[valid]
    density = density[valid]

    if sigma.size == 0:
        return np.full(target_frequency.shape, np.nan)

    order = np.argsort(sigma)
    sigma = sigma[order]
    density = density[order]

    regridded = np.interp(target_frequency, sigma, density, left=np.nan, right=np.nan)
    return regridded


def encounter_spectrum_to_intrinsic_2d(
    encounter_angular_frequency: np.ndarray,
    direction_degrees: np.ndarray,
    variance_density: np.ndarray,
    relative_current_speed: float,
    relative_current_direction_degrees: float,
    root: str = "principal",
    depth: float = np.inf,
    physics_options: PhysicsOptions = None,
    numerical_options: NumericalOptions = None,
    target_frequency: np.ndarray = None,
) -> np.ndarray:
    """
    Convert a 2D `(frequency, direction)` encounter-frequency spectrum to the intrinsic frame, using each
    direction bin's own current projection rather than assuming a single direction for the whole spectrum (compare
    `encounter_spectrum_to_intrinsic_1d`, "Path 1"). `variance_density` is typically obtained from a directional
    spectrum reconstruction (e.g. MEM/MEM2 from directional moments) performed elsewhere -- this function does not
    perform that reconstruction itself.

    The wavenumber vector direction is frame-invariant under a Galilean (moving-observer) transform for the
    principal root, so the 2D problem decomposes into independent 1D problems, one per direction bin: this
    function calls `encounter_spectrum_to_intrinsic_1d` once per direction bin, then regrids each bin's resulting
    (irregular) intrinsic frequency axis onto a common `target_frequency` grid via 1D interpolation.

    NOTE:
        - Frequencies for which the requested root does not exist, or that fall outside a given direction bin's
          valid range after regridding, are NaN in the output, not dropped or extrapolated.
        - See `encounter_spectrum_to_intrinsic_1d` for the `root="secondary"` direction caveat.

    :param encounter_angular_frequency: Encounter angular frequency (rad/s), non-negative. 1D array of length
        `n_frequency`.

    :param direction_degrees: Direction (degrees) of each column of `variance_density`. 1D array of length
        `n_direction`.

    :param variance_density: Variance density (m^2 s/rad/deg), shape `(n_frequency, n_direction)`.

    :param relative_current_speed: Magnitude of the relative current velocity (m/s).

    :param relative_current_direction_degrees: Direction (degrees) of the relative current velocity.

    :param root: Which root to use, `"principal"` (default) or `"secondary"`.

    :param depth: depth in meters.

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values
        are used.

    :param numerical_options: A NumericalOptions object containing the numerical parameters (only used for the
        `intermediate` wave regime). If None, the default values are used.

    :param target_frequency: Frequency grid (rad/s) to regrid onto. Defaults to `encounter_angular_frequency`.

    :return: The intrinsic variance density, shape `(len(target_frequency), n_direction)`.
    """
    if target_frequency is None:
        target_frequency = encounter_angular_frequency

    intrinsic_variance_density = np.full(
        (len(target_frequency), len(direction_degrees)), np.nan
    )
    for j, direction in enumerate(direction_degrees):
        sigma, density = encounter_spectrum_to_intrinsic_1d(
            encounter_angular_frequency,
            variance_density[:, j],
            relative_current_speed,
            relative_current_direction_degrees,
            direction,
            root,
            depth,
            physics_options,
            numerical_options,
        )
        intrinsic_variance_density[:, j] = _regrid_1d_spectrum(
            sigma, density, target_frequency
        )

    return intrinsic_variance_density
