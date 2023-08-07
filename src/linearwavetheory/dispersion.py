"""
Contents: Routines to calculate (inverse) linear dispersion relation and some related quantities such as phase and
group velocity. All results are based on the dispersion relation in 2D for linear gravity waves for an observer moving
at velocity Uo, with an ambient current Uc and including suface tension effects:

        w = sqrt( ( ( g * k + tau * k**3 ) * tanh(k*d) ) + K dot (Uc - Uo)

with

    w   :: the encounter frequency,
    g   :: gravitational acceleration
    tau :: kinematice surface tension
    k   :: wavenumber magnitude
    d   :: depth
    K   :: (kx,ky) intrinsic wavenumber vector.
    Uc  :: (Ux,Uy) current velocity vector in the earth reference frame
    Uo  :: (Ux,Uy) observer velocity vector in the earth reference frame

where the wavenumber vector is defined such that in the intrinsic reference frame (moving with the waves) the wavenumber
is directed into the direction of wave propagation. We refer to this as the intrinsic wavenumber.

Functions:
- `intrinsic_dispersion_relation`, calculate angular frequency for a given wavenumber magnitude and depth
- `inverse_intrinsic_dispersion_relation`, calculate wavenumber magnitude for a given angular frequency and depth
- `encounter_dispersion_relation`, calculate angular frequency for a given wavenumber vector, depth, observer and
        current velocity
- `intrinsic_group_speed`, calculate the group speed given wave number magnitude and depth.
- `intrinsic_phase_speed`, calculate the phase speed given wave number magnitude  and depth.
- `encounter_group_velocity`, calculate the group velocity given wave number vector, depth, observer and current
        velocity.
- `encounter_phase_velocity`, calculate the phase velocity given wave number vector, depth, observer and current
        velocity.
- `encounter_phase_speed`, calculate the phase speed given wave number vector, depth, observer and current velocity.
- `encounter_group_speed`, calculate the group speed given wave number vector, depth, observer and current velocity.

Some notes:
---------
- Evanesent waves are not supported.

- I have used encounter frequency/phase velocity etc over the more commonly used absolute frequency/phase velocity.
  a) This allows for inclusion of moving observers with regard to the earth reference frame,
  b) the calling the earths reference frame an absolute reference frame is technically incorrect (though common),
  c) the "absolute"  frequency may become negative for strong counter currents when using intrinsic wavenumbers,
  which is confusing.

- The implementation uses numba to speed up calculations. This allows for straightforward use of looping which is often
  more consise as an implementation while retaining a python-like implementation. The downsides are (among others) that:
  a) the first call to a function will be slow.
  b) error messages are not always very informative due to LLVM compilation.
  c) not all (numpy) functions are available in numba, leading to ugly workarounds. (e.g. np.atleast_1d as used here)

Copyright (C) 2023
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

"""

import numpy as np
from numba import jit, vectorize
from ._tools import atleast_1d, _to_2d_array
from ._constants import (
    GRAV,
    KINEMATIC_SURFACE_TENSION,
    MAXIMUM_NUMBER_OF_ITERATIONS,
    RELATIVE_TOLERANCE,
    ABSOLUTE_TOLERANCE,
)

from ._numba_settings import numba_default
from typing import Union


@jit(**numba_default)
def inverse_intrinsic_dispersion_relation(
    intrinsic_angular_frequency: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    grav: float = GRAV,
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    maximum_number_of_iterations: int = MAXIMUM_NUMBER_OF_ITERATIONS,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    limit="none",
) -> np.ndarray:
    """
    Find the wavenumber magnitude for a given intrinsic radial frequency through inversion of the dispersion relation
    for linear gravity waves including suface tension effects, i.e. solve for wavenumber k in:

        w = sqrt( ( ( g * k + tau * k**3 ) * tanh(k*d) )

    with g gravitational acceleration, tau kinematice surface tension, k wavenumber and d depth. The dispersion relation
    is solved using Newton Iteration with a first guess based on the dispersion relation for deep or shallow water
    waves.

    Notes:
    - We only solve for real roots (no evanescent waves)
    - For negative depths or zero depths we reduce nan (undefined)
    - For zero frequency we return zero wavenumber
    - Stopping criterium is based on relative and absolute tolerance:

            | w - w_est | / w < relative_tolerance  (default 1e-3)

            and

            | w - w_est |  < absolute_tolerance  (default np.inf

        where w is the intrinsic angular frequency and w_est is the estimated intrinsic angular frequency based on the
        current estimate of the wavenumber. Per default we do not use the absolute stopping criterium.

    - We exit when either maximum number of iterations (default 10 is reached, or tolerance is achieved.
      Typically only 1 to 2 iterations are needed.

    :param  intrinsic_angular_frequency: The radial frequency as observed from a frame of reference moving with the
        wave. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the intrinsic_angular_frequency array, and calculation is performed pairwise. I.e.
        k[j] is calculated for w[j] and d[j].

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :param maximum_number_of_iterations: Maximum number of iterations. Default is 10.

    :param relative_tolerance: Relative accuracy used in the stopping criterium. Default is 1e-3.

    :param absolute_tolerance: Absolute accuracy used in the stopping criterium. Default is np.inf.

    :param limit: Use a limiting form. Can be: 'deep', 'shallow', 'cappilary', 'gravity' or 'none'.
        Default is 'none' (use the full relation). `gravity` is the same as calling the function with the
        kinematic_surface_tension set to zero.

    :return: The wavenumber as a 1D numpy array. Note that even if a scalar is provided for the intrinsic angular
        frequency, a 1D array is returned.

    Example
    ```python
    >>> from linearwavetheory import inverse_intrinsic_dispersion_relation
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> f = np.linspace(0., 20, 1001)
    >>> w = 2 * np.pi * f
    >>> depth = 100

    >>> k1 = inverse_intrinsic_dispersion_relation(w, depth)
    >>> k2 = inverse_intrinsic_dispersion_relation(w, depth,kinematic_surface_tension=0.0)
    >>> k3 = inverse_intrinsic_dispersion_relation(w, depth,limit='deep')
    >>> k4 = inverse_intrinsic_dispersion_relation(w, depth,limit='shallow')
    >>> k5 = inverse_intrinsic_dispersion_relation(w, depth,limit='capillary')
    >>> plt.plot(f, k1, label='with surface tension')
    >>> plt.plot(f, k2, label='without surface tension')
    >>> plt.plot(f, k3, label='deep water limit')
    >>> plt.plot(f, k4, label='shallow water limit')
    >>> plt.plot(f, k5, label='capillary limit')
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('wavenumber [rad/m]')
    >>> plt.grid('on', which='both')
    >>> plt.xscale('log')
    >>> plt.yscale('log')
    >>> plt.legend()
    >>> plt.show()
    ```
    """

    # Numba does not recognize "atleast_1d" for scalars
    intrinsic_angular_frequency = atleast_1d(intrinsic_angular_frequency)
    depth = atleast_1d(depth)

    if kinematic_surface_tension < 0:
        raise ValueError("kinematic_surface_tension must be non-negative.")

    if grav <= 0:
        raise ValueError("grav must be positive.")

    if maximum_number_of_iterations < 1:
        raise ValueError("maximum_number_of_iterations must be at least 1.")

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive.")

    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive.")

    _limit = 0
    # Passing as a number because numba vectorize does not support strings (it seems? - not well tested other than
    # the naive approach of just passing a string fails.)
    if limit == "deep":
        _limit = 1
    elif limit == "shallow":
        _limit = 2
    elif limit == "capillary":
        _limit = 3
    elif limit == "gravity":
        _limit = 4

    return _inverse_intrinsic_dispersion_relation_ufunc(
        intrinsic_angular_frequency,
        depth,
        kinematic_surface_tension,
        grav,
        maximum_number_of_iterations,
        relative_tolerance,
        absolute_tolerance,
        _limit,
    )


@vectorize()
def _inverse_intrinsic_dispersion_relation_ufunc(
    intrinsic_angular_frequency,
    depth,
    kinematic_surface_tension: float,
    grav: float,
    maximum_number_of_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    _limit: int,
):
    limit = "none"
    """
    ** Internal function. Call inverse_intrinsic_dispersion_relation instead. **

    Find the wavenumber magnitude for a given intrinsic radial frequency through inversion of the dispersion relation
    for linear gravity waves including suface tension effects, i.e. solve for wavenumber k in:

        w = sqrt( ( ( g * k + tau * k**3 ) * tanh(k*d) )

    with g gravitational acceleration, tau kinematice surface tension, k wavenumber and d depth. The dispersion relation
    is solved using Newton Iteration with a first guess based on the dispersion relation for deep or shallow water
    waves.

    Notes:
    - We only solve for real roots (no evanescent waves)
    - For negative depths or zero depths we reduce nan (undefined)
    - For zero frequency we return zero wavenumber
    - Stopping criterium is based on relative and absolute tolerance:

            | w - w_est | / w < relative_tolerance  (default 1e-3)

            and

            | w - w_est |  < absolute_tolerance  (default np.inf

        where w is the intrinsic angular frequency and w_est is the estimated intrinsic angular frequency based on the
        current estimate of the wavenumber. Per default we do not use the absolute stopping criterium.

    - We exit when either maximum number of iterations (default 10 is reached, or tolerance is achieved.
      Typically only 1 to 2 iterations are needed.

    :param  intrinsic_angular_frequency: The radial frequency in rad/s as observed from a frame of reference moving with the
        wave. Scalar.

    :param depth: depth in meters. Scalar.

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :param maximum_number_of_iterations: Maximum number of iterations. Default is 10.

    :param relative_tolerance: Relative accuracy used in the stopping criterium. Default is 1e-3.

    :param absolute_tolerance: Absolute accuracy used in the stopping criterium. Default is np.inf.

    :return: The wavenumber. Scalar.
    """

    # == Input Validation ==

    # Since we can try to solve for the negative intrinsic frequency branch of the dispersion relation, we allow
    # negative intrinsic frequencies. In this case the wavenumber magnitude is the same as for the positive branch.
    intrinsic_angular_frequency = np.abs(intrinsic_angular_frequency)

    # For zero or negative depths the solution is undefined.
    if depth <= 0:
        return np.nan

    # For zero intrinsic frequency the wavenumber is zero
    if intrinsic_angular_frequency == 0:
        return 0.0

    if limit == 0:
        return intrinsic_angular_frequency**2 / grav
    elif limit == 1:
        return intrinsic_angular_frequency / np.sqrt(grav * depth)
    elif limit == 2:
        return (intrinsic_angular_frequency**2 / kinematic_surface_tension) ** (1 / 3)
    elif limit == 3:
        kinematic_surface_tension = 0.0

    # == Initial Estimate ==
    if intrinsic_angular_frequency > np.sqrt(grav / depth):
        # use the deep water relation
        wavenumber_estimate = intrinsic_angular_frequency**2 / grav
    else:
        # use the shallow water relation
        wavenumber_estimate = intrinsic_angular_frequency / np.sqrt(grav * depth)

    # If the initial estimate turns out to be in the capillary range- use the pure capillary dispersion relation for
    # the initial estimate.
    if wavenumber_estimate**3 * kinematic_surface_tension / grav > 5:
        wavenumber_estimate = (
            intrinsic_angular_frequency**2 / kinematic_surface_tension
        ) ** (1 / 3)

    # == Newton Iteration ==
    for ii in range(0, maximum_number_of_iterations):
        surface_tension_term = np.sqrt(
            1 + kinematic_surface_tension * wavenumber_estimate**2 / grav
        )

        error = (
            np.sqrt(grav * wavenumber_estimate * np.tanh(wavenumber_estimate * depth))
            * surface_tension_term
            - intrinsic_angular_frequency
        )

        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / intrinsic_angular_frequency) < relative_tolerance
        ):
            break

        kd = wavenumber_estimate * depth

        # Group speed in the absence of surface tension
        cg = (1 / 2 + kd / np.sinh(2 * kd)) * np.sqrt(
            grav / wavenumber_estimate * np.tanh(kd)
        )

        # Calculate the derivative of the error function with respect to wavenumber.
        error_derivative_to_wavenumber = (
            cg * surface_tension_term
            + np.sqrt(grav * wavenumber_estimate * np.tanh(kd))
            * wavenumber_estimate
            * kinematic_surface_tension
            / grav
            / surface_tension_term
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        )

    return wavenumber_estimate


@jit(**numba_default)
def _inverse_intrinsic_dispersion_relation_scalar(
    intrinsic_angular_frequency: float,
    depth: float = np.inf,
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav: float = GRAV,
    maximum_number_of_iterations: int = MAXIMUM_NUMBER_OF_ITERATIONS,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    limit="none",
) -> float:
    """
    ** Internal function. Call inverse_intrinsic_dispersion_relation instead. **

    Find the wavenumber magnitude for a given intrinsic radial frequency through inversion of the dispersion relation
    for linear gravity waves including suface tension effects, i.e. solve for wavenumber k in:

        w = sqrt( ( ( g * k + tau * k**3 ) * tanh(k*d) )

    with g gravitational acceleration, tau kinematice surface tension, k wavenumber and d depth. The dispersion relation
    is solved using Newton Iteration with a first guess based on the dispersion relation for deep or shallow water
    waves.

    Notes:
    - We only solve for real roots (no evanescent waves)
    - For negative depths or zero depths we reduce nan (undefined)
    - For zero frequency we return zero wavenumber
    - Stopping criterium is based on relative and absolute tolerance:

            | w - w_est | / w < relative_tolerance  (default 1e-3)

            and

            | w - w_est |  < absolute_tolerance  (default np.inf

        where w is the intrinsic angular frequency and w_est is the estimated intrinsic angular frequency based on the
        current estimate of the wavenumber. Per default we do not use the absolute stopping criterium.

    - We exit when either maximum number of iterations (default 10 is reached, or tolerance is achieved.
      Typically only 1 to 2 iterations are needed.

    :param  intrinsic_angular_frequency: The radial frequency in rad/s as observed from a frame of reference moving with the
        wave. Scalar.

    :param depth: depth in meters. Scalar.

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :param maximum_number_of_iterations: Maximum number of iterations. Default is 10.

    :param relative_tolerance: Relative accuracy used in the stopping criterium. Default is 1e-3.

    :param absolute_tolerance: Absolute accuracy used in the stopping criterium. Default is np.inf.

    :return: The wavenumber. Scalar.
    """

    # == Input Validation ==

    # Since we can try to solve for the negative intrinsic frequency branch of the dispersion relation, we allow
    # negative intrinsic frequencies. In this case the wavenumber magnitude is the same as for the positive branch.
    intrinsic_angular_frequency = np.abs(intrinsic_angular_frequency)

    # For zero or negative depths the solution is undefined.
    if depth <= 0:
        return np.nan

    # For zero intrinsic frequency the wavenumber is zero
    if intrinsic_angular_frequency == 0:
        return 0.0

    if kinematic_surface_tension < 0:
        raise ValueError("kinematic_surface_tension must be non-negative.")

    if grav <= 0:
        raise ValueError("grav must be positive.")

    if maximum_number_of_iterations < 1:
        raise ValueError("maximum_number_of_iterations must be at least 1.")

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive.")

    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive.")

    if limit == "deep":
        return intrinsic_angular_frequency**2 / grav
    elif limit == "shallow":
        return intrinsic_angular_frequency / np.sqrt(grav * depth)
    elif limit == "capillary":
        return (intrinsic_angular_frequency**2 / kinematic_surface_tension) ** (1 / 3)
    elif limit == "gravity":
        kinematic_surface_tension = 0.0

    # == Initial Estimate ==
    if intrinsic_angular_frequency > np.sqrt(grav / depth):
        # use the deep water relation
        wavenumber_estimate = intrinsic_angular_frequency**2 / grav
    else:
        # use the shallow water relation
        wavenumber_estimate = intrinsic_angular_frequency / np.sqrt(grav * depth)

    # If the initial estimate turns out to be in the capillary range- use the pure capillary dispersion relation for
    # the initial estimate.
    if wavenumber_estimate**3 * kinematic_surface_tension / grav > 5:
        wavenumber_estimate = (
            intrinsic_angular_frequency**2 / kinematic_surface_tension
        ) ** (1 / 3)

    # == Newton Iteration ==
    for ii in range(0, maximum_number_of_iterations):
        surface_tension_term = np.sqrt(
            1 + kinematic_surface_tension * wavenumber_estimate**2 / grav
        )

        error = (
            np.sqrt(grav * wavenumber_estimate * np.tanh(wavenumber_estimate * depth))
            * surface_tension_term
            - intrinsic_angular_frequency
        )

        if (
            np.abs(error) < absolute_tolerance
            and np.abs(error / intrinsic_angular_frequency) < relative_tolerance
        ):
            break

        kd = wavenumber_estimate * depth

        # Group speed in the absence of surface tension
        cg = (1 / 2 + kd / np.sinh(2 * kd)) * np.sqrt(
            grav / wavenumber_estimate * np.tanh(kd)
        )

        # Calculate the derivative of the error function with respect to wavenumber.
        error_derivative_to_wavenumber = (
            cg * surface_tension_term
            + np.sqrt(grav * wavenumber_estimate * np.tanh(kd))
            * wavenumber_estimate
            * kinematic_surface_tension
            / grav
            / surface_tension_term
        )

        # Newton Iteration
        wavenumber_estimate = (
            wavenumber_estimate - error / error_derivative_to_wavenumber
        )
    else:
        print(
            "inverse_intrinsic_dispersion_relation:: No convergence in solving for wavenumber"
        )

    return wavenumber_estimate


@jit(**numba_default)
def intrinsic_dispersion_relation(
    wavenumber_magnitude: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The intrinsic dispersion relation for linear gravity-capillary waves in water of constant depth that relates the
    specific angular frequency to a given wavenumber and depth in a reference frame following mean ambient flow. I.e.

        w = sqrt( ( ( g * k + tau * k**3 ) * tanh(k*d) )

    with g gravitational acceleration, tau kinematic surface tension, k wavenumber and d depth.

    NOTE:
        - if the wavenumber magnitude is zero, the intrinsic frequency is zero.
        - if the depth is smaller or equal to zero, the intrinsic frequency is undefined and np.nan is returned.
        - if the wavenumber magnitude is negative, we return an error (undefined).

    :param  wavenumber_magnitude: The wavenumber_magnitude. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the wavenumber_magnitude array, and calculation is performed pairwise. I.e.
        w[j] is calculated for k[j] and d[j].

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :return: The intrinsic angular frequency as a 1 dimensional numpy array.
    """

    if kinematic_surface_tension < 0:
        raise ValueError("kinematic_surface_tension must be non-negative.")

    if grav < 0:
        raise ValueError("grav must be positive.")

    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    if np.any(wavenumber_magnitude < 0.0):
        raise ValueError("wavenumber_magnitude must be non-negative.")

    return _intrinsic_dispersion_relation_ufunc(
        wavenumber_magnitude, depth, kinematic_surface_tension, grav
    )


@vectorize()
def _intrinsic_dispersion_relation_ufunc(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    if wavenumber_magnitude == 0.0:
        return 0.0

    if depth <= 0:
        return np.nan

    return np.sqrt(
        (
            grav * wavenumber_magnitude
            + kinematic_surface_tension * wavenumber_magnitude**3
        )
        * np.tanh(wavenumber_magnitude * depth)
    )


@jit(**numba_default)
def encounter_dispersion_relation(
    intrinsic_wavenumber_vector: np.ndarray,
    depth=np.inf,
    ambient_current_velocity=(0, 0),
    observer_velocity=(0, 0),
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The dispersion relation for linear waves in water of constant depth with a constant ambient current that relates the
    specific angular frequency to a given wavenumber.

    We work under we work under the convention that intrinsic frequency is positive. The wavenumber is then
    directed in the direction of the phase velocity of the wave in the current following reference frame.
    In this case the dispersion relation always has a single unique solution, and is given by:

            encounter_frequency = intrinsic_frequency + doppler_shift

    However, the encounter frequency may now be positive or negative. A negative frequency indicates that the phase
    velocity of the wave travels in the opposite direction of the wavenumber.


    Input
    -----
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as a 1D
    numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param kinematic_surface_tension: Kinematic surface tension parameter (m^3/s^2). Per default set to 0.0728 N/m
    (water at 20C) divided by density of seawater (1025 kg/m^3).

    :param grav: Gravitational acceleration (m/s^2). Default 9.81 m/s^2.

    :return: Encounter angular frequency (rad/s) as a 1D numpy array.
    """

    intrinsic_wavenumber_vector = _to_2d_array(intrinsic_wavenumber_vector)
    ambient_current_velocity = _to_2d_array(
        ambient_current_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    )
    observer_velocity = _to_2d_array(
        observer_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    )
    relative_velocity = ambient_current_velocity - observer_velocity

    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )
    doppler_shift = np.sum(intrinsic_wavenumber_vector * relative_velocity, axis=-1)

    intrinsic_angular_frequency = intrinsic_dispersion_relation(
        wavenumber_magnitude,
        depth,
        kinematic_surface_tension=kinematic_surface_tension,
        grav=grav,
    )

    return intrinsic_angular_frequency + doppler_shift


@jit(**numba_default)
def intrinsic_phase_speed(
    wavenumber_magnitude,
    depth=np.inf,
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:

    """
    The intrinsic phase speed for linear gravity-capillary waves. I.e.

        c = w / k = sqrt( ( ( g / k + tau * k ) * tanh(k*d) )

    with g gravitational acceleration, tau kinematic surface tension, k wavenumber magitude and d depth.

    NOTE:
        - if the wavenumber magnitude is zero, the phase speed is defined as it's limiting value as k-> 0
            (the shallow water phase speed).
        - if the depth is smaller or equal to zero, the phase speed is undefined and np.nan is returned.
        - if the wavenumber magnitude is negative, we return an error (undefined).

    :param  wavenumber_magnitude: The wavenumber_magnitude. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the wavenumber_magnitude array, and calculation is performed pairwise. I.e.
        w[j] is calculated for k[j] and d[j].

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :return: The intrinsic phase speed as a 1 dimensional numpy array.
    """
    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    depth = atleast_1d(depth)

    if grav <= 0:
        raise ValueError("Gravitational acceleration must be positive")

    if kinematic_surface_tension <= 0:
        raise ValueError("Kinematic surface tension must be positive")

    if np.any(wavenumber_magnitude < 0):
        raise ValueError("Wavenumber magnitude must be positive")

    return _intrinsic_phase_speed_ufunc(
        wavenumber_magnitude, depth, kinematic_surface_tension, grav
    )


@vectorize()
def _intrinsic_phase_speed_ufunc(
    wavenumber_magnitude, depth, kinematic_surface_tension, grav
):
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    return np.sqrt(
        (grav / wavenumber_magnitude + kinematic_surface_tension * wavenumber_magnitude)
        * np.tanh(wavenumber_magnitude * depth)
    )


@jit(**numba_default)
def encounter_phase_velocity(
    intrinsic_wavenumber_vector: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    ambient_current_velocity=(0.0, 0.0),
    observer_velocity=(0.0, 0.0),
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The phase VELOCITY of a wave moving through an ambient current field with velocity ambient_current_velocity
    (defauly 0) as observed by an observer moving with velocity observer_velocity (default 0) relative to the
    earth reference frame.

    We calculate the result as a function of the intrinsic wavenumber, with magnitude and direction such that the
    wavenumber points in the direction of wave propagation in the intrinsic frame of reference. Depending on the
    magnitude of the ambient current and the wavenumber, this wavenumber may differ in sign -
    though magnitude is preserved for all observers.

    Input
    -----
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as a 1D
    numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param kinematic_surface_tension: Kinematic surface tension parameter. Per default set to 0.0728 N/m (water at 20C)
    divided by density of seawater (1025 kg/m^3).

    :param grav: Gravitational acceleration (m/s^2). Default 9.81 m/s^2.

    :return: A 2D numpy array with shape (N,2) where N is the number of wavenumbers.
    """

    intrinsic_wavenumber_vector = _to_2d_array(intrinsic_wavenumber_vector)
    relative_velocity = _to_2d_array(
        ambient_current_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    ) - _to_2d_array(
        observer_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    )
    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )
    _intrinsic_phase_speed = intrinsic_phase_speed(
        wavenumber_magnitude,
        depth,
        kinematic_surface_tension=kinematic_surface_tension,
        grav=grav,
    )

    _encounter_phase_velocity = np.zeros(intrinsic_wavenumber_vector.shape)
    for jj in range(0, len(wavenumber_magnitude)):
        if wavenumber_magnitude[jj] > 0:
            _encounter_phase_velocity[jj, 0] = (
                _intrinsic_phase_speed[jj]
                * intrinsic_wavenumber_vector[jj, 0]
                / wavenumber_magnitude[jj]
                + relative_velocity[jj, 0]
            )

            _encounter_phase_velocity[jj, 1] = (
                _intrinsic_phase_speed[jj]
                * intrinsic_wavenumber_vector[jj, 1]
                / wavenumber_magnitude[jj]
                + relative_velocity[jj, 1]
            )

    return _encounter_phase_velocity


@jit(**numba_default)
def encounter_phase_speed(
    intrinsic_wavenumber_vector,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    observer_velocity=(0.0, 0.0),
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The phase SPEED of a wave moving through an ambient current field with velocity ambient_current_velocity
    (defauly 0) as observed by an observer moving with velocity observer_velocity (default 0) relative to the
    earth reference frame.

    We calculate the result as a function of the intrinsic wavenumber, with magnitude and direction such that the
    wavenumber points in the direction of wave propagation in the intrinsic frame of reference. Depending on the
    magnitude of the ambient current and the wavenumber, this wavenumber may differ in sign -
    though magnitude is preserved for all observers.

    Input
    -----
    :param intrinsic_wavenumber: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as a 1D
    numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param kinematic_surface_tension: Kinematic surface tension parameter. Per default set to 0.0728 N/m (water at 20C)
    divided by density of seawater (1025 kg/m^3).

    :param grav: Gravitational acceleration (m/s^2). Default 9.81 m/s^2.

    :return: A 1D numpy array with shape (N,) where N is the number of wavenumbers.
    """
    _encounter_phase_velocity = encounter_phase_velocity(
        intrinsic_wavenumber_vector,
        depth,
        ambient_current_velocity=ambient_current_velocity,
        observer_velocity=observer_velocity,
        kinematic_surface_tension=kinematic_surface_tension,
        grav=grav,
    )
    return np.sqrt(
        np.sum(_encounter_phase_velocity * _encounter_phase_velocity, axis=-1)
    )


@jit(**numba_default)
def intrinsic_group_speed(
    wavenumber_magnitude: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    NOTE:
        - if the wavenumber magnitude is zero, the group speed is defined as it's limiting value as k-> 0
            (the shallow water group speed).
        - if the depth is smaller or equal to zero, the group speed is undefined and np.nan is returned.
        - if the wavenumber magnitude is negative, we return an error (undefined).

    :param  wavenumber_magnitude: The wavenumber_magnitude. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the wavenumber_magnitude array, and calculation is performed pairwise. I.e.
        w[j] is calculated for k[j] and d[j].

    :param kinematic_surface_tension: kinematic surface tension in m^3/s^2. Per default set to 0.0728 N/m (water at 20C)
        divided by density of seawater (1025 kg/m^3).

    :param grav: gravitational acceleration in m/s^2. Default is 9.81 m/s^2.

    :return: The intrinsic group speed as a 1 dimensional numpy array.
    """
    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    depth = atleast_1d(depth)

    if grav <= 0:
        raise ValueError("Gravitational acceleration must be positive")

    if kinematic_surface_tension <= 0:
        raise ValueError("Kinematic surface tension must be positive")

    if np.any(wavenumber_magnitude < 0):
        raise ValueError("Wavenumber magnitude must be positive")

    return _intrinsic_group_speed_ufunc(
        wavenumber_magnitude, depth, kinematic_surface_tension, grav
    )


@vectorize()
def _intrinsic_group_speed_ufunc(
    wavenumber_magnitude,
    depth,
    kinematic_surface_tension,
    grav,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    """
    if depth <= 0:
        return np.nan

    if wavenumber_magnitude == 0:
        return np.sqrt(grav * depth)

    surface_tension_term = np.sqrt(
        1 + kinematic_surface_tension * wavenumber_magnitude**2 / grav
    )
    kd = wavenumber_magnitude * depth
    n = 1 / 2 + kd / np.sinh(2 * kd)
    c = np.sqrt(grav / wavenumber_magnitude * np.tanh(kd))
    w = wavenumber_magnitude * c

    _intrinsic_group_speed = (
        n * c * surface_tension_term
        + w
        * wavenumber_magnitude
        * kinematic_surface_tension
        / grav
        / surface_tension_term
    )
    return _intrinsic_group_speed


@jit(**numba_default)
def encounter_group_velocity(
    intrinsic_wavenumber_vector,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    observer_velocity=(0.0, 0.0),
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The group VELOCITY of a wave moving through an ambient current field with velocity ambient_current_velocity
    (defauly 0) as observed by an observer moving with velocity observer_velocity (default 0) relative to the
    earth reference frame.

    We calculate the result as a function of the intrinsic wavenumber, with magnitude and direction such that the
    wavenumber points in the direction of wave propagation in the intrinsic frame of reference. Depending on the
    magnitude of the ambient current and the wavenumber, this wavenumber may differ in sign -
    though magnitude is preserved for all observers.

    Input
    -----
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as a 1D
    numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param kinematic_surface_tension: Kinematic surface tension parameter. Per default set to 0.0728 N/m (water at 20C)
    divided by density of seawater (1025 kg/m^3).

    :param grav: Gravitational acceleration (m/s^2). Default 9.81 m/s^2.

    :return: A 2D numpy array with shape (N,2) where N is the number of wavenumbers.
    """

    intrinsic_wavenumber_vector = _to_2d_array(intrinsic_wavenumber_vector)
    relative_velocity = _to_2d_array(
        ambient_current_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    ) - _to_2d_array(
        observer_velocity, target_rows=intrinsic_wavenumber_vector.shape[0]
    )
    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )
    _intrinsic_group_speed = intrinsic_group_speed(
        wavenumber_magnitude,
        depth,
        kinematic_surface_tension=kinematic_surface_tension,
        grav=grav,
    )

    _encounter_group_velocity = np.zeros(intrinsic_wavenumber_vector.shape)
    for jj in range(0, len(wavenumber_magnitude)):
        if wavenumber_magnitude[jj] > 0:
            _encounter_group_velocity[jj, 0] = (
                _intrinsic_group_speed[jj]
                * intrinsic_wavenumber_vector[jj, 0]
                / wavenumber_magnitude[jj]
                + relative_velocity[jj, 0]
            )

            _encounter_group_velocity[jj, 1] = (
                _intrinsic_group_speed[jj]
                * intrinsic_wavenumber_vector[jj, 1]
                / wavenumber_magnitude[jj]
                + relative_velocity[jj, 1]
            )
        else:
            # group velocity is undefined for zero wavenumber.
            _encounter_group_velocity[jj, 0] = np.nan
            _encounter_group_velocity[jj, 1] = np.nan
    return _encounter_group_velocity


@jit(**numba_default)
def encounter_group_speed(
    wavenumber,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    relative_velocity=(0.0, 0.0),
    kinematic_surface_tension: float = KINEMATIC_SURFACE_TENSION,
    grav=GRAV,
) -> np.ndarray:
    """
    The group SPEED of a wave moving through an ambient current field with velocity ambient_current_velocity
    (defauly 0) as observed by an observer moving with velocity observer_velocity (default 0) relative to the
    earth reference frame.

    We calculate the result as a function of the intrinsic wavenumber, with magnitude and direction such that the
    wavenumber points in the direction of wave propagation in the intrinsic frame of reference. Depending on the
    magnitude of the ambient current and the wavenumber, this wavenumber may differ in sign -
    though magnitude is preserved for all observers.

    Input
    -----
    :param intrinsic_wavenumber: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as a 1D
    numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param kinematic_surface_tension: Kinematic surface tension parameter. Per default set to 0.0728 N/m (water at 20C)
    divided by density of seawater (1025 kg/m^3).

    :param grav: Gravitational acceleration (m/s^2). Default 9.81 m/s^2.

    :return: A 1D numpy array with shape (N,) where N is the number of wavenumbers.
    """
    _encounter_group_velocity = encounter_group_velocity(
        wavenumber,
        depth,
        ambient_current_velocity,
        relative_velocity,
        kinematic_surface_tension=kinematic_surface_tension,
        grav=grav,
    )
    return np.sqrt(
        np.sum(_encounter_group_velocity * _encounter_group_velocity, axis=-1)
    )
