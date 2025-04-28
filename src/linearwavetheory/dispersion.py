"""
Contents: Routines to calculate (inverse) linear dispersion relation and some related quantities such as phase and
group velocity. All results are based on the dispersion relation in 2D for linear gravity waves for an observer moving
at velocity $U_o$, with an ambient current $U_c$ and including suface tension effects

$$\\omega = \\sqrt{  ( g k + \\tau k^3 ) \\tanh(kd) } + \\vec{k} \\cdot \\vec{U}$$

with
-    $\\omega$   :: the encounter frequency,
-    g   :: gravitational acceleration
-    $\\tau$ :: kinematice surface tension
-    k   :: wavenumber magnitude
-    d   :: depth
 -   $\\vec{k}$   :: (kx,ky) intrinsic wavenumber vector.
 -   $\\vec{U}_{\\text{rel}}$ :: (Ux,Uy) current velocity vector in the observers reference frame. Typically the
    referene velocity is taken in the earth reference frame.

where the wavenumber vector is defined such that in the intrinsic reference frame (moving with the waves) the wavenumber
is directed into the direction of wave propagation. We refer to this as the intrinsic wavenumber.

Functions:
- `intrinsic_dispersion_relation`, calculate angular frequency for a given wavenumber magnitude and depth
- `inverse_intrinsic_dispersion_relation`, calculate wavenumber magnitude for a given angular frequency and depth
- `encounter_dispersion_relation`, calculate angular frequency for a given wavenumber vector, depth, and
        current velocity
- `intrinsic_group_speed`, calculate the group speed given wave number magnitude and depth.
- `intrinsic_phase_speed`, calculate the phase speed given wave number magnitude  and depth.
- `encounter_group_velocity`, calculate the group velocity given wave number vector, depth and current velocity.
- `encounter_phase_velocity`, calculate the phase velocity given wave number vector, depth and current velocity.
- `encounter_phase_speed`, calculate the phase speed given wave number vector, depth and current velocity.
- `encounter_group_speed`, calculate the group speed given wave number vector, depth and current velocity.

"""

import numpy as np
from numba import jit
from linearwavetheory.settings import PhysicsOptions, NumericalOptions, _parse_options
from linearwavetheory._array_shape_preprocessing import (
    atleast_1d,
    _vector_preprocessing,
)
from linearwavetheory._dispersion_ufuncs import (
    _intrinsic_phase_speed_shallow,
    _intrinsic_phase_speed_intermediate,
    _intrinsic_phase_speed_deep,
    _intrinsic_dispersion_relation_shallow,
    _intrinsic_dispersion_relation_intermediate,
    _intrinsic_dispersion_relation_deep,
    _intrinsic_group_speed_shallow,
    _intrinsic_group_speed_intermediate,
    _intrinsic_group_speed_deep,
    _inverse_intrinsic_dispersion_relation_shallow,
    _inverse_intrinsic_dispersion_relation_intermediate,
    _inverse_intrinsic_dispersion_relation_deep,
)

from linearwavetheory._numba_settings import numba_default
from typing import Union


@jit(**numba_default)
def intrinsic_dispersion_relation(
    wavenumber_magnitude: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
) -> np.ndarray:
    """
    The intrinsic dispersion relation for linear gravity-capillary waves in water of constant depth that relates the
    specific angular frequency $\\sigma$ to a given wavenumber and depth in a reference frame following mean ambient
    flow. I.e.

    $$\\sigma = \\sqrt{ ( gk + \\tau k^3 ) \\tanh(kd) }$$

    with g gravitational acceleration, $\\tau$ kinematic surface tension, k wavenumber and d depth.

    NOTE:
        - if the wavenumber magnitude is zero, the intrinsic frequency is zero.
        - if the depth is smaller or equal to zero, the intrinsic frequency is undefined and np.nan is returned.
        - if the wavenumber magnitude is negative, we return an error (undefined).

    :param  wavenumber_magnitude: The wavenumber_magnitude. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the wavenumber_magnitude array, and calculation is performed pairwise. I.e.
        w[j] is calculated for k[j] and d[j].

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: The intrinsic angular frequency.
    """
    numerical_option, physics_options, _ = _parse_options(None, physics_options, None)

    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    if np.any(wavenumber_magnitude < 0.0):
        raise ValueError("wavenumber_magnitude must be non-negative.")

    args = (
        wavenumber_magnitude,
        depth,
        physics_options.kinematic_surface_tension,
        physics_options.grav,
    )
    if physics_options.wave_regime == "shallow":
        return _intrinsic_dispersion_relation_shallow(*args)

    elif physics_options.wave_regime == "deep":
        return _intrinsic_dispersion_relation_deep(*args)

    else:
        return _intrinsic_dispersion_relation_intermediate(*args)


@jit(**numba_default)
def intrinsic_group_speed(
    wavenumber_magnitude: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
) -> np.ndarray:
    """
    The intrinsic group speed for linear gravity-capillary waves. I.e.

        cg = dw / dk

    NOTE:
        - if the wavenumber magnitude is zero, the group speed is defined as its limiting value as k-> 0
            (the shallow water group speed).
        - if the depth is smaller or equal to zero, the group speed is undefined and np.nan is returned.
        - if the wavenumber magnitude is negative, we return an error (undefined).

    :param  wavenumber_magnitude: The wavenumber_magnitude. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the wavenumber_magnitude array, and calculation is performed pairwise. I.e.
        w[j] is calculated for k[j] and d[j].

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: The intrinsic group speed as a 1 dimensional numpy array.
    """
    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    depth = atleast_1d(depth)

    if np.any(wavenumber_magnitude < 0):
        raise ValueError("Wavenumber magnitude must be positive")

    numerical_option, physics_options, _ = _parse_options(None, physics_options, None)

    args = (
        wavenumber_magnitude,
        depth,
        physics_options.kinematic_surface_tension,
        physics_options.grav,
    )
    if physics_options.wave_regime == "shallow":
        return _intrinsic_group_speed_shallow(*args)

    elif physics_options.wave_regime == "deep":
        return _intrinsic_group_speed_deep(*args)

    else:
        return _intrinsic_group_speed_intermediate(*args)


@jit(**numba_default)
def inverse_intrinsic_dispersion_relation(
    intrinsic_angular_frequency: Union[float, np.ndarray],
    depth: Union[float, np.ndarray] = np.inf,
    physics_options: PhysicsOptions = None,
    numerical_options: NumericalOptions = None,
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

            | w - w_est | / w < relative_tolerance  (default 1e-4)

            and

            | w - w_est |  < absolute_tolerance  (default np.inf)

        where w is the intrinsic angular frequency and w_est is the estimated intrinsic angular frequency based on the
        current estimate of the wavenumber. Per default we do not use the absolute stopping criterium.

    - We exit when either maximum number of iterations (default 10 is reached, or tolerance is achieved.
      Typically only 1 to 2 iterations are needed.

    :param  intrinsic_angular_frequency: The radial frequency as observed from a frame of reference moving with the
        wave. Can be a scalar or a 1 dimensional numpy array.

    :param depth: depth in meters. Can be a scalar or a 1 dimensional numpy array. If a 1d array is provided, it must
        have the same length as the intrinsic_angular_frequency array, and calculation is performed pairwise. I.e.
        k[j] is calculated for w[j] and d[j].

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :param numerical_options: A NumericalOptions object containing the numerical parameters. If None, the default values
        are used.

    :return: The wavenumber as a 1D numpy array. Note that even if a scalar is provided for the intrinsic angular
        frequency, a 1D array is returned.
    """

    # Numba does not recognize "atleast_1d" for scalars
    intrinsic_angular_frequency = atleast_1d(intrinsic_angular_frequency)
    depth = atleast_1d(depth)

    numerical_options, physics_options, _ = _parse_options(
        numerical_options, physics_options, None
    )

    args = (
        intrinsic_angular_frequency,
        depth,
        0.0,
        physics_options.kinematic_surface_tension,
        physics_options.grav,
        numerical_options.maximum_number_of_iterations,
        numerical_options.relative_tolerance,
        numerical_options.absolute_tolerance,
    )
    if physics_options.wave_regime == "shallow":
        return _inverse_intrinsic_dispersion_relation_shallow(*args)

    elif physics_options.wave_regime == "deep":
        return _inverse_intrinsic_dispersion_relation_deep(*args)

    else:
        return _inverse_intrinsic_dispersion_relation_intermediate(*args)


@jit(**numba_default)
def encounter_dispersion_relation(
    intrinsic_wavenumber_vector: np.ndarray,
    depth=np.inf,
    ambient_current_velocity=(0, 0),
    physics_options: PhysicsOptions = None,
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
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as
    a 1D numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the observers reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: Encounter angular frequency (rad/s) as a 1D numpy array.
    """

    intrinsic_wavenumber_vector, ambient_current_velocity = _vector_preprocessing(
        intrinsic_wavenumber_vector, ambient_current_velocity
    )

    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )
    doppler_shift = np.sum(
        intrinsic_wavenumber_vector * ambient_current_velocity, axis=-1
    )

    intrinsic_angular_frequency = intrinsic_dispersion_relation(
        wavenumber_magnitude,
        depth,
        physics_options,
    )

    return intrinsic_angular_frequency + doppler_shift


@jit(**numba_default)
def intrinsic_phase_speed(
    wavenumber_magnitude,
    depth=np.inf,
    physics_options: PhysicsOptions = None,
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

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: The intrinsic phase speed as a 1 dimensional numpy array.
    """
    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)
    depth = atleast_1d(depth)

    if np.any(wavenumber_magnitude < 0):
        raise ValueError("Wavenumber magnitude must be positive")

    (numerical_options, physics_options, _) = _parse_options(
        None, physics_options, None
    )

    # Function pointer would be nicer- but I could not get it to work with numba typing
    args = (
        wavenumber_magnitude,
        depth,
        physics_options.kinematic_surface_tension,
        physics_options.grav,
    )
    if physics_options.wave_regime == "deep":
        c = _intrinsic_phase_speed_deep(*args)

    elif physics_options.wave_regime == "shallow":
        c = _intrinsic_phase_speed_shallow(*args)

    else:
        c = _intrinsic_phase_speed_intermediate(*args)

    return c


@jit(**numba_default)
def encounter_phase_velocity(
    intrinsic_wavenumber_vector: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    ambient_current_velocity=(0.0, 0.0),
    physics_options: PhysicsOptions = None,
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
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as
    a 1D numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the observers reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: A 2D numpy array with shape (N,2) where N is the number of wavenumbers.
    """
    intrinsic_wavenumber_vector, ambient_current_velocity = _vector_preprocessing(
        intrinsic_wavenumber_vector, ambient_current_velocity
    )

    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )

    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)

    _intrinsic_phase_speed = intrinsic_phase_speed(
        wavenumber_magnitude, depth, physics_options
    )
    _intrinsic_phase_speed = np.expand_dims(_intrinsic_phase_speed, axis=-1)
    wavenumber_magnitude = np.expand_dims(wavenumber_magnitude, axis=-1)

    direction = np.where(
        wavenumber_magnitude > 0,
        intrinsic_wavenumber_vector / wavenumber_magnitude,
        np.nan,
    )
    return direction * _intrinsic_phase_speed + ambient_current_velocity


@jit(**numba_default)
def encounter_phase_speed(
    intrinsic_wavenumber_vector,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    physics_options: PhysicsOptions = None,
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

    :param ambient_current_velocity: current (m/s) in the observers reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: A 1D numpy array with shape (N,) where N is the number of wavenumbers.
    """
    _encounter_phase_velocity = encounter_phase_velocity(
        intrinsic_wavenumber_vector,
        depth,
        ambient_current_velocity=ambient_current_velocity,
        physics_options=physics_options,
    )
    return np.sqrt(
        np.sum(_encounter_phase_velocity * _encounter_phase_velocity, axis=-1)
    )


@jit(**numba_default)
def encounter_group_velocity(
    intrinsic_wavenumber_vector,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    physics_options: PhysicsOptions = None,
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
    :param intrinsic_wavenumber_vector: Wavenumber (rad/m) in the intrinsic frame of reference of the wave specified as
    a 1D numpy array (kx,ky), or as a 2D numpy array with shape (N,2) where N is the number of wavenumbers.

    :param depth: Depth (m). May be a scalar or a numpy array. If a numpy array, must have the same number of rows as
    the number of wavenumbers in intrinsic_wavenumber.

    :param ambient_current_velocity: current (m/s) in the earth reference frame specified as a 1D numpy array (u,v),
    which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with the same
    shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber. Default (0,0).

    :param observer_velocity: velocity (m/s) of the observer in the earth reference frame specified as a 1D numpy array
    (u,v), which will be broadcast to an array of the same shape as the wavenumber array, or as a 2D numpy array with
    the same shape as the wavenumber array, in which case the j^th row will be used for the j^th wavenumber.
    Default (0,0).

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: A 2D numpy array with shape (N,2) where N is the number of wavenumbers.
    """
    intrinsic_wavenumber_vector, ambient_current_velocity = _vector_preprocessing(
        intrinsic_wavenumber_vector, ambient_current_velocity
    )

    # Norm does not support axis argument in Numba
    wavenumber_magnitude = np.sqrt(
        np.sum(intrinsic_wavenumber_vector * intrinsic_wavenumber_vector, axis=-1)
    )

    # Must be 1D for the expand_dims to work below
    wavenumber_magnitude = atleast_1d(wavenumber_magnitude)

    _intrinsic_group_speed = intrinsic_group_speed(
        wavenumber_magnitude, depth, physics_options
    )

    _intrinsic_group_speed = np.expand_dims(_intrinsic_group_speed, axis=-1)
    wavenumber_magnitude = np.expand_dims(wavenumber_magnitude, axis=-1)

    direction = np.where(
        wavenumber_magnitude > 0,
        intrinsic_wavenumber_vector / wavenumber_magnitude,
        np.nan,
    )
    return direction * _intrinsic_group_speed + ambient_current_velocity


@jit(**numba_default)
def encounter_group_speed(
    wavenumber,
    depth=np.inf,
    ambient_current_velocity=(0.0, 0.0),
    physics_options: PhysicsOptions = None,
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

    :param physics_options: A PhysicsOptions object containing the physical parameters. If None, the default values are
        used.

    :return: A 1D numpy array with shape (N,) where N is the number of wavenumbers.
    """
    _encounter_group_velocity = encounter_group_velocity(
        wavenumber, depth, ambient_current_velocity, physics_options
    )
    return np.sqrt(
        np.sum(_encounter_group_velocity * _encounter_group_velocity, axis=-1)
    )
