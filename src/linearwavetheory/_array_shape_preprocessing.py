import numpy as np
from numba import types, jit
from numba.extending import overload
from ._numba_settings import numba_default


# The following overloading trick is needed because "atleast_1d" is not supported for scalars by default in numba.
def atleast_1d(x) -> np.ndarray:
    if type(x) in types.number_domain:
        return np.array([x])
    return np.atleast_1d(x)


@overload(atleast_1d)
def overloaded_atleast_1d(x):
    if x in types.number_domain:
        return lambda x: np.array([x])
    elif isinstance(x, types.Array):
        return lambda x: np.atleast_1d(x)
    return lambda x: np.atleast_1d(np.array(x))


def atleast_2d(x) -> np.ndarray:
    if x in types.number_domain:
        return np.array([x])
    return np.atleast_2d(np.array(x))


@overload(atleast_2d)
def overloaded_atleast_2d(x):
    if x in types.number_domain:
        return lambda x: np.array([[x]])
    if isinstance(x, types.Array):
        return lambda x: np.atleast_2d(x)
    return lambda x: np.atleast_2d(np.array(x))


@jit(**numba_default)
def _to_2d_array(x, target_rows=0):
    """
    Transform the input array to a 2D array with the specified number of rows and 2 columns. Representing an array of
    2D-vectors.

    The output is always a 2D array with 2 columns. The number of rows is determined by the input array (target_rows==0)
    or specified. If the input array is 1D, it is assumed to be an array of 2 elements, representing a 2D-vector, and
    the vector is repeated to fill the output array.

    Special cases:
    - If the input is None, return an array of zeros with the specified number of rows.
    - If no target rows are specified (=0), the number of rows is determined by the input array.

    """
    target_cols = 2

    x = atleast_2d(x)
    if target_rows == 0:
        target_rows = x.shape[0]

    out = np.zeros((target_rows, target_cols))

    if not (x.shape[1] == 2):
        raise ValueError("x has an invalid shape, must be (n, 2)")

    if x.shape[0] == 1:
        out[:, 0] = x[0, 0]
        out[:, 1] = x[0, 1]

    elif x.shape[0] == target_rows:
        out[:, :] = x[:, :]

    else:
        raise ValueError("x has an invalid shape")

    return out


@jit(**numba_default)
def _vector_preprocessing(intrinsic_wavenumber_vector, ambient_current_velocity):
    # convert to numpy arrays. Intrinsic_wavenumber_vector needs to have at least 2 dimensions, as
    # we specifically define output to be at least 1D

    intrinsic_wavenumber_vector = atleast_1d(intrinsic_wavenumber_vector)
    ambient_current_velocity = atleast_1d(ambient_current_velocity)

    if not (intrinsic_wavenumber_vector.shape[-1] == 2):
        raise ValueError(
            "intrinsic_wavenumber_vector must be a 2D array with shape (...,2)"
        )

    if not (ambient_current_velocity.shape[-1] == 2):
        raise ValueError(
            "ambient_current_velocity must be a 2D array with shape (...,2)"
        )

    return intrinsic_wavenumber_vector, ambient_current_velocity
