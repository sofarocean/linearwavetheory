"""
Linear wave theory package.
===========================

A package containing routines to calculate results from the theory of linear surface gravity waves (see e.g.
Holthuijsen, 2010). As of now, only the linear dispersion relation (and derived parameters such as group speed etc) are
implemented. All functions support numpy arrays as input, including broadcasting and vectorization if dimensions are
compatible.

Example
--------
Example that calculates the wavenumber for a range of frequencies for a water depth of 100 m using different
approximations of the dispersion relation.
    ```python
    >>> from linearwavetheory import inverse_intrinsic_dispersion_relation
    >>> from linearwavetheory.settings import physics_options
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> f = np.linspace(0., 20, 1001)
    >>> w = 2 * np.pi * f
    >>> depth = 100
    >>> k1 = inverse_intrinsic_dispersion_relation(w, depth)
    >>> k2 = inverse_intrinsic_dispersion_relation(w, depth, physics_options(wave_type='gravity'))
    >>> k3 = inverse_intrinsic_dispersion_relation(w, depth, physics_options(wave_regime='deep'))
    >>> k4 = inverse_intrinsic_dispersion_relation(w, depth, physics_options(wave_regime='shallow'))
    >>> k5 = inverse_intrinsic_dispersion_relation(w, depth, physics_options(wave_type='capillary'))
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

Functions
---------
- `linearwavetheory.dispersion.intrinsic_dispersion_relation`, calculate angular frequency for a given wavenumber
    magnitude and depth
- `linearwavetheory.dispersion.inverse_intrinsic_dispersion_relation`, calculate wavenumber magnitude for a given
    angular frequency and depth
- `linearwavetheory.dispersion.encounter_dispersion_relation`, calculate angular frequency for a given wavenumber
vector, depth, and current velocity
- `linearwavetheory.dispersion.intrinsic_group_speed`, calculate the group speed given wave number magnitude and depth.
- `linearwavetheory.dispersion.intrinsic_phase_speed`, calculate the phase speed given wave number magnitude  and depth.
- `linearwavetheory.dispersion.encounter_group_velocity`, calculate the group velocity given wave number vector, depth
    and current velocity.
- `linearwavetheory.dispersion.encounter_phase_velocity`, calculate the phase velocity given wave number vector, depth
    and current velocity.
- `linearwavetheory.dispersion.encounter_phase_speed`, calculate the phase speed given wave number vector, depth and
    current velocity.
- `linearwavetheory.dispersion.encounter_group_speed`, calculate the group speed given wave number vector, depth and
    current velocity.

Classes
-------
- `linearwavetheory.settings.PhysicsOptions`, a class that contains options for the physics of the wave. Allowing
    selection of the wave type (gravity, gravity-capillary, or capillary) and the wave regime (deep, intermediate or
    shallow).
- `linearwavetheory.settings.NumericalOptions`, a class that contains options for the numerical implementation.

Implementation notes:
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

References
==========

    Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.


"""

from .dispersion import (
    intrinsic_dispersion_relation,
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_speed,
    intrinsic_phase_speed,
    encounter_dispersion_relation,
    encounter_phase_speed,
    encounter_group_speed,
)

from .settings import (
    PhysicsOptions,
)

from .stokes_theory._estimate_primary_spectra import (
    estimate_primary_spectrum_from_nonlinear_spectra,
)
from .stokes_theory._higher_order_spectra import (
    nonlinear_wave_spectra_1d,
    nonlinear_wave_spectra_2d,
)
