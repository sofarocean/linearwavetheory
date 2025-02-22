from linearwavetheory.settings import _parse_options


def get_wave_regime(**kwargs):
    numerical_options, physics_options, nonlinear_options = _parse_options(
        kwargs.get("physics_options", None),
        kwargs.get("physics_options", None),
        kwargs.get("nonlinear_options", None),
    )
    return physics_options.wave_regime
