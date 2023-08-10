_nopython = True
_nogil = True
_cache = True
_forceobj = False
_parallel = False
_error_model = "python"
_fastmath = False
_boundscheck = False
_target = "cpu"

numba_default = {
    "nopython": _nopython,
    "nogil": _nogil,
    "cache": _cache,
    "forceobj": _forceobj,
    "parallel": _parallel,
    "error_model": _error_model,
    "fastmath": _fastmath,
    "boundscheck": _boundscheck,
}

numba_default_vectorize = {
    "nopython": _nopython,
    "cache": _cache,
    "target": _target,
}
