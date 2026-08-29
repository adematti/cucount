"""Adapter from the numpy frontend to the CUDA backend.

Mirrors :mod:`_cpu`: availability, decline-by-name, tuning, and the calls
themselves. The extension is imported lazily, so a build with
``-DCUCOUNT_BUILD_CUDA=OFF`` (or a machine without CUDA libraries) can still
import ``cucount.numpy`` and serve ``backend='cpu'``; nothing outside this
module may touch ``cucountlib.cuda``.
"""

import logging

import numpy as np

logger = logging.getLogger('cucount')

_lib = None


TUNING_KEYS = ('nthreads',)
"""Tuning keys the CUDA backend accepts through the public tuning= keyword.

nthreads  number of GPUs (within the same node) to run in parallel on

Kernel launch geometry is chosen by occupancy (CONFIGURE_KERNEL_LAUNCH) and
is not exposed.
"""


# Mirrors ELLMAX in cuda/include/count3close.h. The kernel applies it as `ellmax = MIN(ellmax, ELLMAX)`,
# i.e. it CLAMPS SILENTLY: asking for higher orders returns fewer poles than the binning describes,
# with no error. Keep in sync with the header (raising it also needs MMAX_SIZE = ELLMAX + 1).
KERNEL_ELLMAX = 5


def _level_name():
    return logging.getLevelName(logger.getEffectiveLevel()).lower()


def lib():
    """Return the CUDA extension, importing it on first use."""
    global _lib
    if _lib is None:
        try:
            import cucountlib.cuda
        except ImportError as exc:
            raise ImportError(
                "the cucount CUDA extension is not available (was cucount built with "
                "-DCUCOUNT_BUILD_CUDA=OFF, or is CUDA missing?); only backend='cpu' "
                "can be served") from exc
        _lib = cucountlib.cuda
        _lib.setup_logging(_level_name())
    return _lib


def available():
    """Whether the extension can be imported -- WITHOUT importing it, so that
    asking the question does not pull in the CUDA libraries."""
    if _lib is not None:
        return True
    import importlib.util
    try:
        return importlib.util.find_spec('cucountlib.cuda') is not None
    except (ImportError, ValueError):
        return False


def setup_logging(level):
    """Sync the level into the extension, but only once it has been imported:
    importing it here would defeat the laziness this module exists for."""
    if _lib is not None:
        _lib.setup_logging(level)


def _check_tuning(tuning):
    """Validate the tuning dict, rejecting unknown keys by name."""
    tuning = dict(tuning or {})
    unknown = set(tuning) - set(TUNING_KEYS)
    if unknown:
        raise ValueError(f'CUDA backend tuning: unknown keys {sorted(unknown)}; accepted: {list(TUNING_KEYS)} '
                         '(kernel launch geometry is chosen by occupancy and is not exposed yet)')
    return tuning


def unsupported(particles, battrs, mattrs, wattrs, sattrs, spattrs):
    """Return a reason string if the CUDA backend cannot serve this call.

    It serves everything the frontend can express, so this is None; it exists
    so both backends answer the same question the same way.
    """
    if not available():
        return 'CUDA backend not built (-DCUCOUNT_BUILD_CUDA=ON)'
    return None


def check_kernel_ells(*battrs_or_ells):
    """
    Raise if any requested multipole exceeds what the count3 kernel can compute.

    Without this the kernel clamps silently, and the mismatch surfaces far downstream -- as an
    opaque IndexError while packing the poles, or, worse, as counts quietly built from the wrong
    multipoles.
    """
    from . import _get_ells  # deferred: the frontend imports this module

    ells = [_get_ells(x) for x in battrs_or_ells if x is not None]  # battrs23 is optional
    ells = [ell for ell in ells if ell is not None and len(ell)]
    if not ells: return
    requested = max(max(np.ravel(ell)) for ell in ells)
    if requested > KERNEL_ELLMAX:
        raise ValueError(
            f'requested multipoles up to ell = {requested}, but the count3/count3close cucount kernel supports only '
            f'ell <= {KERNEL_ELLMAX} (ELLMAX in cuda/include/count3close.h) and would clamp silently. '
            f'Lower the requested multipoles, or rebuild cucount with a larger ELLMAX '
            f'(and MMAX_SIZE = ELLMAX + 1).')


def count2(cparticles, battrs, mattrs, wattrs=None, sattrs=None, spattrs=None,
           tuning=None):
    """Run count2 on the CUDA backend.

    ``cparticles`` are already-converted native Particles (the same objects the
    CPU path consumes); the attrs cross into the extension through pybind's
    foreign module_local loading.
    """
    tuning = _check_tuning(tuning)
    return lib().count2(*cparticles, mattrs._to_c(), battrs=battrs,
                        wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
                        nthreads=tuning.get('nthreads', 1))


def count3close(cparticles, mattrs1, mattrs2, mattrs3, battrs12, battrs13,
                battrs23=None, wattrs=None, sattrs12=None, sattrs13=None,
                sattrs23=None, veto12=None, veto13=None, veto23=None,
                close_pair=(1, 2), tuning=None):
    """Run count3close on the CUDA backend (the CPU backend has no triplets)."""
    tuning = _check_tuning(tuning)
    return lib().count3close(
        *cparticles,
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12=battrs12,
        battrs13=battrs13,
        battrs23=battrs23,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        sattrs23=sattrs23,
        veto12=veto12,
        veto13=veto13,
        veto23=veto23,
        close_pair=close_pair,
        nthreads=tuning.get('nthreads', 1),
    )


def count3(cparticles, mattrs1, mattrs2, mattrs3, battrs12, battrs13,
           wattrs=None, sattrs12=None, sattrs13=None, veto12=None, veto13=None,
           tuning=None):
    """Run count3 on the CUDA backend (the CPU backend has no triplets)."""
    tuning = _check_tuning(tuning)
    return lib().count3(
        *cparticles,
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12=battrs12,
        battrs13=battrs13,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        veto12=veto12,
        veto13=veto13,
        nthreads=tuning.get('nthreads', 1),
    )
