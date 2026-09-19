"""Adapter from the numpy frontend to the portable CPU backend.

Since the bindings converged, the native cucountlib.cpu count2 takes the same
Particles/attrs objects and returns the same named channels as the CUDA
backend; what remains here is backend selection support: availability,
thread defaults and tuning. What neither kernel can serve is declined once in
the backend-neutral frontend, not here.
"""

import logging
import os

# One import path for every build: even a standalone `cmake -S cpu` build
# lands the module in a build-tree cucountlib/ (a namespace package that
# merges with the installed one), so PYTHONPATH=<build dir> suffices.
try:
    from cucountlib import cpu as cpulib
except ImportError:  # -DCUCOUNT_BUILD_CPU=OFF
    cpulib = None

logger = logging.getLogger('cucount')

TUNING_KEYS = ('nthreads', 'isa', 'scatter')
"""Tuning keys the CPU backend accepts through the public tuning= keyword.

nthreads  CPU threads (default: CUCOUNT_CPU_NTHREADS, else the affinity mask)
isa       Highway target to pin for this call, e.g. 'AVX2' (default: automatic)
scatter   'scalar' (default) or 'binmajor' accumulation strategy
"""


def available():
    return cpulib is not None


def setup_logging(level):
    """Sync the level into the extension (a no-op when it is not built)."""
    if cpulib is not None:
        cpulib.setup_logging(level)


def nthreads():
    """CPU threads, which is not what cucount's nthreads means (that is GPUs)."""
    n = os.environ.get('CUCOUNT_CPU_NTHREADS')
    return int(n) if n else len(os.sched_getaffinity(0))


def _check_tuning(tuning):
    """Validate the tuning dict, rejecting unknown keys by name."""
    tuning = dict(tuning or {})
    unknown = set(tuning) - set(TUNING_KEYS)
    if unknown:
        raise ValueError(f'CPU backend tuning: unknown keys {sorted(unknown)}; accepted: {list(TUNING_KEYS)}')
    return tuning


def unavailable():
    """Return a reason string if this backend cannot run at all, else None.

    Nothing else is declined here. What the two kernels cannot serve is the
    same for both, so it is checked once in the backend-neutral frontend
    (_check_count2 / _check_count3) before either is chosen -- otherwise the
    CPU backend reports a limit by name while CUDA goes ahead and computes
    something else.
    """
    if cpulib is None:
        return 'CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)'
    return None


def count3(cparticles, mattrs, battrs12, battrs13, wattrs=None, sattrs=None,
           vetos=None, tuning=None):
    """Run count3 on the CPU backend. The request is validated in the frontend."""
    tuning = _check_tuning(tuning)
    unknown = set(tuning) - {'nthreads'}
    if unknown:
        raise ValueError(f'CPU backend tuning: count3 accepts only nthreads, got {sorted(unknown)}')

    kwargs = dict(sattrs12=sattrs[0], sattrs13=sattrs[1],
                  veto12=vetos[0], veto13=vetos[1],
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True, wattrs=wattrs._to_c())

    result, (mesh_seconds, count_seconds) = cpulib.count3(
        *cparticles, *[m._to_c() for m in mattrs], battrs12, battrs13, **kwargs)
    logger.debug('cpu backend: mesh %.1f ms, count %.1f ms',
                 mesh_seconds * 1e3, count_seconds * 1e3)
    return result


def count3close(cparticles, mattrs, battrs, wattrs=None, sattrs=None, vetos=None,
                close_pair='12', tuning=None):
    """Run count3close on the CPU backend. The request is validated in the frontend."""
    tuning = _check_tuning(tuning)
    unknown = set(tuning) - {'nthreads'}
    if unknown:
        raise ValueError(f'CPU backend tuning: count3close accepts only nthreads, got {sorted(unknown)}')

    # battrs23 is the one that really can be None: the (2, 3) axis is optional.
    kwargs = dict(battrs23=battrs[2],
                  sattrs12=sattrs[0], sattrs13=sattrs[1], sattrs23=sattrs[2],
                  veto12=vetos[0], veto13=vetos[1], veto23=vetos[2],
                  close_pair=str(close_pair),
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True, wattrs=wattrs._to_c())

    result, (mesh_seconds, count_seconds) = cpulib.count3close(
        *cparticles, *[m._to_c() for m in mattrs], battrs[0], battrs[1], **kwargs)
    logger.debug('cpu backend: mesh %.1f ms, count %.1f ms',
                 mesh_seconds * 1e3, count_seconds * 1e3)
    return result


def count2(cparticles, battrs, mattrs, wattrs=None, sattrs=None, spattrs=None,
           tuning=None):
    """Run count2 on the CPU backend. The request is validated in the frontend.

    ``cparticles`` are already-converted native Particles (the same objects the
    CUDA path consumes); the attrs cross into the extension through pybind's
    foreign module_local loading.
    """
    tuning = _check_tuning(tuning)

    # Selections and splits go through too: the binding lowers them, and the
    # kernel applies the selection as a per-pair veto.
    kwargs = dict(scatter=str(tuning.get('scatter', 'scalar')),
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True,
                  wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs)

    isa = tuning.get('isa')
    if isa is not None and cpulib.set_target(isa) is None:
        cpulib.set_target('')
        raise ValueError(f'CPU backend tuning: ISA {isa!r} unknown or unavailable; '
                         f'available: {cpulib.available_targets()}')
    try:
        result, (mesh_seconds, count_seconds) = cpulib.count2(
            *cparticles, mattrs._to_c(), battrs, **kwargs)
    finally:
        if isa is not None:
            cpulib.set_target('')
    # The mesh build is still serial, so its share grows with thread count.
    logger.debug('cpu backend: mesh %.1f ms, count %.1f ms',
                 mesh_seconds * 1e3, count_seconds * 1e3)
    return result
