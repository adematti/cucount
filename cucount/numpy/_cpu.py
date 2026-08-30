"""Adapter from the numpy frontend to the portable CPU backend.

Since the bindings converged, the native cucountlib.cpu count2 takes the same
Particles/attrs objects and returns the same named channels as the CUDA
backend; what remains here is backend selection support: availability,
decline-by-name, thread defaults and tuning.
"""

import logging
import os

import numpy as np

# One import path for every build: even a standalone `cmake -S cpu` build
# lands the module in a build-tree cucountlib/ (a namespace package that
# merges with the installed one), so PYTHONPATH=<build dir> suffices.
try:
    from cucountlib import cpu as cpucount
except ImportError:  # -DCUCOUNT_BUILD_CPU=OFF
    cpucount = None

logger = logging.getLogger('cucount')

# Mirrors MAX_POLE in include/common.h: the size of the kernel's Legendre cache.
MAX_POLE = 8


TUNING_KEYS = ('nthreads', 'isa', 'scatter')
"""Tuning keys the CPU backend accepts through the public tuning= keyword.

nthreads  CPU threads (default: CUCOUNT_CPU_NTHREADS, else the affinity mask)
isa       Highway target to pin for this call, e.g. 'AVX2' (default: automatic)
scatter   'scalar' (default) or 'binmajor' accumulation strategy
"""


def available():
    return cpucount is not None


def setup_logging(level):
    """Sync the level into the extension (a no-op when it is not built)."""
    if cpucount is not None:
        cpucount.setup_logging(level)


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


def _is_linear(edges):
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 3:
        return True
    d = np.diff(edges)
    return bool(np.allclose(d, d[0], rtol=1e-12, atol=0))


def unsupported(particles, battrs, mattrs, wattrs, sattrs, spattrs):
    """Return a reason string if the CPU backend cannot serve this call."""
    if cpucount is None:
        return 'CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)'

    names = list(battrs.varnames)
    if names not in (['s'], ['s', 'mu'], ['s', 'pole']):
        return f'binning {names} not implemented (only s, (s, mu), (s, pole))'
    if names == ['s', 'mu'] and not _is_linear(battrs.array[1]):
        return 'non-linear mu binning not implemented'
    if names == ['s', 'pole'] and len(battrs.array[1]) > MAX_POLE + 1:
        return f'more than {MAX_POLE + 1} multipoles not implemented'

    if str(getattr(mattrs, 'type', 'cartesian')) != 'cartesian':
        return f'{mattrs.type} mesh not implemented (only cartesian)'

    for p in particles:
        sizes = dict(p.index_value._sizes)
        extra = {k: v for k, v in sizes.items()
                 if v and k not in ('individual_weight', 'spin',
                                    'bitwise_weight', 'negative_weight')}
        if extra:
            return f'weight scheme {sorted(extra)} not implemented'
        if sizes.get('spin') and sizes['spin'] != 2:
            return f'spin with {sizes["spin"]} components not implemented (only 2)'

    for name in getattr(sattrs, 'varnames', []):
        if name not in ('s', 'theta'):
            return f'{name} selection not implemented (only s, theta)'
    if getattr(spattrs, 'size', 0) > 1:
        return 'jackknife splits not implemented'
    angular = getattr(wattrs, 'angular', None)
    if angular is not None and angular.ndim != 1:
        return 'N-dimensional angular weights not implemented (only 1D)'
    return None


def count2(cparticles, battrs, mattrs, wattrs=None, sattrs=None, spattrs=None,
           tuning=None):
    """Run count2 on the CPU backend. Caller must have checked unsupported().

    ``cparticles`` are already-converted native Particles (the same objects the
    CUDA path consumes); the attrs cross into the extension through pybind's
    foreign module_local loading.
    """
    tuning = _check_tuning(tuning)

    kwargs = dict(scatter=str(tuning.get('scatter', 'scalar')),
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True)
    if wattrs is not None:
        kwargs['wattrs'] = wattrs._to_c()
    # Selections and splits go through too: the binding lowers them, and the
    # kernel applies the selection as a per-pair veto.
    if sattrs is not None:
        kwargs['sattrs'] = sattrs
    if spattrs is not None:
        kwargs['spattrs'] = spattrs

    isa = tuning.get('isa')
    if isa is not None and cpucount.set_target(isa) is None:
        cpucount.set_target('')
        raise ValueError(f'CPU backend tuning: ISA {isa!r} unknown or unavailable; '
                         f'available: {cpucount.available_targets()}')
    try:
        result, (mesh_seconds, pair_seconds) = cpucount.count2(
            *cparticles, mattrs._to_c(), battrs, **kwargs)
    finally:
        if isa is not None:
            cpucount.set_target('')
    # The mesh build is still serial, so its share grows with thread count.
    logger.debug('cpu backend: mesh %.1f ms, pairs %.1f ms',
                 mesh_seconds * 1e3, pair_seconds * 1e3)
    return result
