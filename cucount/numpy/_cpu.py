"""Adapter from the numpy frontend to the portable CPU backend.

Since the bindings converged, the native cucountlib.cpu count2 takes the same
Particles/attrs objects and returns the same named channels as the CUDA
backend; what remains here is backend selection support: availability,
decline-by-name, thread defaults and tuning.
"""

import logging
import os

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


def unsupported(particles, battrs, mattrs, wattrs, sattrs, spattrs):
    """Return a reason string if the CPU backend cannot serve this call.

    The vectorised kernel covers s, (s, mu) and (s, pole) binning on a
    cartesian mesh; everything else listed here as served goes through the
    scalar generic path instead, which is slower but produces the same result.
    """
    if cpucount is None:
        return 'CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)'

    names = list(battrs.varnames)
    for name, array in zip(names, battrs.array):
        if name == 'pole' and len(array) > MAX_POLE + 1:
            return f'more than {MAX_POLE + 1} multipoles not implemented'

    if str(getattr(mattrs, 'type', 'cartesian')) not in ('cartesian', 'angular'):
        return f'{mattrs.type} mesh not implemented (only cartesian, angular)'

    for p in particles:
        sizes = dict(p.index_value._sizes)
        extra = {k: v for k, v in sizes.items()
                 if v and k not in ('split', 'individual_weight', 'spin',
                                    'bitwise_weight', 'negative_weight')}
        if extra:
            return f'weight scheme {sorted(extra)} not implemented'
        if sizes.get('spin') and sizes['spin'] != 2:
            return f'spin with {sizes["spin"]} components not implemented (only 2)'

    for name in getattr(sattrs, 'varnames', []):
        if name not in ('s', 'theta'):
            return f'{name} selection not implemented (only s, theta)'
    if getattr(spattrs, 'size', 0) > 1 and not all(
            dict(p.index_value._sizes).get('split') for p in particles):
        return 'jackknife splits need a split label on both catalogues'
    # Matching the CUDA count2, which looks the angular upweight up against
    # one cos(theta) axis; the N-dimensional form belongs to the triplet
    # counts, which this backend does not serve yet.
    angular = getattr(wattrs, 'angular', None)
    if angular is not None and angular.ndim != 1:
        return 'N-dimensional angular weights not implemented (only 1D)'
    return None


# Mirrors ELLMAX in include/layout.h: the highest multipole the triplet
# projection has closed forms for.
ELLMAX = 5


def _angular_ndim(wattrs):
    """Dimensionality of the angular upweight table, or 0 when there is none.

    The Python AngularWeight carries `weight`/`ndim`; `size` belongs to the C
    struct it lowers to.
    """
    angular = getattr(wattrs, 'angular', None)
    if angular is None:
        return 0
    weight = getattr(angular, 'weight', None)
    if weight is None or not weight.size:
        return 0
    return angular.ndim


def _has_bitwise(wattrs):
    bitwise = getattr(wattrs, 'bitwise', None)
    return bitwise is not None and bool(getattr(bitwise, 'weights', None))


def unsupported3(particles, battrs12, battrs13, mattrs, wattrs, sattrs, vetos):
    """Return a reason string if the CPU backend cannot serve this count3."""
    if cpucount is None:
        return 'CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)'

    poles = []
    for name, battrs in [('battrs12', battrs12), ('battrs13', battrs13)]:
        names = list(battrs.varnames)
        if names[:1] not in (['s'], ['theta']):
            return f'{name} {names} not implemented (triplet legs bin in s or theta)'
        if len(names) > 2 or (len(names) == 2 and names[1] != 'pole'):
            return f'{name} {names} not implemented (one separation axis, optionally pole)'
        poles.append(len(names) == 2)
        if poles[-1] and max(battrs.array[1]) > ELLMAX:
            return f'triplet multipoles above ell = {ELLMAX} not implemented'
    if poles[0] != poles[1]:
        return 'a multipole axis on one triplet leg only is not implemented'

    for m in mattrs:
        if str(getattr(m, 'type', 'cartesian')) not in ('cartesian', 'angular'):
            return f'{m.type} mesh not implemented (only cartesian, angular)'

    for p in particles:
        sizes = dict(p.index_value._sizes)
        extra = {k: v for k, v in sizes.items()
                 if v and k not in ('individual_weight', 'negative_weight')}
        if extra:
            return f'triplet weight scheme {sorted(extra)} not implemented'

    for s in list(sattrs) + list(vetos):
        for name in getattr(s, 'varnames', []):
            if name not in ('s', 'theta'):
                return f'{name} selection not implemented (only s, theta)'

    if wattrs is not None and _angular_ndim(wattrs):
        return "angular weights in factorized triplet counts not implemented"
    if wattrs is not None and _has_bitwise(wattrs):
        return "bitwise weights in triplet counts not implemented"
    return None


def count3(cparticles, mattrs, battrs12, battrs13, wattrs=None, sattrs=None,
           vetos=None, tuning=None):
    """Run count3 on the CPU backend. Caller must have checked unsupported3()."""
    tuning = _check_tuning(tuning)
    unknown = set(tuning) - {'nthreads'}
    if unknown:
        raise ValueError(f'CPU backend tuning: count3 accepts only nthreads, got {sorted(unknown)}')

    kwargs = dict(sattrs12=sattrs[0], sattrs13=sattrs[1],
                  veto12=vetos[0], veto13=vetos[1],
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True)
    if wattrs is not None:
        kwargs['wattrs'] = wattrs._to_c()

    result, (mesh_seconds, triplet_seconds) = cpucount.count3(
        *cparticles, *[m._to_c() for m in mattrs], battrs12, battrs13, **kwargs)
    logger.debug('cpu backend: mesh %.1f ms, triplets %.1f ms',
                 mesh_seconds * 1e3, triplet_seconds * 1e3)
    return result


def unsupported3close(particles, battrs, mattrs, wattrs, sattrs, vetos):
    """Return a reason string if the CPU backend cannot serve this count3close.

    ``battrs`` is (battrs12, battrs13, battrs23); the third may be None.
    """
    battrs12, battrs13, battrs23 = battrs
    why = unsupported3(particles, battrs12, battrs13, mattrs, None,
                       sattrs[:2], vetos[:2])
    if why is not None:
        return why

    if battrs23 is not None:
        names = list(battrs23.varnames)
        if names not in (['s'], ['theta']):
            return f'battrs23 {names} not implemented (the (2, 3) axis bins in s or theta)'

    for s in sattrs[2:] + vetos[2:]:
        for name in getattr(s, 'varnames', []):
            if name not in ('s', 'theta'):
                return f'{name} selection not implemented (only s, theta)'

    # The 3-dimensional angular table is what close triplets use; anything
    # else would be indexed against coordinates that do not exist here.
    ndim = _angular_ndim(wattrs)
    if ndim and ndim != 3:
        return (f"{ndim}-dimensional angular weights not implemented in "
                "close triplet counts (only 3D)")
    if _has_bitwise(wattrs):
        return "bitwise weights in triplet counts not implemented"
    return None


def count3close(cparticles, mattrs, battrs, wattrs=None, sattrs=None, vetos=None,
                close_pair='12', tuning=None):
    """Run count3close on the CPU backend. Caller must have checked unsupported3close()."""
    tuning = _check_tuning(tuning)
    unknown = set(tuning) - {'nthreads'}
    if unknown:
        raise ValueError(f'CPU backend tuning: count3close accepts only nthreads, got {sorted(unknown)}')

    kwargs = dict(battrs23=battrs[2],
                  sattrs12=sattrs[0], sattrs13=sattrs[1], sattrs23=sattrs[2],
                  veto12=vetos[0], veto13=vetos[1], veto23=vetos[2],
                  close_pair=str(close_pair),
                  nthreads=int(tuning.get('nthreads') or nthreads()),
                  return_timings=True)
    if wattrs is not None:
        kwargs['wattrs'] = wattrs._to_c()

    result, (mesh_seconds, triplet_seconds) = cpucount.count3close(
        *cparticles, *[m._to_c() for m in mattrs], battrs[0], battrs[1], **kwargs)
    logger.debug('cpu backend: mesh %.1f ms, triplets %.1f ms',
                 mesh_seconds * 1e3, triplet_seconds * 1e3)
    return result


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
