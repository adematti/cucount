"""Adapter from the numpy frontend's objects to the portable CPU backend.

This is a shim between the Attrs/Particles objects and the native types the
CPU backend understands.
"""

import logging
import os

import numpy as np

try:
    from cucountlib import cpucount
except ImportError:  # built out of tree, or -DCUCOUNT_BUILD_CPU=OFF
    try:
        import cpucount
    except ImportError:
        cpucount = None

logger = logging.getLogger('cucount')

# The CPU kernel bins mu linearly; s may use any of the three policies.
_LOS = {'z': 'z', 'midpoint': 'midpoint'}


def available():
    return cpucount is not None


def nthreads():
    """CPU threads, which is not what cucount's nthreads means (that is GPUs)."""
    n = os.environ.get('CUCOUNT_CPU_NTHREADS')
    return int(n) if n else len(os.sched_getaffinity(0))


def _bin_kind(edges):
    """Match the edge array to the cheapest policy that reproduces it exactly."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 3:
        return 'edges'
    d = np.diff(edges)
    if np.allclose(d, d[0], rtol=1e-12, atol=0):
        return 'lin'
    if edges[0] > 0 and np.allclose(np.diff(np.log(edges)),
                                    np.log(edges[1] / edges[0]),
                                    rtol=1e-12, atol=0):
        return 'log'
    return 'edges'


def unsupported(particles, battrs, mattrs, wattrs, sattrs, spattrs):
    """Return a reason string if the CPU backend cannot serve this call."""
    if cpucount is None:
        return 'CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)'

    names = list(battrs.varnames)
    if names not in (['s'], ['s', 'mu']):
        return f'binning {names} not implemented (only s, (s, mu))'
    if names == ['s', 'mu']:
        los = battrs.losnames[1]
        if los not in _LOS:
            return f'line of sight {los!r} not implemented (only z, midpoint)'
        if _bin_kind(battrs.array[1]) != 'lin':
            return 'non-linear mu binning not implemented'

    if str(getattr(mattrs, 'type', 'cartesian')) != 'cartesian':
        return f'{mattrs.type} mesh not implemented (only cartesian)'

    for p in particles:
        sizes = dict(p.index_value._sizes)
        extra = {k: v for k, v in sizes.items()
                 if v and k != 'individual_weight'}
        if extra:
            return f'weight scheme {sorted(extra)} not implemented'

    if getattr(sattrs, 'ndim', 0):
        return 'selection attributes not implemented'
    if getattr(spattrs, 'size', 0) > 1:
        return 'jackknife splits not implemented'
    if getattr(getattr(wattrs, 'angular', None), 'size', 0):
        return 'angular weights not implemented'
    if getattr(getattr(wattrs, 'bitwise', None), 'nrealizations', 0):
        return 'bitwise weights not implemented'
    return None


def count2(particles, battrs, mattrs):
    """Run count2 on the CPU backend. Caller must have checked unsupported()."""
    def arrays(p):
        w = p.get('individual_weight')
        w = np.ascontiguousarray(w[0], dtype=float) if w else np.ones(p.size)
        return np.ascontiguousarray(p.positions, dtype=float), w

    pos1, w1 = arrays(particles[0])
    pos2, w2 = arrays(particles[1])

    names = list(battrs.varnames)
    sedges = np.ascontiguousarray(battrs.array[0], dtype=float)
    muedges = (np.ascontiguousarray(battrs.array[1], dtype=float)
               if names == ['s', 'mu'] else None)
    los = _LOS[battrs.losnames[1]] if muedges is not None else 'z'

    boxsize = np.asarray(mattrs.boxsize, dtype=float)
    origin = np.asarray(mattrs.boxcenter, dtype=float) - boxsize / 2.

    counts, (mesh_seconds, pair_seconds) = cpucount.count2(
        pos1, w1, pos2, w2, sedges,
        muedges=muedges,
        boxsize=tuple(boxsize), origin=tuple(origin),
        bin=_bin_kind(sedges), los=los,
        periodic=bool(mattrs.periodic), nthreads=nthreads(),
        return_timings=True)
    # The mesh build is still serial, so its share grows with thread count.
    logger.debug('cpu backend: mesh %.1f ms, pairs %.1f ms',
                 mesh_seconds * 1e3, pair_seconds * 1e3)
    return {'weight': counts.reshape(battrs.shape)}
