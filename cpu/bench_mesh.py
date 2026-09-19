"""Kernel run time against mesh resolution, for the cells-per-smax default.

The mesh comes from MeshAttrs, and how fine it should be is a trade-off between
wasted distance evaluations and per-cell overhead: finer cells fit the swept
volume to the sphere of radius smax more tightly, but leave fewer particles per
cell for the inner loop to vectorise over. The CUDA backend wants one end of
that range, the Highway kernel the other. This measures the CPU end.

meshsize is set explicitly here, so cells_per_smax = meshsize * smax / boxsize
is exact rather than inferred through the O(nparticles) cap, which would
otherwise confound the sweep at large n. Every resolution gives the same
counts, which is asserted, so this is purely a timing question.

    python cpu/bench_mesh.py            # the sweep over resolutions
    python cpu/bench_mesh.py --defaults # the candidate defaults, at one size

Run it on a compute node, not a login node.
"""

import argparse
import time

import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, Particles, count2

BOX, SMAX = 1000., 100.
MU = np.linspace(-1., 1., 9)
CELLS_PER_SMAX = [1, 2, 3, 6]


def catalogs(n):
    out = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        out.append((rng.uniform(0., BOX, (n, 3)), rng.uniform(0.5, 1.5, n)))
    return [Particles(pos, w) for pos, w in out]


def time_one(particles, battrs, meshsize, nthreads, repeat):
    mattrs = MeshAttrs(*particles, battrs=battrs, meshsize=meshsize)
    assert np.all(mattrs.meshsize == meshsize), 'the mesh was not the one requested'
    best, counts = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        counts = count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu',
                        tuning={'nthreads': nthreads})['weight']
        best = min(best, time.perf_counter() - t0)
    return best, counts


def binning(ndim):
    sedges = np.linspace(1., SMAX, 11)
    return BinAttrs(s=sedges, mu=(MU, 'midpoint')) if ndim == 2 else BinAttrs(s=sedges)


def sweep(sizes, nthreads, repeat):
    for n in sizes:
        particles = catalogs(n)
        for ndim in (1, 2):
            battrs = binning(ndim)
            print(f'\nn = {n:,}  binning {"s" if ndim == 1 else "(s, mu)"}  '
                  f'box {BOX:.0f}  smax {SMAX:.0f}  {nthreads} threads  '
                  f'best of {repeat}', flush=True)
            print(f'  {"c":>2}  {"meshsize":>8}  {"n/cell":>8}  {"seconds":>8}  {"vs c=1":>7}')
            ref, base = None, None
            for c in CELLS_PER_SMAX:
                meshsize = int(round(c * BOX / SMAX))
                secs, counts = time_one(particles, battrs, meshsize, nthreads, repeat)
                if ref is None:
                    ref, base = counts, secs
                else:
                    assert np.allclose(counts, ref, rtol=1e-9, atol=0), f'counts differ at c={c}'
                print(f'  {c:>2}  {meshsize:>8}  {n / meshsize**3:>8.1f}  {secs:>8.3f}  '
                      f'{secs / base:>6.2f}x', flush=True)


def defaults(n, nthreads, repeat):
    """The candidate defaults as the kernel would receive them, cap included.

    Pick n so the smax term binds rather than the cap, which is the regime where
    the rules differ at all.
    """
    particles = catalogs(n)
    cap = (0.5 * n) ** (1. / 3.)
    for ndim in (1, 2):
        battrs = binning(ndim)
        print(f'\nn = {n:,}  binning {"s" if ndim == 1 else "(s, mu)"}  '
              f'cap cbrt(0.5n) = {cap:.1f}  best of {repeat}', flush=True)
        print(f'  {"rule":<34} {"meshsize":>8} {"c":>5} {"seconds":>8} {"vs old":>7}')
        ref, base = None, None
        for label, c in [('old kernel heuristic (c = 1)', 1.),
                         ('CUDA default         (c = 6)', 6.),
                         ('cpu default          (c = 2)', 2.)]:
            meshsize = int(max(min(c * BOX / SMAX, cap), 1))
            secs, counts = time_one(particles, battrs, meshsize, nthreads, repeat)
            if ref is None:
                ref, base = counts, secs
            else:
                assert np.allclose(counts, ref, rtol=1e-9, atol=0)
            print(f'  {label:<34} {meshsize:>8} {meshsize * SMAX / BOX:>5.1f} '
                  f'{secs:>8.3f} {secs / base:>6.2f}x', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--defaults', action='store_true',
                        help='compare the candidate defaults instead of sweeping')
    parser.add_argument('--nthreads', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--sizes', type=int, nargs='+', default=[50_000, 200_000])
    args = parser.parse_args()
    if args.defaults:
        defaults(500_000, args.nthreads, args.repeat)
    else:
        sweep(args.sizes, args.nthreads, args.repeat)
