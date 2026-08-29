"""Kernel benchmark for the CPU backend, for A/B checks across builds.

The metric is kernel-only pair_seconds (mesh build excluded), best of
--repeats per config, on fixed synthetic catalogues. Run it once per build
and compare the saved JSONs; two builds of the same-named module cannot be
loaded into one process, hence the two-step flow:

    # on the baseline build
    python cpu/bench.py --save base.json
    # on the candidate build
    python cpu/bench.py --against base.json [--fail-above 1.02]

Configs cover the scalar fast path (1D, 2D z, 2D midpoint), the BinMajor
strategy as a control, and a single-thread run. Pin the machine and, for
sub-percent decisions, use an exclusive node: the control row shows the
noise floor of the measurement.
"""
import argparse
import json
import platform
import sys

import numpy as np

BOX = 1000.0
SEDGES = np.linspace(1., 100., 21)
MUEDGES = np.linspace(-1., 1., 11)


def catalog(seed, n):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, BOX, (n, 3)), rng.uniform(0.5, 1.5, n)


def configs(n, n1t, nthreads):
    return [
        ('1D lin, scalar', dict(n=n, ndim=1, los='z', scatter='scalar', nthreads=nthreads)),
        ('2D z, scalar', dict(n=n, ndim=2, los='z', scatter='scalar', nthreads=nthreads)),
        ('2D midpoint, scalar', dict(n=n, ndim=2, los='midpoint', scatter='scalar', nthreads=nthreads)),
        ('1D lin, binmajor', dict(n=n, ndim=1, los='z', scatter='binmajor', nthreads=nthreads)),
        ('1D lin, scalar, 1 thread', dict(n=n1t, ndim=1, los='z', scatter='scalar', nthreads=1)),
    ]


def run(args):
    from cucountlib import cpu

    rows = []
    for name, cfg in configs(args.n, args.n1t, args.nthreads):
        n = cfg.pop('n')
        pos1, w1 = catalog(1, n)
        pos2, w2 = catalog(2, n)
        best = np.inf
        for _ in range(args.repeats):
            _, (_, pair_s) = cpu.count2_arrays(
                pos1, w1, pos2, w2, SEDGES,
                muedges=MUEDGES if cfg['ndim'] == 2 else None,
                boxsize=(BOX,) * 3, bin='lin', los=cfg['los'],
                periodic=True, scatter=cfg['scatter'],
                nthreads=cfg['nthreads'], return_timings=True)
            best = min(best, pair_s)
        rows.append(dict(name=name, n=n, seconds=best))
        print(f'{name:26s} {best * 1e3:9.1f} ms', flush=True)
    return dict(machine=platform.node(), isa=cpu.current_target(),
                nthreads=args.nthreads, rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--save', help='write results to this JSON')
    parser.add_argument('--against', help='baseline JSON to compare with')
    parser.add_argument('--nthreads', type=int, default=16)
    parser.add_argument('--n', type=int, default=400000, help='particles per catalogue')
    parser.add_argument('--n1t', type=int, default=150000, help='particles for the 1-thread config')
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--fail-above', type=float, default=None,
                        help='exit 1 if any scalar-path ratio exceeds this')
    args = parser.parse_args(argv)

    result = run(args)
    if args.save:
        json.dump(result, open(args.save, 'w'), indent=1)
        print(f'saved to {args.save}')
    if args.against:
        base = json.load(open(args.against))
        if base.get('isa') != result['isa'] or base.get('machine') != result['machine']:
            print(f'warning: baseline from {base.get("machine")}/{base.get("isa")}, '
                  f'this run on {result["machine"]}/{result["isa"]}')
        ref = {r['name']: r['seconds'] for r in base['rows']}
        worst = 0.
        for r in result['rows']:
            if r['name'] not in ref:  # baseline may cover only some configs
                print(f'{r["name"]:26s} (not in baseline)')
                continue
            ratio = r['seconds'] / ref[r['name']]
            print(f'{r["name"]:26s} ratio {ratio:5.3f}')
            if 'binmajor' not in r['name']:
                worst = max(worst, ratio)
        if args.fail_above is not None and worst > args.fail_above:
            print(f'FAIL: worst scalar-path ratio {worst:.3f} > {args.fail_above}')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
