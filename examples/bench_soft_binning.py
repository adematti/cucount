"""Soft versus hard binning: what it costs, and what it does to the counts.

Run on a GPU node::

    python examples/bench_soft_binning.py

(``scripts/run_bench_soft.sh`` wraps this in an ``srun`` allocation.)

All times are kernel-only seconds reported by the bindings (``count2`` and
``count2_soft`` return their own timings), best of three, on one device with
the same catalogues for every row.  Three things are measured:

* hard ``count2`` -- the denominator;
* soft ``count2_soft`` -- exact taps (double ``exp``) and ``soft_taps='fast'``
  (``__expf`` on the general path only, so it is a no-op on a uniform grid);
* ``count2_soft_vjp`` -- the adjoint, i.e. the cost of a gradient.

The figure follows the pyfcfc soft-histogramming examples: the histograms
themselves, the soft/hard ratio that shows the kernel smoothing the bin
contents, and the cost of that smoothing.

Everything here goes through the **pybind11** binding (``cucountlib.cuda``),
which is what the numpy frontend uses and the only door that reports kernel
timings.  The jax frontend reaches the *same* ``count2_soft`` host function
through an XLA FFI custom call (``examples/soft_hist_grad_jax.py``); measured on
this catalogue the FFI door adds 0.0 ms to an 806 ms kernel and returns counts
that agree with the pybind11 door to 6e-16, so these ratios describe both.
"""

import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR',
                      os.path.join(tempfile.gettempdir(), 'cucount-mpl'))

import numpy as np

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           SplitAttrs, WeightAttrs, _to_c_particles, cudalib)

BOX = 200.0
N = 50000
EDGES = np.linspace(1.0, 60.0, 21)
DS = EDGES[1] - EDGES[0]
SIGMAS_BINS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
NBINS_SWEEP = np.array([8, 16, 32, 64, 128])
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def _catalogs():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, BOX, (N, 3))
    w = rng.uniform(0.5, 1.5, N)
    parts = Particles(pos, w)
    return parts, _to_c_particles(parts)


def _kernel_seconds(timings):
    """Kernel time only: the bindings report (mesh, count) per device."""
    out = timings[1] if isinstance(timings, tuple) else timings
    return float(out[0]) if isinstance(out, (tuple, list)) else float(out)


def main():
    parts, cparts = _catalogs()
    wattrs = WeightAttrs()._to_c()
    sattrs, spattrs = SelectionAttrs(), SplitAttrs()

    def mesh(edges):
        return MeshAttrs(parts, parts, boxsize=BOX, battrs=BinAttrs(s=edges),
                         periodic=True)

    def hard(edges):
        lib = cudalib
        ma = mesh(edges)
        ba = BinAttrs(s=edges)
        return lib.count2(cparts, cparts, ma._to_c(), ba, wattrs=wattrs,
                          sattrs=sattrs, spattrs=spattrs, nthreads=1,
                          return_timings=True)

    def soft(edges, **kwargs):
        lib = cudalib
        ma = mesh(edges)
        ba = BinAttrs(s=edges)
        return lib.count2_soft(cparts, cparts, ma._to_c(), ba, wattrs=wattrs,
                               sattrs=sattrs, spattrs=spattrs, nthreads=1,
                               return_timings=True, **kwargs)

    def vjp(edges, cotangent, **kwargs):
        lib = cudalib
        ma = mesh(edges)
        ba = BinAttrs(s=edges)
        return lib.count2_soft_vjp(cparts, cparts, ma._to_c(), ba, cotangent,
                                   wattrs=wattrs, sattrs=sattrs,
                                   spattrs=spattrs, nthreads=1,
                                   return_timings=True, **kwargs)

    rng = np.random.default_rng(11)

    # --- hard reference on the base grid -----------------------------------
    (hard_res, hard_t) = hard(EDGES)
    hard_counts = np.asarray(hard_res['weight'])
    t_hard = _kernel_seconds(hard_t)

    # --- sigma sweep -------------------------------------------------------
    print(f'catalogs: {N} points, box {BOX:g}, {EDGES.size - 1} bins of '
          f'ds = {DS:g}, smax = {EDGES[-1]:g}')
    print(f'hard count2: {t_hard * 1e3:8.1f} ms, total weight '
          f'{hard_counts.sum():,.4f}\n')

    header = (f'{"sigma/ds":>9} {"sigma":>8} {"taps":>5} {"kmax":>5} '
              f'{"path":>7} {"soft ms":>9} {"ratio":>7} {"fast ms":>9} '
              f'{"ratio":>7} {"vjp ms":>9} {"vjp/hard":>9} {"dW/W":>9}')
    print(header)
    print('-' * len(header))

    sig_rows = []
    for f in SIGMAS_BINS:
        sigma = float(f * DS)
        (res, tim) = soft(EDGES, soft_sigma=sigma)
        counts = np.asarray(res['weight'])
        meta = res['soft']
        t_exact = _kernel_seconds(tim)

        (res_f, tim_f) = soft(EDGES, soft_sigma=sigma, soft_taps='fast')
        t_fast = _kernel_seconds(tim_f)

        cot = rng.normal(size=EDGES.size - 1)
        (vres, vtim) = vjp(EDGES, cot, soft_sigma=sigma)
        t_vjp = _kernel_seconds(vtim)

        dsum = abs(counts.sum() - hard_counts.sum()) / hard_counts.sum()
        sig_rows.append(dict(sigma_bins=float(f), sigma=sigma,
                             taps=int(meta['taps']), kmax=int(meta['kmax']),
                             path=str(meta['path']),
                             copies=int(meta.get('copies', 0)),
                             exact_s=t_exact, fast_s=t_fast, vjp_s=t_vjp,
                             ratio_exact=t_exact / t_hard,
                             ratio_fast=t_fast / t_hard,
                             ratio_vjp=t_vjp / t_hard,
                             total_weight_rel_dev=float(dsum),
                             total_weight_soft=float(counts.sum()),
                             counts=counts.tolist()))
        print(f'{f:9.2f} {sigma:8.3f} {meta["taps"]:5d} {meta["kmax"]:5d} '
              f'{meta["path"]:>7} {t_exact * 1e3:9.1f} {t_exact / t_hard:7.2f} '
              f'{t_fast * 1e3:9.1f} {t_fast / t_hard:7.2f} '
              f'{t_vjp * 1e3:9.1f} {t_vjp / t_hard:9.2f} {dsum:9.2e}')

    # --- bin-count sweep ---------------------------------------------------
    # Sigma is one *local* bin width, so the tap count stays fixed and the row
    # measures the staging cost of more bins, not a wider kernel.
    print('\nbin-count sweep at sigma = 1 ds of each grid (fixed tap count)')
    nb_rows = []
    for nb in NBINS_SWEEP:
        edges = np.linspace(EDGES[0], EDGES[-1], int(nb) + 1)
        (_, ht) = hard(edges)
        th = _kernel_seconds(ht)
        sigma = float(edges[1] - edges[0])
        (res, tim) = soft(edges, soft_sigma=sigma)
        ts = _kernel_seconds(tim)
        meta = res['soft']
        nb_rows.append(dict(nbins=int(nb), hard_s=th, soft_s=ts,
                            ratio=ts / th, path=str(meta['path']),
                            copies=int(meta.get('copies', 0)),
                            taps=int(meta['taps'])))
        print(f'{nb:8d} bins {th * 1e3:9.1f} ms hard {ts * 1e3:9.1f} ms soft '
              f'ratio {ts / th:5.2f}  {meta["path"]:>7} '
              f'({meta.get("copies", 0)} copies), {meta["taps"]} taps')

    # --- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    cols = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, 3))
    show = [r for r in sig_rows if r['sigma_bins'] in (0.5, 1.0, 2.0)]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout='constrained')
    fig.suptitle('Soft vs hard binning via the pybind11 binding (kernel time)',
                 fontsize=11.5)

    # Panel 1: the histograms.
    ax = axes[0, 0]
    ax.step(np.repeat(EDGES, 2)[1:-1], np.repeat(hard_counts, 2), where='mid',
            color='0.3', lw=1.6, label='hard binning')
    ax.plot(centres, hard_counts, 'o', ms=3.5, color='0.3')
    for row, colour in zip(show, cols):
        ax.plot(centres, row['counts'], '-', marker='o', ms=3.5, lw=1.4,
                color=colour,
                label=rf'soft, $\sigma={row["sigma_bins"]:g}\,\Delta s$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('pair counts')
    ax.set_title('Soft versus hard pair counts')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    # Panel 2: soft / hard.
    ax = axes[0, 1]
    ax.axhline(1.0, color='0.3', lw=1.2)
    for row, colour in zip(show, cols):
        ax.plot(centres, np.asarray(row['counts']) / hard_counts, '-',
                marker='o', ms=3.5, lw=1.4, color=colour,
                label=rf'$\sigma={row["sigma_bins"]:g}\,\Delta s$ '
                      f'({row["taps"]} taps)')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('soft / hard counts')
    ax.set_title('The kernel smooths the bin contents')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # Panel 3: cost of the smoothing.
    ax = axes[1, 0]
    f_axis = np.array([r['sigma_bins'] for r in sig_rows])
    ax.plot(f_axis, [r['ratio_exact'] for r in sig_rows], 'o-', lw=1.5,
            color='0.25', label='exact taps')
    ax.plot(f_axis, [r['ratio_fast'] for r in sig_rows], 's--', lw=1.5,
            color=plt.get_cmap('viridis')(0.55),
            label='fast taps (identical here: uniform grid)')
    for r in sig_rows:
        ax.annotate(f'{r["taps"]}', (r['sigma_bins'], r['ratio_exact']),
                    textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=7, color='0.25')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma\ [\Delta s]$')
    ax.set_ylabel('time / hard count2')
    ax.set_title('Cost of soft binning (labels: taps per pair)')
    ax.set_yscale('log')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    # Panel 4: where the histogram is staged, across bin counts.
    ax = axes[1, 1]
    nb_axis = np.array([r['nbins'] for r in nb_rows])
    shared = np.array([r['path'] == 'shared' for r in nb_rows])
    ax.plot(nb_axis[shared], [r['ratio'] for r in nb_rows if r['path'] == 'shared'],
            'o-', lw=1.5, color='0.25', label='shared-memory tile')
    if (~shared).any():
        ax.plot(nb_axis[~shared],
                [r['ratio'] for r in nb_rows if r['path'] != 'shared'],
                's--', lw=1.5, color=plt.get_cmap('viridis')(0.55),
                label='global atomics')
    ax.axhline(1.0, color='0.3', lw=1.2, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('number of bins')
    ax.set_ylabel('soft / hard time')
    ax.set_yscale('log')
    ax.set_title(r'Staging across bin counts ($\sigma = 1\,\Delta s$)')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    png = os.path.join(FIGDIR, 'soft_benchmark.png')
    fig.savefig(png, dpi=150)
    print(f'\nfigure saved to {png}')

    payload = dict(n=N, box=BOX, edges=EDGES.tolist(), ds=float(DS),
                   hard_s=t_hard, hard_total=float(hard_counts.sum()),
                   hard_counts=hard_counts.tolist(), sigma_sweep=sig_rows,
                   nbins_sweep=nb_rows)
    js = os.path.join(FIGDIR, 'soft_benchmark.json')
    with open(js, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f'measurements saved to {js}')


if __name__ == '__main__':
    main()
