"""Soft pair counts through the lsstypes frontend, next to the hard binning.

``cucount.types.count2`` places every pair in exactly one bin; ``count2_soft``
shares it between neighbouring bins with a Gaussian softmax. This script does
the comparison a user actually cares about:

* isotropic soft counts against the hard ones, on a **logarithmic** separation
  grid (so ``sigma`` scales with each bin width) and with an explicit constant
  ``sigma``;
* weight conservation: softening moves counts between bins, it never loses them;
* soft multipoles: ``pole=([0, 2, 4], 'midpoint')`` also smooths the mu axis, so
  the quadrupole/hexadecapole come out differentiable instead of piecewise
  constant.

Run it on a GPU node (the soft kernel is CUDA-only)::

    python examples/soft_counts_lsstypes.py

and it writes ``examples/figures/soft_counts_lsstypes.png``.
"""

import os

import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, Particles
from cucount.types import count2, count2_soft
from lsstypes import Count2Poles

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BOX = 400.0
N = 4000
EDGES = np.logspace(np.log10(1.0), np.log10(120.0), 21)
ELLS = [0, 2, 4]
LOS = 'midpoint'
CUTOFF = 1e-13


def catalogue(n=N, box=BOX, seed=7):
    """A clustered catalogue: a few Gaussian clumps plus a smooth background.

    A uniform catalogue in a periodic box is scale-free, so a soft histogram of
    it looks exactly like the hard one up to noise; clumps give structure that
    the smoothing visibly changes.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 24
    centres = rng.uniform(0.0, box, (n_clusters, 3))
    n_clustered = int(0.6 * n)
    which = rng.integers(0, n_clusters, n_clustered)
    positions = centres[which] + rng.normal(0.0, 6.0, (n_clustered, 3))
    n_random = n - n_clustered
    positions = np.vstack([positions, rng.uniform(0.0, box, (n_random, 3))])
    positions %= box
    weights = rng.uniform(0.5, 1.5, n)
    return Particles(positions, weights)


def _mattrs(p, battrs, box=BOX):
    return MeshAttrs(p, battrs=battrs, periodic=True, boxsize=box)


def _edges(battrs, name='s'):
    """The (low, high) pairs from BinAttrs as a single monotonic edge array."""
    low_high = battrs.edges()[name]
    return np.append(low_high[:, 0], low_high[-1, 1])


def _counts(observable):
    """Raw counts, stacking the multipoles of a Count2Poles into columns."""
    if isinstance(observable, Count2Poles):
        return np.stack([np.asarray(pole.values('counts')) for pole in observable], axis=-1)
    return np.asarray(observable.values('counts'))


def _values(observable):
    """Normalized counts (counts / norm), i.e. the correlation itself."""
    if isinstance(observable, Count2Poles):
        return np.stack([np.asarray(pole.value()) for pole in observable], axis=-1)
    return np.asarray(observable.value())


def main():
    p = catalogue()

    # --- isotropic counts -------------------------------------------------
    battrs_s = BinAttrs(s=EDGES)
    mattrs = _mattrs(p, battrs_s)
    hard = count2(p, battrs=battrs_s, mattrs=mattrs)['weight']
    soft_default = count2_soft(p, battrs=battrs_s, mattrs=mattrs)['weight']
    soft_wide = count2_soft(p, battrs=battrs_s, mattrs=mattrs, soft_sigma=4.0)['weight']

    soft_meta = dict(soft_default.attrs)
    print(f'grid: {EDGES.size - 1} logarithmic bins in [{EDGES[0]:g}, {EDGES[-1]:g}]')
    print(f"sigma (default) = {soft_meta['soft_sigma']:.4g} "
          f"[{soft_meta['soft_sigma_kind']}], cutoff = {soft_meta['soft_cutoff']:g}, "
          f"kmax = {soft_meta['soft_kmax']}")

    hard_c = _counts(hard)
    soft_c = _counts(soft_default)
    wide_c = _counts(soft_wide)
    print(f'\nweight conservation: hard {hard_c.sum():.10g}, '
          f'soft {soft_c.sum():.10g}, soft wide {wide_c.sum():.10g} '
          f'(relative spread {abs(soft_c.sum() / hard_c.sum() - 1):.2e})')

    # --- soft multipoles --------------------------------------------------
    battrs_p = BinAttrs(s=EDGES, pole=(ELLS, LOS))
    mattrs_p = _mattrs(p, battrs_p)
    hard_p = count2(p, battrs=battrs_p, mattrs=mattrs_p)['weight']
    soft_p = count2_soft(p, battrs=battrs_p, mattrs=mattrs_p, soft_mu_sigma=0.05)['weight']

    hard_v, soft_v = _values(hard_p), _values(soft_p)
    print(f'\nmultipoles {ELLS} with los = {LOS!r}: '
          f"mu grid {soft_p.get(0).attrs['soft_mu_bins']} bins, "
          f"sigma_mu {soft_p.get(0).attrs['soft_mu_sigma']:g}, "
          f"mu_kmax {soft_p.get(0).attrs['soft_mu_kmax']}")
    for i, ell in enumerate(ELLS):
        mask = hard_c.sum() > 0
        rel = np.abs(soft_v[mask, i] / hard_v[mask, i] - 1)
        print(f'  ell = {ell}: median |soft/hard - 1| = {np.median(rel):.2e}, '
              f'max = {rel.max():.2e}')

    # --- figure -----------------------------------------------------------
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    sep = _edges(battrs_s)
    viridis = plt.get_cmap('viridis')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout='constrained')
    fig.suptitle('Soft pair counts through the lsstypes frontend', fontsize=11.5)

    ax = axes[0, 0]
    ax.step(np.repeat(sep, 2)[1:-1], np.repeat(hard_c, 2), where='mid',
            color='0.3', lw=1.6, label='hard binning')
    ax.plot(centres, hard_c, 'o', ms=3.5, color='0.3')
    ax.plot(centres, soft_c, '-o', ms=3.5, lw=1.4, color=viridis(0.5),
            label=rf"soft, $\sigma=\Delta_i/2$ ({soft_meta['soft_sigma']:.2f} at the peak)")
    ax.plot(centres, wide_c, '--', lw=1.3, color=viridis(0.8),
            label=r'soft, $\sigma=4$ (absolute)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('pair counts')
    ax.set_title('Isotropic counts on a logarithmic grid')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    ax = axes[0, 1]
    ax.axhline(1.0, color='0.3', lw=1.2)
    ax.plot(centres, soft_c / hard_c, '-o', ms=3.5, lw=1.4, color=viridis(0.5),
            label=r'$\sigma=\Delta_i/2$')
    ax.plot(centres, wide_c / hard_c, '--', lw=1.3, color=viridis(0.8),
            label=r'$\sigma=4$ (absolute)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('soft / hard counts')
    # A width-scaled sigma smears by roughly the same amount in every bin; a
    # fixed sigma is large next to the small bins and small next to the big ones,
    # which is why it drifts back towards the hard counts at large separations.
    ax.set_title(rf'Weight is conserved: $\sum$ counts = {soft_c.sum():.6g}')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    for ax, ill, label in ((axes[1, 0], 0, r'$\ell = 0$'), (axes[1, 1], 2, r'$\ell = 2$')):
        ax.plot(centres, hard_v[:, ill], 'o', ms=3.5, color='0.3', label='hard')
        ax.plot(centres, soft_v[:, ill], '-o', ms=3.5, lw=1.4, color=viridis(0.5),
                label=r'soft ($\sigma_\mu=0.05$)')
        ax.set_xscale('log')
        ax.set_xlabel(r'$s$')
        ax.set_ylabel(r'$C_\ell(s)$')
        ax.set_title(f'{label}: counts / norm, los = {LOS!r}')
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, which='both')

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, 'soft_counts_lsstypes.png')
    fig.savefig(out, dpi=150)
    print(f'\nfigure saved to {out}')


if __name__ == '__main__':
    main()
