"""The lsstypes frontend for soft pair counts.

``cucount.types.count2_soft`` is the entry point most external users reach
first, so it has to satisfy two things at once: it must return the same
``Count2`` / ``Count2Poles`` objects as the hard ``cucount.types.count2``, and
it must remove the autocorrelation's self pairs with the *soft* profile the
kernel actually deposited -- a self pair sits at ``s = 0`` and spreads over the
separation bins instead of landing in one of them.

These tests therefore compare against the raw frontend (``count2_soft``) plus
an independent numpy self-pair profile, and one case against a pair-by-pair
reference, so the wrapper is never checked against itself.
"""

import numpy as np
import pytest

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SplitAttrs,
                           cudalib, soft as nsoft)
from cucount.types import count2, count2_soft
from lsstypes import Count2Poles

needs_cuda = pytest.mark.skipif(
    cudalib is None or cudalib.device_count() == 0,
    reason='CUDA extension not built, or no GPU visible')

BOX = 300.0
N = 64
ELLS = [0, 2, 4]
CUTOFF = 1e-13


def _catalogue(n=N, seed=42, box=BOX):
    rng = np.random.default_rng(seed)
    return Particles(rng.uniform(0.0, box, (n, 3)), rng.uniform(0.5, 1.5, n))


def _w(p):
    """The single individual weight column of a catalogue."""
    weights = p.get('weights')
    assert len(weights) == 1
    return np.asarray(weights[0])


def _mattrs(p1, p2, battrs, box=BOX):
    return MeshAttrs(p1, p2, battrs=battrs, periodic=True, boxsize=box)


def _counts(observable):
    """Raw counts of one Count2, or the stacked counts of a Count2Poles."""
    if isinstance(observable, Count2Poles):
        return np.stack([np.asarray(pole.values('counts')) for pole in observable], axis=-1)
    return np.asarray(observable.values('counts'))


def _self_profile(edges, ells=None, soft_mu_sigma=None):
    """What a single self pair (s = 0, mu = 0) deposits on the grid.

    The kernel drops every pair outside ``[edges[0], edges[-1])`` *before*
    softening, so a self pair only deposits something when the grid starts at
    ``s = 0``; otherwise the soft profile must not be subtracted.
    """
    if edges[0] <= 0. < edges[-1]:
        profile = nsoft.soft_weights(0., edges)[0]
    else:
        profile = np.zeros(edges.size - 1)
    if ells is None:
        return profile
    moments = nsoft.soft_mu_weights(np.array([0.]), np.asarray(ells), soft_mu_sigma)[0]
    return np.outer(profile, moments)


@needs_cuda
@pytest.mark.parametrize('s_min', [0.0, 0.5])
def test_isotropic_counts_match_the_raw_frontend(s_min):
    """The wrapper only adds lsstypes packing and the self-pair removal.

    ``s_min = 0`` puts the self pairs inside the first bin; ``s_min = 0.5``
    leaves them outside the counted range, so the subtraction must vanish.
    """
    edges = np.linspace(s_min, 50., 16)
    p = _catalogue()
    battrs = BinAttrs(s=edges)
    DD = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs))['weight']

    raw = np.asarray(nsoft.count2_soft(p, p, battrs=battrs,
                                      mattrs=_mattrs(p, p, battrs))['weight'])
    sumw2 = (_w(p)**2).sum()
    want = raw - sumw2 * _self_profile(edges)

    assert type(DD).__name__ == 'Count2'
    assert np.shape(_counts(DD)) == (edges.size - 1,)
    np.testing.assert_allclose(_counts(DD), want, rtol=0., atol=0.)

    # The normalization is the hard one: (sum w)^2 - sum w^2 for an autocorrelation.
    np.testing.assert_allclose(np.ravel(DD.norm)[0],
                               _w(p).sum() ** 2 - sumw2, rtol=1e-13)
    # An arithmetic grid resolves sigma to half a bin, so the kernel reports an
    # absolute width ('width' would mean sigma = alpha * per-bin width).
    assert DD.attrs['soft_sigma'] == pytest.approx(np.diff(edges).mean() / 2)
    assert DD.attrs['soft_cutoff'] == pytest.approx(CUTOFF)
    assert DD.attrs['soft_sigma_kind'] == 'absolute'


@needs_cuda
def test_poles_match_the_raw_frontend():
    """With a pole axis the wrapper must subtract the soft mu profile too."""
    edges = np.linspace(0., 50., 16)
    p = _catalogue()
    battrs = BinAttrs(s=edges, pole=(ELLS, 'z'))
    DD = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs),
                     soft_mu_sigma=0.05)['weight']

    raw = np.asarray(nsoft.count2_soft(p, p, battrs=battrs, soft_mu_sigma=0.05,
                                      mattrs=_mattrs(p, p, battrs))['weight'])
    sumw2 = (_w(p)**2).sum()
    want = raw - sumw2 * _self_profile(edges, ELLS, soft_mu_sigma=0.05)

    assert type(DD).__name__ == 'Count2Poles'
    assert [pole.ell for pole in DD] == ELLS
    np.testing.assert_allclose(_counts(DD), want, rtol=0., atol=0.)
    assert list(np.atleast_1d(DD.get(0).attrs['soft_multipoles'])) == ELLS

    # l = 0 is the isotropic soft count, as in the kernel.
    iso = count2_soft(p, battrs=BinAttrs(s=edges), mattrs=_mattrs(p, p, battrs))['weight']
    np.testing.assert_allclose(_counts(DD)[:, 0], _counts(iso), rtol=1e-13, atol=1e-15)


@needs_cuda
def test_cross_correlation_removes_no_self_pairs():
    """Two catalogues have no diagonal pairs, so the counts are the raw ones."""
    edges = np.linspace(0., 50., 16)
    p1, p2 = _catalogue(seed=3), _catalogue(seed=4)
    battrs = BinAttrs(s=edges)
    DD = count2_soft(p1, p2, battrs=battrs, mattrs=_mattrs(p1, p2, battrs))['weight']

    raw = np.asarray(nsoft.count2_soft(p1, p2, battrs=battrs,
                                      mattrs=_mattrs(p1, p2, battrs))['weight'])
    np.testing.assert_allclose(_counts(DD), raw, rtol=1e-15, atol=0.)
    np.testing.assert_allclose(np.ravel(DD.norm)[0],
                               _w(p1).sum() * _w(p2).sum(), rtol=1e-13)


@needs_cuda
def test_matches_a_pair_by_pair_reference():
    """Independent O(N^2) reference: nothing is shared with the kernel."""
    edges = np.linspace(0., 40., 9)
    p = _catalogue(n=12, seed=11)
    pos, w = np.asarray(p.positions), _w(p)
    battrs = BinAttrs(s=edges, pole=(ELLS, 'z'))
    DD = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs),
                     soft_mu_sigma=0.1)['weight']

    want = np.zeros((edges.size - 1, len(ELLS)))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            d = pos[i] - pos[j]
            d -= BOX * np.round(d / BOX)
            s = np.sqrt(d @ d)
            if not edges[0] <= s < edges[-1]:
                continue
            mu = 0. if s == 0. else d[2] / s
            moments = nsoft.soft_mu_weights(np.array([mu]), ELLS, 0.1)[0]
            want += w[i] * w[j] * np.outer(nsoft.soft_weights(s, edges)[0], moments)
    want -= (w**2).sum() * _self_profile(edges, ELLS, soft_mu_sigma=0.1)

    np.testing.assert_allclose(_counts(DD), want, rtol=1e-11,
                               atol=1e-11 * np.abs(want).max())


@needs_cuda
def test_attrs_survive_an_hdf5_round_trip(tmp_path):
    """The soft configuration travels with the counts, or a file is ambiguous."""
    import lsstypes as types

    edges = np.linspace(0., 50., 16)
    p = _catalogue()
    battrs = BinAttrs(s=edges, pole=(ELLS, 'z'))
    DD = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs))['weight']

    path = tmp_path / 'soft.h5'
    DD.write(path)
    back = types.read(path)
    np.testing.assert_allclose(_counts(back), _counts(DD), rtol=0., atol=0.)
    assert back.get(0).attrs['soft_los'] == 'z'
    assert list(np.atleast_1d(back.get(0).attrs['soft_multipoles'])) == ELLS
    assert float(back.get(0).attrs['soft_sigma']) == pytest.approx(float(DD.get(0).attrs['soft_sigma']))


@needs_cuda
def test_split_requests_are_rejected():
    """The kernel writes one histogram; a jackknife request must not look served."""
    edges = np.linspace(0., 50., 16)
    p = _catalogue()
    p_split = Particles(np.asarray(p.positions), _w(p),
                        splits=np.arange(N) % 2)
    battrs = BinAttrs(s=edges)
    spattrs = SplitAttrs(mode='jackknife', nsplits=2)
    mattrs = _mattrs(p_split, p_split, battrs)

    with pytest.raises(NotImplementedError, match='split'):
        count2_soft(p_split, battrs=battrs, spattrs=spattrs, mattrs=mattrs)
    # The raw frontend rejects it too, rather than silently dropping the splits.
    with pytest.raises(NotImplementedError, match='split'):
        nsoft.count2_soft(p_split, p_split, battrs=battrs, spattrs=spattrs,
                          mattrs=mattrs)


@needs_cuda
def test_jax_particles_dispatch_to_the_jax_frontend():
    """The lsstypes packing is shared by the numpy and JAX frontends."""
    cjax = pytest.importorskip('cucount.jax')
    if not hasattr(cjax, 'count2_soft'):
        pytest.skip('cucount.jax has no soft frontend')
    import jax
    import jax.numpy as jnp
    from cucount.jax import Particles as JParticles

    jax.config.update('jax_enable_x64', True)
    edges = np.linspace(0., 50., 16)
    p = _catalogue()
    battrs = BinAttrs(s=edges, pole=(ELLS, 'z'))
    jp = JParticles(jnp.asarray(np.asarray(p.positions)), jnp.asarray(_w(p)))
    dj = count2_soft(jp, battrs=battrs, mattrs=_mattrs(p, p, battrs))['weight']
    dn = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs))['weight']
    got, want = _counts(dj), _counts(dn)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13 * np.abs(want).max())


@needs_cuda
def test_hard_and_soft_wrappers_agree_in_the_narrow_sigma_limit():
    """sigma -> 0 recovers the hard Count2 on a uniform grid."""
    edges = np.linspace(0.5, 50., 16)  # no self pair inside the counted range
    p = _catalogue()
    battrs = BinAttrs(s=edges)
    hard = count2(p, battrs=battrs, mattrs=_mattrs(p, p, battrs))['weight']
    soft = count2_soft(p, battrs=battrs, mattrs=_mattrs(p, p, battrs),
                       soft_sigma=1e-3)['weight']
    got, want = _counts(soft), _counts(hard)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * np.abs(want).max())
