"""Correctness of the portable CPU backend.

Exercised through the public numpy API, so this covers the whole stack --
Particles packing, BinAttrs/MeshAttrs marshalling and the backend shim -- not
just the kernel. Two independent oracles are used: a numpy O(N^2) brute force,
and the CUDA backend via compare mode.

Requests are made with backend='cpu' explicitly: the default is 'cuda', so
without it these would pass without ever running the code under test.

The few axes the public API does not expose (scatter strategy, single
precision) are tested against cucountlib.cpucount directly at the end.
"""

import itertools

import numpy as np
import pytest
from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           SplitAttrs, WeightAttrs, _cpu, count2)

pytestmark = pytest.mark.skipif(
    not _cpu.available(), reason='CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)')

BOX = 1000.0
SMAX = 100.0
MU = np.linspace(-1.0, 1.0, 9)

EDGES = {
    'lin': np.linspace(1.0, SMAX, 21),
    'log': np.geomspace(1.0, SMAX, 21),
    # Deliberately irregular: only the generic bin policy can serve these.
    'edges': np.sort(np.r_[0.0, 3.0, 7.5, 11.0, 23.0, 24.0, 40.0, 61.5, 88.0, SMAX]),
}

# LOS is meaningless without a mu axis, so only one variant is built for 1D.
MATRIX = [c for c in itertools.product(['lin', 'log', 'edges'], [1, 2],
                                       ['z', 'midpoint'], [False, True])
          if not (c[1] == 1 and c[2] == 'midpoint')]
MATRIX_IDS = [f'{k}-{n}d-{los}-{"per" if p else "nonper"}' for k, n, los, p in MATRIX]


def catalog(seed, n=800):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, BOX, (n, 3)), rng.uniform(0.5, 1.5, n)


def _bin_index(v, edges):
    """Half-open [lo, hi) for every bin, matching the kernel.

    np.histogram would fold the rightmost edge into the last bin instead.
    """
    idx = np.searchsorted(edges, v, side='right') - 1
    return idx, (v >= edges[0]) & (v < edges[-1])


def brute(pos1, w1, pos2, w2, sedges, muedges=None, los='z', periodic=False):
    d = pos2[None, :, :] - pos1[:, None, :]
    if periodic:
        d -= BOX * np.round(d / BOX)
    s = np.sqrt((d * d).sum(-1))
    w = w1[:, None] * w2[None, :]

    si, sok = _bin_index(s, sedges)
    nb = len(sedges) - 1
    if muedges is None:
        return np.bincount(si[sok], weights=w[sok], minlength=nb)

    if los == 'z':
        num, den = d[..., 2], s
    else:  # midpoint: los = p1 + p2
        ell = pos2[None, :, :] + pos1[:, None, :]
        num = (d * ell).sum(-1)
        den = s * np.sqrt((ell * ell).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(den > 0, num / np.where(den > 0, den, 1.0), -2.0)
    mu[s == 0] = 0.0

    mi, mok = _bin_index(mu, muedges)
    nm = len(muedges) - 1
    ok = sok & mok
    return np.bincount(si[ok] * nm + mi[ok], weights=w[ok],
                       minlength=nb * nm).reshape(nb, nm)


def setup(kind, ndim, los, periodic, seeds=(1, 2)):
    """Public-API objects plus the raw arrays the reference needs."""
    pos1, w1 = catalog(seeds[0])
    pos2, w2 = catalog(seeds[1])
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    sedges = EDGES[kind]
    battrs = (BinAttrs(s=sedges, mu=(MU, los)) if ndim == 2
              else BinAttrs(s=sedges))
    # A periodic run needs the box stated; otherwise MeshAttrs derives an
    # extent from the data, which is what non-periodic callers actually do.
    mattrs = (MeshAttrs(*particles, boxsize=BOX, battrs=battrs, periodic=True)
              if periodic else MeshAttrs(*particles, battrs=battrs))
    return particles, battrs, mattrs, (pos1, w1, pos2, w2), sedges


@pytest.mark.parametrize('kind,ndim,los,periodic', MATRIX, ids=MATRIX_IDS)
def test_matches_brute_force(kind, ndim, los, periodic):
    particles, battrs, mattrs, raw, sedges = setup(kind, ndim, los, periodic)
    got = count2(*particles, battrs=battrs, mattrs=mattrs,
                 backend='cpu')['weight']
    want = brute(*raw, sedges, MU if ndim == 2 else None, los, periodic)
    assert np.allclose(got, want, rtol=1e-9, atol=0)


@pytest.mark.parametrize('kind,ndim,los,periodic', MATRIX, ids=MATRIX_IDS)
def test_matches_cuda(kind, ndim, los, periodic):
    """compare mode raises if the two backends disagree."""
    particles, battrs, mattrs, _, _ = setup(kind, ndim, los, periodic)
    count2(*particles, battrs=battrs, mattrs=mattrs, backend='compare')


def test_autocorrelation_self_pairs():
    """Every ordered pair is visited, so s=0 self-pairs land in the first bin."""
    pos, w = catalog(3, n=500)
    p = Particles(pos, w)
    sedges = np.linspace(0.0, SMAX, 11)
    battrs = BinAttrs(s=sedges)
    mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True)
    got = count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
    assert np.allclose(got, brute(pos, w, pos, w, sedges, periodic=True), rtol=1e-9)
    assert got[0] >= (w * w).sum() * (1 - 1e-12)


@pytest.mark.parametrize('los', ['z', 'midpoint'])
def test_self_pairs_bin_at_mu_zero(los):
    """Coincident points take mu = 0 in (s, mu) counts, on both backends."""
    pos, w = catalog(4, n=500)
    p = Particles(pos, w)
    sedges = np.linspace(0.0, SMAX, 11)
    battrs = BinAttrs(s=sedges, mu=(MU, los))
    mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True)
    got = count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
    want = brute(pos, w, pos, w, sedges, MU, los, periodic=True)
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    imu0 = np.searchsorted(MU, 0.0, side='right') - 1
    assert got[0, imu0] >= (w * w).sum() * (1 - 1e-12)
    # compare mode raises if the CUDA backend disagrees on the convention
    count2(p, p, battrs=battrs, mattrs=mattrs, backend='compare')


def test_smax_much_smaller_than_box():
    """boxsize/smax = 1000: the mesh must stay O(n) cells, not (box/smax)^3."""
    pos1, w1 = catalog(5)
    rng = np.random.default_rng(6)
    # Pair each point with a nearby partner so counts are nonzero at s < 1.
    pos2 = (pos1 + rng.uniform(-0.4, 0.4, pos1.shape)) % BOX
    w2 = rng.uniform(0.5, 1.5, len(pos2))
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    sedges = np.linspace(0.0, 1.0, 6)
    battrs = BinAttrs(s=sedges)
    mattrs = MeshAttrs(*particles, boxsize=BOX, battrs=battrs, periodic=True)
    got = count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
    want = brute(pos1, w1, pos2, w2, sedges, periodic=True)
    assert want.sum() > 0
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    count2(*particles, battrs=battrs, mattrs=mattrs, backend='compare')


def test_zero_bins_is_empty_result():
    """A 1-edge array requests zero bins; both backends serve it as empty."""
    pos, w = catalog(7, n=200)
    p = Particles(pos, w)
    for battrs in (BinAttrs(s=np.array([5.0])),
                   BinAttrs(s=EDGES['lin'], mu=(np.array([0.5]), 'z'))):
        mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True)
        got = count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
        assert got.shape == battrs.shape and got.size == 0
        count2(p, p, battrs=battrs, mattrs=mattrs, backend='compare')


@pytest.mark.parametrize('los', ['z', 'midpoint'])
def test_single_mu_bin(los):
    """A 2-edge mu array is one linear bin, served rather than declined."""
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    mu1 = np.array([-1.0, 1.0])
    battrs = BinAttrs(s=EDGES['lin'], mu=(mu1, los))
    mattrs = MeshAttrs(*particles, boxsize=BOX, battrs=battrs, periodic=True)
    got = count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
    want = brute(pos1, w1, pos2, w2, EDGES['lin'], mu1, los, periodic=True)
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    count2(*particles, battrs=battrs, mattrs=mattrs, backend='compare')


def test_thread_count_invariance(monkeypatch):
    particles, battrs, mattrs, _, _ = setup('lin', 2, 'z', True)
    ref = None
    for nt in (1, 2, 8, 16):
        monkeypatch.setenv('CUCOUNT_CPU_NTHREADS', str(nt))
        got = count2(*particles, battrs=battrs, mattrs=mattrs,
                     backend='cpu')['weight']
        if ref is None:
            ref = got
        else:
            # Reduction order varies with thread count, so this is a
            # floating-point agreement check, not bit-for-bit reproducibility.
            assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_simd_targets_agree():
    """Every compiled ISA must produce the same answer."""
    cpucount = _cpu.cpucount
    particles, battrs, mattrs, _, _ = setup('log', 2, 'midpoint', True)

    reached, results = [], []
    try:
        for name in cpucount.available_targets():
            if cpucount.set_target(name) is None:
                continue
            actual = cpucount.current_target()
            if actual in reached:
                continue  # not compiled in; dispatch fell back to one already seen
            reached.append(actual)
            results.append(count2(*particles, battrs=battrs, mattrs=mattrs,
                                  backend='cpu')['weight'])
    finally:
        cpucount.set_target('')

    assert len(reached) >= 2, f'need >=2 targets to compare, got {reached}'
    for name, res in zip(reached[1:], results[1:]):
        assert np.allclose(res, results[0], rtol=1e-12, atol=0), \
            f'{name} disagrees with {reached[0]}'


@pytest.mark.parametrize('kind', ['lin', 'log', 'edges'])
def test_bin_kind_inference(kind):
    """The shim must not claim a fast policy for edges that are not regular."""
    assert _cpu._bin_kind(EDGES[kind]) == kind
    # A linear array perturbed past the tolerance has to fall back to generic.
    perturbed = EDGES['lin'].copy()
    perturbed[5] += 1e-3
    assert _cpu._bin_kind(perturbed) == 'edges'


def test_unsupported_is_declined_by_name():
    pos, w = catalog(8, n=200)
    p = Particles(pos, w)
    battrs = BinAttrs(theta=np.linspace(0.1, 5.0, 11))
    mattrs = MeshAttrs(p, p, battrs=battrs)
    with pytest.raises(NotImplementedError, match='theta'):
        count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')
    # The same request is served when routed to the CUDA backend.
    assert 'weight' in count2(p, p, battrs=battrs, mattrs=mattrs, backend='cuda')


UNSUPPORTED = ['rp-pi binning', 'los x', 'los firstpoint', 'non-linear mu',
               'angular mesh', 'theta selection', 'jackknife splits',
               'spin weights', 'bitwise weights', 'negative weights',
               'angular weights']


def _unsupported_request(feature, n=200):
    """A fully-formed count2 request using one feature the CPU backend lacks."""
    rng = np.random.default_rng(10)
    pos1, w1 = catalog(11, n)
    pos2, w2 = catalog(12, n)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    kw = dict(battrs=BinAttrs(s=EDGES['lin']))

    if feature == 'rp-pi binning':
        kw['battrs'] = BinAttrs(rp=(np.linspace(1.0, 50.0, 11), 'z'),
                                pi=(np.linspace(1.0, 50.0, 11), 'z'))
        match = 'rp'
    elif feature in ('los x', 'los firstpoint'):
        los = feature.split()[1]
        kw['battrs'] = BinAttrs(s=EDGES['lin'], mu=(MU, los))
        match = 'line of sight'
    elif feature == 'non-linear mu':
        kw['battrs'] = BinAttrs(s=EDGES['lin'],
                                mu=(np.array([-1.0, -0.5, 0.8, 1.0]), 'z'))
        match = 'non-linear mu'
    elif feature == 'angular mesh':
        sattrs = SelectionAttrs(theta=(0.0, 1.0))
        kw.update(sattrs=sattrs, mattrs=MeshAttrs(*particles, battrs=kw['battrs'],
                                                  sattrs=sattrs))
        match = 'angular mesh'
    elif feature == 'theta selection':
        # A cartesian mesh, so the decline must name the selection itself.
        kw['sattrs'] = SelectionAttrs(theta=(0.0, 1.0))
        match = 'selection'
    elif feature == 'jackknife splits':
        particles = (Particles(pos1, w1, splits=rng.integers(0, 4, n)),
                     Particles(pos2, w2, splits=rng.integers(0, 4, n)))
        kw['spattrs'] = SplitAttrs(mode='jackknife', nsplits=4)
        match = 'split'
    elif feature == 'spin weights':
        particles = (Particles(pos1, w1, spin_values=rng.uniform(-1, 1, (n, 2))),
                     Particles(pos2, w2, spin_values=rng.uniform(-1, 1, (n, 2))))
        match = 'spin'
    elif feature in ('bitwise weights', 'negative weights'):
        bits = [rng.integers(0, 0xffffffff, n, dtype=np.uint64) for _ in range(2)]
        # A float array after a bitwise one is read as a negative weight.
        extra = [w1] if feature == 'negative weights' else []
        particles = (Particles(pos1, [w1, bits[0]] + extra),
                     Particles(pos2, [w2, bits[1]] + extra))
        kw['wattrs'] = WeightAttrs(bitwise=dict(weights=[bits[0]]))
        match = feature.split()[0] + '_weight'
    elif feature == 'angular weights':
        sep = np.linspace(0.0, 5.0, 41)
        kw['wattrs'] = WeightAttrs(angular=dict(sep=sep, weight=np.ones(sep.size)))
        match = 'angular weights'

    kw.setdefault('mattrs', MeshAttrs(*particles, battrs=kw['battrs']))
    return particles, kw, match


@pytest.mark.parametrize('feature', UNSUPPORTED)
def test_unsupported_modes_are_declined(feature):
    """Every feature the backend lacks must raise, naming the feature."""
    particles, kw, match = _unsupported_request(feature)
    with pytest.raises(NotImplementedError, match=match):
        count2(*particles, backend='cpu', **kw)


def test_unbuilt_backend_is_declined(monkeypatch):
    """Requesting cpu without the extension built must name the build flag."""
    pos, w = catalog(13, n=100)
    p = Particles(pos, w)
    battrs = BinAttrs(s=EDGES['lin'])
    mattrs = MeshAttrs(p, p, battrs=battrs)
    monkeypatch.setattr(_cpu, 'cpucount', None)
    with pytest.raises(NotImplementedError, match='not built'):
        count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')


def test_rejects_unknown_backend():
    pos, w = catalog(9, n=200)
    p = Particles(pos, w)
    battrs = BinAttrs(s=EDGES['lin'])
    mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True)
    with pytest.raises(ValueError, match='backend must be one of'):
        count2(p, p, battrs=battrs, mattrs=mattrs, backend='gpu')


def test_backend_env_var(monkeypatch):
    """CUCOUNT_BACKEND selects the backend when no keyword is given."""
    particles, battrs, mattrs, raw, sedges = setup('lin', 1, 'z', True)
    monkeypatch.setenv('CUCOUNT_BACKEND', 'cpu')
    got = count2(*particles, battrs=battrs, mattrs=mattrs)['weight']
    assert np.allclose(got, brute(*raw, sedges, periodic=True), rtol=1e-9)
    monkeypatch.setenv('CUCOUNT_BACKEND', 'compare')
    count2(*particles, battrs=battrs, mattrs=mattrs)


# --- axes the public API does not expose -----------------------------------

@pytest.mark.parametrize('kind,ndim', [('lin', 1), ('log', 1), ('edges', 1),
                                       ('lin', 2), ('log', 2)])
def test_scatter_strategies_agree(kind, ndim):
    """Both histogram strategies must give identical answers."""
    cpucount = _cpu.cpucount
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    sedges = EDGES[kind]
    kw = dict(muedges=MU if ndim == 2 else None, boxsize=(BOX,) * 3,
              bin=kind, periodic=True, nthreads=4)
    a = cpucount.count2(pos1, w1, pos2, w2, sedges, scatter='scalar', **kw)
    b = cpucount.count2(pos1, w1, pos2, w2, sedges, scatter='binmajor', **kw)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


@pytest.mark.parametrize('ndim', [1, 2])
def test_float32_close_to_double(ndim):
    cpucount = _cpu.cpucount
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    sedges = EDGES['lin']
    kw = dict(muedges=MU if ndim == 2 else None, boxsize=(BOX,) * 3,
              periodic=True, nthreads=4)
    f64 = cpucount.count2(pos1, w1, pos2, w2, sedges, float32=False, **kw)
    f32 = cpucount.count2(pos1, w1, pos2, w2, sedges, float32=True, **kw)
    # Single precision moves pairs across bin edges, so compare loosely and
    # scale the floor to the typical bin population.
    assert np.allclose(f32, f64, rtol=2e-2, atol=1e-2 * f64.mean())
