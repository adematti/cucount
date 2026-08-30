"""Correctness of the portable CPU backend.

Exercised through the public numpy API, so this covers the whole stack --
Particles packing, BinAttrs/MeshAttrs marshalling and the backend shim -- not
just the kernel. Two independent oracles are used: a numpy O(N^2) brute force,
and the CUDA backend via compare mode.

Requests are made with backend='cpu' explicitly: the default is 'cuda', so
without it these would pass without ever running the code under test.

The few axes the public API does not expose (scatter strategy, single
precision) are tested against cucountlib.cpu directly at the end.
"""

import itertools

import numpy as np
import pytest
from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           SplitAttrs, WeightAttrs, _cpu, count2)

pytestmark = pytest.mark.skipif(
    not _cpu.available(), reason='CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)')


def _cuda_available():
    try:
        import cucountlib.cuda  # noqa: F401
        return True
    except ImportError:
        return False


# CUDA-oracle checks are skipped on a CUDA-less build (-DCUCOUNT_BUILD_CUDA=OFF)
# so the suite still validates the CPU backend against the numpy brute force.
CUDA = _cuda_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason='CUDA extension not built')

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
                                       ['z', 'midpoint', 'firstpoint',
                                        'endpoint', 'x'], [False, True])
          if not (c[1] == 1 and c[2] != 'z')]
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
    elif los == 'x':
        num, den = d[..., 0], s
    elif los == 'firstpoint':
        r1 = pos1 / np.linalg.norm(pos1, axis=1, keepdims=True)
        num = (d * r1[:, None, :]).sum(-1)
        den = s
    elif los == 'endpoint':
        r2 = pos2 / np.linalg.norm(pos2, axis=1, keepdims=True)
        num = (d * r2[None, :, :]).sum(-1)
        den = s
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


@needs_cuda
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
    if CUDA:
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
    if CUDA:
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
        if CUDA:
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
    if CUDA:
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


def test_irregular_edges_still_served():
    """A near-linear grid perturbed past tolerance must take the generic
    policy (the classification now happens in the C++ lowering)."""
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    sedges = EDGES['lin'].copy()
    sedges[5] += 1e-3
    battrs = BinAttrs(s=sedges)
    mattrs = MeshAttrs(*particles, boxsize=BOX, battrs=battrs, periodic=True)
    got = count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']
    want = brute(pos1, w1, pos2, w2, sedges, periodic=True)
    assert np.allclose(got, want, rtol=1e-9, atol=0)


def test_unsupported_is_declined_by_name():
    pos, w = catalog(8, n=200)
    p = Particles(pos, w)
    battrs = BinAttrs(theta=np.linspace(0.1, 5.0, 11))
    mattrs = MeshAttrs(p, p, battrs=battrs)
    with pytest.raises(NotImplementedError, match='theta'):
        count2(p, p, battrs=battrs, mattrs=mattrs, backend='cpu')
    # The same request is served when routed to the CUDA backend.
    if CUDA:
        assert 'weight' in count2(p, p, battrs=battrs, mattrs=mattrs, backend='cuda')


UNSUPPORTED = ['rp-pi binning', 'non-linear mu',
               'angular mesh', 'jackknife splits',
               'spin components', 'angular weights']


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
    elif feature == 'non-linear mu':
        kw['battrs'] = BinAttrs(s=EDGES['lin'],
                                mu=(np.array([-1.0, -0.5, 0.8, 1.0]), 'z'))
        match = 'non-linear mu'
    elif feature == 'angular mesh':
        sattrs = SelectionAttrs(theta=(0.0, 1.0))
        kw.update(sattrs=sattrs, mattrs=MeshAttrs(*particles, battrs=kw['battrs'],
                                                  sattrs=sattrs))
        match = 'angular mesh'
    elif feature == 'jackknife splits':
        particles = (Particles(pos1, w1, splits=rng.integers(0, 4, n)),
                     Particles(pos2, w2, splits=rng.integers(0, 4, n)))
        kw['spattrs'] = SplitAttrs(mode='jackknife', nsplits=4)
        match = 'split'
    elif feature == 'spin components':
        # Spin itself is served now; only a non-2 component count is declined.
        particles = (Particles(pos1, w1, spin_values=rng.uniform(-1, 1, n)),
                     Particles(pos2, w2))
        match = 'spin'
    elif feature == 'angular weights':
        # 1D angular weights are served now; only N-dimensional tables decline.
        sep = np.linspace(0.0, 5.0, 11)
        kw['wattrs'] = WeightAttrs(angular=dict(sep=[sep, sep],
                                                weight=np.ones((sep.size, sep.size))))
        match = 'N-dimensional angular'

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
    if CUDA:
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
    a = cpucount.count2_arrays(pos1, w1, pos2, w2, sedges, scatter='scalar', **kw)
    b = cpucount.count2_arrays(pos1, w1, pos2, w2, sedges, scatter='binmajor', **kw)
    # Non-triviality first: two dead kernels agree on all-zeros, which is how a
    # stale build once passed this test.
    assert a.sum() > 0
    assert np.allclose(a, b, rtol=1e-12, atol=0)


@pytest.mark.parametrize('ndim', [1, 2])
def test_float32_close_to_double(ndim):
    cpucount = _cpu.cpucount
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    sedges = EDGES['lin']
    kw = dict(muedges=MU if ndim == 2 else None, boxsize=(BOX,) * 3,
              periodic=True, nthreads=4)
    f64 = cpucount.count2_arrays(pos1, w1, pos2, w2, sedges, float32=False, **kw)
    f32 = cpucount.count2_arrays(pos1, w1, pos2, w2, sedges, float32=True, **kw)
    assert f64.sum() > 0  # see test_scatter_strategies_agree
    # Single precision moves pairs across bin edges, so compare loosely and
    # scale the floor to the typical bin population.
    assert np.allclose(f32, f64, rtol=2e-2, atol=1e-2 * f64.mean())


# ---------------------------------------------------------------------------
# Tuning plumbing: flat dict addressed to the selected backend, unknown keys
# rejected by name, nested form for compare mode.
# ---------------------------------------------------------------------------

def test_tuning_unknown_keys_rejected():
    particles, battrs, mattrs, _, _ = setup('lin', 1, 'z', True)
    kw = dict(battrs=battrs, mattrs=mattrs)
    with pytest.raises(ValueError, match='bogus'):
        count2(*particles, backend='cpu', tuning={'bogus': 1}, **kw)
    # The CUDA key check runs before any kernel launch, so it needs no GPU.
    with pytest.raises(ValueError, match='isa'):
        count2(*particles, backend='cuda', tuning={'isa': 'AVX2'}, **kw)
    # compare mode runs both backends, so a flat dict is ambiguous there.
    with pytest.raises(ValueError, match='nested'):
        count2(*particles, backend='compare', tuning={'nthreads': 2}, **kw)


def test_tuning_does_not_change_results():
    particles, battrs, mattrs, _, _ = setup('lin', 2, 'midpoint', True)
    kw = dict(battrs=battrs, mattrs=mattrs, backend='cpu')
    want = count2(*particles, **kw)['weight']
    got = count2(*particles, tuning={'nthreads': 2, 'scatter': 'binmajor'}, **kw)['weight']
    assert np.allclose(got, want, rtol=1e-12, atol=0)


def test_tuning_isa_pins_and_restores():
    cpucount = _cpu.cpucount
    particles, battrs, mattrs, _, _ = setup('lin', 1, 'z', True)
    kw = dict(battrs=battrs, mattrs=mattrs, backend='cpu')
    before = cpucount.current_target()
    want = count2(*particles, **kw)['weight']
    # The narrowest attainable target is always compiled in, so it is a safe pin.
    isa = cpucount.available_targets()[-1]
    got = count2(*particles, tuning={'isa': isa}, **kw)['weight']
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    # Automatic selection must be restored after the call, error or not.
    assert cpucount.current_target() == before
    with pytest.raises(ValueError, match='unknown or unavailable'):
        count2(*particles, tuning={'isa': 'NOT_AN_ISA'}, **kw)
    assert cpucount.current_target() == before


def test_nthreads_keyword_deprecated():
    particles, battrs, mattrs, _, _ = setup('lin', 1, 'z', True)
    with pytest.warns(DeprecationWarning, match='tuning'):
        count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu', nthreads=2)


# ---------------------------------------------------------------------------
# Spin (galaxy-shear / shear-shear): the SIMD cull is unchanged and surviving
# lanes take the scalar projection shared with CUDA (include/pair_math.h).
# ---------------------------------------------------------------------------

def _unit_catalog(seed, n=500):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    w = rng.uniform(0.5, 1.5, n)
    e = rng.normal(0., 0.2, (n, 2))
    return p, w, e


def _spin_projection(r1, r2, e0, e1, spin):
    """Vectorized copy of compute_spin_projection_cartesian in pair_math.h.

    r1 (n1, 3), r2 (n2, 3): unit vectors. e0, e1: components pre-shaped to
    broadcast against (n1, n2) from whichever side carries them.
    """
    east = np.cross(np.array([0., 0., 1.]), r1)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(r1, east)
    dot12 = r1 @ r2.T
    p = r2[None, :, :] - dot12[..., None] * r1[:, None, :]
    pe = np.einsum('ijk,ik->ij', p, east)
    pn = np.einsum('ijk,ik->ij', p, north)
    phi = np.arctan2(pe, pn)
    c, s = np.cos(spin * phi), np.sin(spin * phi)
    splus = -(e0 * c + e1 * s)
    scross = e0 * s - e1 * c
    return splus, scross


def _brute_multi(pos1, pos2, wmats, sedges, muedges=None, los='z'):
    """Bin several (n1, n2) pair-weight matrices at once (non-periodic)."""
    d = pos2[None, :, :] - pos1[:, None, :]
    s = np.sqrt((d * d).sum(-1))
    si, ok = _bin_index(s, sedges)
    nb = len(sedges) - 1
    shape, idx = (nb,), si
    if muedges is not None:
        if los == 'z':
            num, den = d[..., 2], s
        else:
            ell = pos2[None, :, :] + pos1[:, None, :]
            num = (d * ell).sum(-1)
            den = s * np.sqrt((ell * ell).sum(-1))
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = np.where(den > 0, num / np.where(den > 0, den, 1.0), -2.0)
        mu[s == 0] = 0.0
        mi, mok = _bin_index(mu, muedges)
        nm = len(muedges) - 1
        ok = ok & mok
        shape, idx = (nb, nm), si * nm + mi
    return [np.bincount(idx[ok], weights=w[ok],
                        minlength=nb * (shape[1] if muedges is not None else 1)
                        ).reshape(shape) for w in wmats]


def _spin_setup(mode, ndim, seeds=(1, 2)):
    pos1, w1, e1 = _unit_catalog(seeds[0])
    pos2, w2, e2 = _unit_catalog(seeds[1])
    s1, s2 = mode[0] == 's', mode[1] == 's'
    particles = (Particles(pos1, w1, spin_values=e1 if s1 else None),
                 Particles(pos2, w2, spin_values=e2 if s2 else None))
    wattrs = WeightAttrs(spin=(2 if s1 else 0, 2 if s2 else 0))
    sedges = np.linspace(0.05, 1.5, 11)
    muedges = np.linspace(-1., 1., 7)
    battrs = (BinAttrs(s=sedges, mu=(muedges, 'z')) if ndim == 2
              else BinAttrs(s=sedges))
    mattrs = MeshAttrs(*particles, battrs=battrs)

    # Reference channels, CUDA add_weight2 formulas (cross*plus included).
    w = w1[:, None] * w2[None, :]
    if s1:
        p1, x1 = _spin_projection(pos1, pos2, e1[:, None, 0], e1[:, None, 1], 2)
    if s2:
        p2, x2 = _spin_projection(pos1, pos2, e2[None, :, 0], e2[None, :, 1], 2)
    if s1 and s2:
        names = ['weight_plus_plus', 'weight_plus_cross', 'weight_cross_cross']
        wmats = [w * p1 * p2, w * x1 * p2, w * x1 * x2]
    elif s1:
        names, wmats = ['weight_plus', 'weight_cross'], [w * p1, w * x1]
    else:
        names, wmats = ['weight_plus', 'weight_cross'], [w * p2, w * x2]
    want = _brute_multi(pos1, pos2, wmats, sedges,
                        muedges if ndim == 2 else None)
    kw = dict(battrs=battrs, mattrs=mattrs, wattrs=wattrs)
    return particles, kw, names, want


@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('mode', ['gs', 'sg', 'ss'])
def test_spin_matches_brute_force(mode, ndim):
    particles, kw, names, want = _spin_setup(mode, ndim)
    got = count2(*particles, backend='cpu', **kw)
    assert sorted(got) == sorted(names)
    for name, ref in zip(names, want):
        scale = np.abs(ref).max()
        assert np.allclose(got[name], ref, rtol=1e-9, atol=1e-12 * scale), name


@needs_cuda
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('mode', ['gs', 'ss'])
def test_spin_matches_cuda(mode, ndim):
    particles, kw, names, _ = _spin_setup(mode, ndim)
    cpu = count2(*particles, backend='cpu', **kw)
    cuda = count2(*particles, backend='cuda', **kw)
    for name in names:
        # Both sides normalize in full precision, but CUDA's sin/cos/atan2
        # intrinsics differ from libm by ulps, and near-cancelling bins
        # amplify that; hence the atol scaled to the channel.
        scale = np.abs(cuda[name]).max()
        np.testing.assert_allclose(cpu[name], cuda[name],
                                   rtol=1e-7, atol=1e-10 * scale,
                                   err_msg=name)


# ---------------------------------------------------------------------------
# Bitwise (PIP) and negative weights: the pair weight stops factorizing as
# w1 * w2, so surviving lanes take the scalar tail with the shared
# pair_bitwise_weight (include/pair_math.h).
# ---------------------------------------------------------------------------

def _pip_reference(bits1, bits2, bw):
    """(n1, n2) PIP pair weights, mirroring pair_bitwise_weight exactly."""
    from cucount.numpy import popcount
    nb = bw.noffset + sum(popcount(b1[:, None] & b2[None, :])
                          for b1, b2 in zip(bits1, bits2))
    w = bw.nrealizations / np.where(nb == 0, 1, nb)
    if bw.p_correction_nbits is not None:
        c1 = sum(popcount(b) for b in bits1)
        c2 = sum(popcount(b) for b in bits2)
        w = w / np.asarray(bw.p_correction_nbits)[c1[:, None], c2[None, :]]
    return np.where(nb == 0, bw.default_value, w)


def _pip_setup(mode, ndim, seeds=(1, 2), n=400):
    rng = np.random.default_rng(20)
    pos1, w1 = catalog(seeds[0], n)
    pos2, w2 = catalog(seeds[1], n)
    bits1 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64) for _ in range(2)]
    bits2 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64) for _ in range(2)]
    # Small negative weights keep the bins clear of cancellation.
    nw1, nw2 = rng.uniform(0., 0.1, n), rng.uniform(0., 0.1, n)

    extra = [nw1, nw2] if mode == 'pip-negative' else [None, None]
    # A float array after the bitwise ones is read as a negative weight.
    particles = (Particles(pos1, [w1] + bits1 + ([extra[0]] if extra[0] is not None else [])),
                 Particles(pos2, [w2] + bits2 + ([extra[1]] if extra[1] is not None else [])))
    correction = mode != 'pip-nocorrection'
    wattrs = WeightAttrs(bitwise=dict(weights=bits1, p_correction_nbits=correction))

    sedges = EDGES['lin']
    muedges = np.linspace(-1., 1., 5)
    battrs = (BinAttrs(s=sedges, mu=(muedges, 'z')) if ndim == 2
              else BinAttrs(s=sedges))
    mattrs = MeshAttrs(*particles, battrs=battrs)

    w = w1[:, None] * w2[None, :]
    w = w * _pip_reference(bits1, bits2, wattrs.bitwise)
    if mode == 'pip-negative':
        w = w - nw1[:, None] * nw2[None, :]
    want = _brute_multi(pos1, pos2, [w], sedges,
                        muedges if ndim == 2 else None)[0]
    kw = dict(battrs=battrs, mattrs=mattrs, wattrs=wattrs)
    return particles, kw, want


@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('mode', ['pip', 'pip-nocorrection', 'pip-negative'])
def test_bitwise_matches_brute_force(mode, ndim):
    particles, kw, want = _pip_setup(mode, ndim)
    got = count2(*particles, backend='cpu', **kw)['weight']
    scale = np.abs(want).max()
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


@needs_cuda
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('mode', ['pip', 'pip-negative'])
def test_bitwise_matches_cuda(mode, ndim):
    """Popcount math is exact on both sides, so compare mode's tight rtol holds."""
    particles, kw, _ = _pip_setup(mode, ndim)
    count2(*particles, backend='compare', **kw)


# ---------------------------------------------------------------------------
# Angular (PIP) upweights: 1D tables, applied per surviving lane via the
# lookup_angular_weight shared with CUDA (include/pair_math.h).
# ---------------------------------------------------------------------------

def _angular_reference(ct, angular):
    """(n1, n2) angular weights, mirroring the shared 1D lookup: linear
    interpolation over sep points, piecewise-constant over edges, 1 outside."""
    state = angular._to_c()  # axes converted to ascending cos(theta)
    wt = state['weight']
    w = np.ones_like(ct)
    if angular.tabulation == 'sep':
        sep = state['sep'][0]
        inside = (ct >= sep[0]) & (ct <= sep[-1])
        w[inside] = np.interp(ct[inside], sep, wt)
    else:
        e = state['edges'][0]
        inside = (ct >= e[0]) & (ct < e[-1])
        idx = np.clip(np.searchsorted(e, ct, side='right') - 1, 0, wt.size - 1)
        w[inside] = wt[idx[inside]]
    return w


@pytest.mark.parametrize('tabulation', ['sep', 'edges'])
def test_angular_matches_brute_force(tabulation):
    pos1, w1, _ = _unit_catalog(1)
    pos2, w2, _ = _unit_catalog(2)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    # Tabulate only to 60 deg so pairs beyond it exercise the weight-1 branch.
    theta = np.linspace(0., 60., 41)
    table = 1. + 0.5 * np.exp(-theta / 20.)
    if tabulation == 'sep':
        wattrs = WeightAttrs(angular=dict(sep=theta, weight=table))
    else:
        wattrs = WeightAttrs(angular=dict(edges=theta, weight=table[:-1]))
    sedges = np.linspace(0.05, 1.5, 11)
    battrs = BinAttrs(s=sedges)
    mattrs = MeshAttrs(*particles, battrs=battrs)

    ct = pos1 @ pos2.T
    w = w1[:, None] * w2[None, :] * _angular_reference(ct, wattrs.angular)
    want = _brute_multi(pos1, pos2, [w], sedges)[0]

    got = count2(*particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs,
                 backend='cpu')['weight']
    scale = np.abs(want).max()
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


@needs_cuda
@pytest.mark.parametrize('tabulation', ['sep', 'edges'])
def test_angular_matches_cuda(tabulation):
    """Both backends run the same shared lookup, so compare mode's rtol holds."""
    pos1, w1, _ = _unit_catalog(1)
    pos2, w2, _ = _unit_catalog(2)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    theta = np.linspace(0., 60., 41)
    table = 1. + 0.5 * np.exp(-theta / 20.)
    kw = dict(sep=theta, weight=table) if tabulation == 'sep' else \
         dict(edges=theta, weight=table[:-1])
    wattrs = WeightAttrs(angular=kw)
    battrs = BinAttrs(s=np.linspace(0.05, 1.5, 11))
    mattrs = MeshAttrs(*particles, battrs=battrs)
    count2(*particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs,
           backend='compare')


@needs_cuda
def test_all_weight_schemes_combined_match_cuda():
    """individual + bitwise + angular + negative together, in the CUDA order."""
    rng = np.random.default_rng(30)
    n = 400
    pos1, w1 = catalog(1, n)
    pos2, w2 = catalog(2, n)
    bits1 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64) for _ in range(2)]
    bits2 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64) for _ in range(2)]
    nw1, nw2 = rng.uniform(0., 0.1, n), rng.uniform(0., 0.1, n)
    particles = (Particles(pos1, [w1] + bits1 + [nw1]),
                 Particles(pos2, [w2] + bits2 + [nw2]))
    theta = np.linspace(0., 30., 16)
    wattrs = WeightAttrs(bitwise=dict(weights=bits1),
                         angular=dict(sep=theta, weight=1. + theta / 60.))
    battrs = BinAttrs(s=EDGES['lin'])
    mattrs = MeshAttrs(*particles, battrs=battrs)
    count2(*particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs,
           backend='compare')


# ---------------------------------------------------------------------------
# Multipole (pole) binning: mu is computed but not binned, and each pair adds
# (2 ell + 1) P_ell(mu) into the nells bins that follow its s bin, using the
# set_legendre shared with CUDA (include/pair_math.h).
# ---------------------------------------------------------------------------

def _pole_reference(pos1, w1, pos2, w2, sedges, ells, los='firstpoint'):
    """(nsbins, nells) multipole counts, non-periodic."""
    d = pos2[None, :, :] - pos1[:, None, :]
    s = np.sqrt((d * d).sum(-1))
    if los == 'firstpoint':
        hat = pos1 / np.linalg.norm(pos1, axis=-1, keepdims=True)
        num = np.einsum('ijk,ik->ij', d, hat)
    elif los == 'endpoint':
        hat = pos2 / np.linalg.norm(pos2, axis=-1, keepdims=True)
        num = np.einsum('ijk,jk->ij', d, hat)
    else:  # midpoint
        ell_ = pos2[None, :, :] + pos1[:, None, :]
        num = (d * ell_).sum(-1)
        s = s * 1.0
    with np.errstate(invalid='ignore', divide='ignore'):
        if los == 'midpoint':
            ell_ = pos2[None, :, :] + pos1[:, None, :]
            den = s * np.sqrt((ell_ * ell_).sum(-1))
        else:
            den = s
        mu = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    mu[s == 0] = 0.0

    si, ok = _bin_index(s, sedges)
    w = w1[:, None] * w2[None, :]
    nb = len(sedges) - 1
    out = np.zeros((nb, len(ells)))
    for ill, ell in enumerate(ells):
        c = np.zeros(ell + 1)
        c[ell] = 1.
        leg = np.polynomial.legendre.legval(mu, c)
        out[:, ill] = np.bincount(si[ok], weights=(w * (2 * ell + 1) * leg)[ok],
                                  minlength=nb)
    return out


def _pole_setup(ells, los='firstpoint', n=400):
    pos1, w1 = catalog(1, n)
    pos2, w2 = catalog(2, n)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    sedges = np.linspace(1., 60., 13)
    battrs = BinAttrs(s=sedges, pole=(np.array(ells), los))
    mattrs = MeshAttrs(*particles, battrs=battrs)
    kw = dict(battrs=battrs, mattrs=mattrs)
    return particles, kw, (pos1, w1, pos2, w2, sedges)


@pytest.mark.parametrize('ells', [[0, 2, 4], [0], [1, 2, 3]])
def test_poles_match_brute_force(ells):
    particles, kw, raw = _pole_setup(ells)
    got = count2(*particles, backend='cpu', **kw)['weight']
    want = _pole_reference(*raw, ells)
    assert got.shape == want.shape
    scale = np.abs(want).max()
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


@needs_cuda
@pytest.mark.parametrize('los', ['firstpoint', 'endpoint', 'midpoint'])
def test_poles_match_cuda(los):
    """Both backends run the same set_legendre, so compare mode's rtol holds."""
    particles, kw, _ = _pole_setup([0, 2, 4], los=los)
    count2(*particles, backend='compare', **kw)


@needs_cuda
def test_poles_with_weights_match_cuda():
    """Multipoles compose with the scalar-tail weights (bitwise + negative)."""
    rng = np.random.default_rng(41)
    n = 300
    pos1, w1 = catalog(1, n)
    pos2, w2 = catalog(2, n)
    bits1 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64)]
    bits2 = [rng.integers(0, 0xffffffff, n, dtype=np.uint64)]
    nw1, nw2 = rng.uniform(0., 0.1, n), rng.uniform(0., 0.1, n)
    particles = (Particles(pos1, [w1] + bits1 + [nw1]),
                 Particles(pos2, [w2] + bits2 + [nw2]))
    wattrs = WeightAttrs(bitwise=dict(weights=bits1))
    battrs = BinAttrs(s=np.linspace(1., 60., 13),
                      pole=(np.array([0, 2]), 'firstpoint'))
    mattrs = MeshAttrs(*particles, battrs=battrs)
    count2(*particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs,
           backend='compare')


def test_k_pole_binning_is_declined():
    """(k, pole) has no s binning and is not implemented; decline by name."""
    particles, _, _ = _pole_setup([0, 2])
    battrs = BinAttrs(k=np.linspace(0.01, 0.2, 10),
                      pole=(np.array([0, 2]), 'firstpoint'))
    mattrs = MeshAttrs(*particles, battrs=battrs)
    with pytest.raises(NotImplementedError, match='binning'):
        count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu')


# ---------------------------------------------------------------------------
# Pair selections (s, theta): a per-pair veto, applied in the scalar tail so
# the plain vector path carries no extra compare. Bounds are inclusive on both
# ends, matching is_selected_pair in the CUDA kernel.
# ---------------------------------------------------------------------------

# Unit-sphere catalogues: separations and angular separations then span the
# full range, so a theta cut actually removes pairs. (With points spread over
# a 1000^3 box and s < 80, every pair sits within ~9 deg and any sane theta
# selection is a no-op -- a test that cannot fail.)
SEL_S = (0.5, 1.2)
SEL_THETA = (0., 60.)


def _selection_setup(kind, n=400):
    pos1, w1, _ = _unit_catalog(1, n)
    pos2, w2, _ = _unit_catalog(2, n)
    particles = (Particles(pos1, w1), Particles(pos2, w2))
    sedges = np.linspace(0.05, 2.0, 11)
    battrs = BinAttrs(s=sedges)
    if kind == 's':
        sattrs = SelectionAttrs(s=SEL_S)
    elif kind == 'theta':
        sattrs = SelectionAttrs(theta=SEL_THETA)
    else:
        sattrs = SelectionAttrs(s=SEL_S, theta=SEL_THETA)
    # A cartesian mesh: MeshAttrs would switch to the angular mesh if it saw
    # the theta selection, and that mesh is a separate (declined) feature.
    mattrs = MeshAttrs(*particles, battrs=battrs)
    return particles, dict(battrs=battrs, mattrs=mattrs, sattrs=sattrs), \
        (pos1, w1, pos2, w2, sedges)


def _selection_reference(pos1, w1, pos2, w2, sedges, kind):
    """Bounds are INCLUSIVE on both ends, matching is_selected_pair."""
    d = pos2[None, :, :] - pos1[:, None, :]
    s = np.sqrt((d * d).sum(-1))
    keep = np.ones(s.shape, dtype=bool)
    if kind in ('s', 'both'):
        keep &= (s >= SEL_S[0]) & (s <= SEL_S[1])
    if kind in ('theta', 'both'):
        h1 = pos1 / np.linalg.norm(pos1, axis=-1, keepdims=True)
        h2 = pos2 / np.linalg.norm(pos2, axis=-1, keepdims=True)
        ct = h1 @ h2.T
        # cos decreases with theta, so the bounds swap.
        keep &= (ct >= np.cos(np.radians(SEL_THETA[1]))) & \
                (ct <= np.cos(np.radians(SEL_THETA[0])))
    si, ok = _bin_index(s, sedges)
    w = w1[:, None] * w2[None, :]
    sel = np.bincount(si[ok & keep], weights=w[ok & keep],
                      minlength=len(sedges) - 1)
    allp = np.bincount(si[ok], weights=w[ok], minlength=len(sedges) - 1)
    return sel, allp


@pytest.mark.parametrize('kind', ['s', 'theta', 'both'])
def test_selections_match_brute_force(kind):
    particles, kw, raw = _selection_setup(kind)
    got = count2(*particles, backend='cpu', **kw)['weight']
    want, unselected = _selection_reference(*raw, kind)
    # The selection must actually bite, or this test proves nothing.
    assert want.sum() > 0 and want.sum() < 0.95 * unselected.sum()
    assert np.allclose(got, want, rtol=1e-9, atol=0)


@needs_cuda
@pytest.mark.parametrize('kind', ['s', 'theta', 'both'])
def test_selections_match_cuda(kind):
    particles, kw, _ = _selection_setup(kind)
    count2(*particles, backend='compare', **kw)
