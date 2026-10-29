"""Correctness of the soft mu binning (differentiable multipoles).

The mu axis is softened with the same Gaussian-softmax construction as s, but it
is not histogrammed: each pair deposits ``w_pair * Ws_i * M_l(mu)`` in slot
``(i, l)``, where ``M_l`` is the truncated softmax over a uniform mu grid folded
onto ``(2l+1) P_l`` of the grid *centres*. These tests pin that definition down
independently of the kernel:

* a numpy O(N^2) reference builds ``M_l`` from ``soft_mu_weights`` and the
  separation weights from ``soft_weights``, so forward agreement is a real check
  and not two implementations sharing a formula;
* the ``l = 0`` slot must equal the soft-s counts exactly (``M_0 = 1`` by
  construction), which catches a mu profile that does not sum to one;
* ``sigma_mu -> 0`` must approach the *hard* multipoles of ``count2`` as the mu
  quadrature is refined;
* the analytic adjoint is checked against central differences of the forward,
  for every line of sight and for odd multipoles (which only a cross-correlation
  can exercise: in an autocorrelation the pairs (i, j) and (j, i) cancel).
"""

import numpy as np
import pytest

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, cudalib)
from cucount.numpy import (count2, count2_soft, count2_soft_vjp, mu_grid,
                           mu_kmax, mu_spec, soft_mu_weights, soft_weights)

needs_cuda = pytest.mark.skipif(
    cudalib is None or cudalib.device_count() == 0,
    reason='CUDA extension not built, or no GPU visible')

BOX = 400.0
EDGES = np.linspace(5.0, 60.0, 12)
ELLS = np.array([0, 2, 4])
CUTOFF = 1e-13


def _catalogue(n, seed, box=BOX):
    rng = np.random.default_rng(seed)
    return Particles(rng.uniform(0.0, box, (n, 3)), rng.uniform(0.5, 1.5, n))


def _setup(ells=ELLS, los='z', n1=90, n2=None, seed=7, box=BOX):
    p1 = _catalogue(n1, seed, box)
    p2 = p1 if n2 is None else _catalogue(n2, seed + 1, box)
    ba = BinAttrs(s=EDGES, pole=(ells, los))
    ma = MeshAttrs(p1, p2, boxsize=box, battrs=ba, periodic=True)
    return p1, p2, ba, ma


def _lhat(los, pos1, pos2):
    """The line of sight of a pair, matching ``compute_pair_mu``."""
    if los == 'firstpoint':
        v = pos1
    elif los == 'endpoint':
        v = pos2
    elif los == 'midpoint':
        v = pos1 + pos2
    else:
        v = None
    if v is None:
        e = np.zeros(3)
        e['xyz'.index(los)] = 1.0
        return e
    return v / np.linalg.norm(v)


def _brute(p1, p2, edges, ells, los, soft_sigma=None, soft_mu_sigma=None,
           soft_mu_bins=0, box=BOX, cutoff=CUTOFF, soft_mu_edges=None):
    """Ordered-pair reference histogram, shape ``(nbins, nell)``."""
    pos1 = np.asarray(p1.positions)
    pos2 = np.asarray(p2.positions)
    w1 = np.asarray(p1.values[0])
    w2 = np.asarray(p2.values[0])
    hist = np.zeros((edges.size - 1, len(ells)))

    for i in range(pos1.shape[0]):
        diff = pos2 - pos1[i]
        if box is not None:
            diff -= box * np.round(diff / box)
        s = np.sqrt((diff ** 2).sum(axis=-1))
        keep = (s >= edges[0]) & (s < edges[-1]) & (s > 0)
        if not np.any(keep):
            continue
        d = diff[keep]
        sk = s[keep]
        mu = np.array([np.dot(d[k], _lhat(los, pos1[i], pos2[keep][k])) / sk[k]
                       for k in range(sk.size)])
        ws = soft_weights(sk, edges, soft_sigma)
        prof = soft_mu_weights(mu, ells, soft_mu_sigma, soft_mu_bins, cutoff,
                               soft_mu_edges=soft_mu_edges)
        hist += np.einsum('p,pb,pl->bl', w1[i] * w2[keep], ws, prof)
    return hist


@needs_cuda
def test_matches_brute_force_for_every_line_of_sight():
    ells = np.array([0, 2, 4])
    for los in ('z', 'x', 'midpoint', 'firstpoint', 'endpoint'):
        p1, p2, ba, ma = _setup(ells=ells, los=los, n1=40, n2=35)
        got = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma,
                                     soft_cutoff=CUTOFF)['weight'])
        want = _brute(p1, p2, EDGES, ells, los)
        assert got.shape == want.shape
        assert np.allclose(got, want, rtol=1e-11, atol=1e-12), los


@needs_cuda
def test_matches_brute_force_with_a_finite_mu_width():
    ells = np.array([0, 2])
    p1, p2, ba, ma = _setup(ells=ells, los='midpoint', n1=30, n2=25)
    got = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma,
                                 soft_cutoff=CUTOFF,
                                 soft_mu_sigma=0.1)['weight'])
    want = _brute(p1, p2, EDGES, ells, 'midpoint', soft_mu_sigma=0.1)
    assert np.allclose(got, want, rtol=1e-11, atol=1e-12)


@needs_cuda
def test_monopole_is_the_soft_s_count():
    """M_0 = 1 exactly, so asking for ell=0 must not change the answer."""
    p1, p2, ba, ma = _setup(ells=ELLS, los='z', n1=60, n2=50)
    soft = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF)
    ba_s = BinAttrs(s=EDGES)
    ma_s = MeshAttrs(p1, p2, boxsize=BOX, battrs=ba_s, periodic=True)
    iso = count2_soft(p1, p2, battrs=ba_s, mattrs=ma_s, soft_cutoff=CUTOFF)
    ells = [int(e) for e in np.asarray(ba.coords('pole'))]
    i0 = ells.index(0)
    mono = np.asarray(soft['weight'])[:, i0]
    ref = np.asarray(iso['weight'])
    assert np.allclose(mono, ref, rtol=1e-14, atol=1e-14)
    assert np.allclose(np.asarray(soft['weight']).sum(axis=0)[i0], ref.sum())


@needs_cuda
def test_refining_the_mu_grid_approaches_hard_multipoles():
    """sigma_mu -> 0 with a fine grid recovers count2's hard multipoles."""
    ells = np.array([0, 2, 4])
    p1, p2, ba, ma = _setup(ells=ells, los='z', n1=40, n2=35)
    soft = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma,
                                  soft_cutoff=CUTOFF, soft_mu_sigma=1e-3,
                                  soft_mu_bins=1024)['weight'])
    hard = np.asarray(count2(p1, p2, battrs=ba, mattrs=ma)['weight'])
    # Per-bin values differ because s is still smoothed: sigma -> 0 is a limit of
    # the *multipole moments*, so the s bins are summed over. The mu profile also
    # snaps mu onto the quadrature, so agreement is O(dmu * P_l'), not exact.
    assert np.allclose(soft.sum(axis=0), hard.sum(axis=0), rtol=2e-2, atol=1e-6)


@needs_cuda
def test_adjoint_matches_central_differences():
    """dL/dpositions and dL/dweights against finite differences of the forward."""
    ells = np.array([0, 2])
    p1, p2, ba, ma = _setup(ells=ells, los='midpoint', n1=12, n2=10)
    rng = np.random.default_rng(3)
    g = rng.normal(size=(EDGES.size - 1, ells.size))

    def loss(pos1, pos2, w1, w2):
        a = Particles(pos1, w1)
        b = Particles(pos2, w2)
        hist = np.asarray(count2_soft(a, b, battrs=ba, mattrs=ma,
                                      soft_cutoff=CUTOFF)['weight'])
        return float((hist * g).sum())

    pos1 = np.asarray(p1.positions)
    pos2 = np.asarray(p2.positions)
    w1 = np.asarray(p1.values[0])
    w2 = np.asarray(p2.values[0])
    adj = count2_soft_vjp(p1, p2, battrs=ba, cotangent=g, mattrs=ma,
                          soft_cutoff=CUTOFF)

    h = 1e-6
    for name, base, key in (('pos1', pos1, 'grad_position1'),
                            ('pos2', pos2, 'grad_position2')):
        fd = np.zeros_like(base)
        for i in range(base.shape[0]):
            for ax in range(3):
                plus = base.copy(); plus[i, ax] += h
                minus = base.copy(); minus[i, ax] -= h
                lp = loss(plus, pos2, w1, w2) if name == 'pos1' else loss(pos1, plus, w1, w2)
                lm = loss(minus, pos2, w1, w2) if name == 'pos1' else loss(pos1, minus, w1, w2)
                fd[i, ax] = (lp - lm) / (2 * h)
        got = np.asarray(adj[key])
        assert np.allclose(got, fd, rtol=1e-4, atol=1e-6), name

    for key, base, arg in (('grad_weight1', w1, 0), ('grad_weight2', w2, 1)):
        fd = np.zeros_like(base)
        for i in range(base.shape[0]):
            plus, minus = base.copy(), base.copy()
            plus[i] += h
            minus[i] -= h
            if arg == 0:
                lp, lm = loss(pos1, pos2, plus, w2), loss(pos1, pos2, minus, w2)
            else:
                lp, lm = loss(pos1, pos2, w1, plus), loss(pos1, pos2, w1, minus)
            fd[i] = (lp - lm) / (2 * h)
        assert np.allclose(np.asarray(adj[key]), fd, rtol=1e-5, atol=1e-7), key


@needs_cuda
def test_odd_multipoles_are_differentiable():
    """Odd ells vanish in an autocorrelation, so a cross-correlation is needed."""
    ells = np.array([1, 3])
    p1, p2, ba, ma = _setup(ells=ells, los='z', n1=10, n2=8)
    rng = np.random.default_rng(4)
    g = rng.normal(size=(EDGES.size - 1, ells.size))
    adj = count2_soft_vjp(p1, p2, battrs=ba, cotangent=g, mattrs=ma,
                          soft_cutoff=CUTOFF)
    pos1 = np.asarray(p1.positions)
    h = 1e-6
    i, ax = 3, 1

    def loss(pos):
        hist = np.asarray(count2_soft(Particles(pos, p1.values[0]), p2, battrs=ba,
                                      mattrs=ma, soft_cutoff=CUTOFF)['weight'])
        return float((hist * g).sum())

    plus, minus = pos1.copy(), pos1.copy()
    plus[i, ax] += h
    minus[i, ax] -= h
    fd = (loss(plus) - loss(minus)) / (2 * h)
    got = np.asarray(adj['grad_position1'])[i, ax]
    assert abs(got - fd) < 1e-4 * max(abs(fd), 1.0)
    assert abs(fd) > 1e-3  # the case is not degenerate


@needs_cuda
def test_meta_reports_the_multipole_configuration():
    p1, p2, ba, ma = _setup(ells=ELLS, los='z', n1=20, n2=15)
    meta = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                       soft_mu_sigma=0.1)['soft']
    assert meta['multipoles'] == [int(e) for e in ELLS]
    assert meta['los'] == 'z'
    centres, sigma_mu, _ = mu_grid(0.1, 0)
    assert meta['mu_bins'] == centres.size
    assert meta['mu_sigma'] == pytest.approx(sigma_mu)
    assert meta['mu_kmax'] == mu_kmax(0.1, 0, CUTOFF)
    assert meta['nbins'] == EDGES.size - 1


@needs_cuda
def test_rejects_a_pole_axis_without_a_line_of_sight():
    p1, p2, _, _ = _setup(ells=np.array([0, 2]), los='z', n1=10, n2=9)
    ba = BinAttrs(s=EDGES, pole=(np.array([0, 2]), ''))
    ma = MeshAttrs(p1, p2, boxsize=BOX, battrs=ba, periodic=True)
    with pytest.raises(ValueError, match='line of sight'):
        count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF)


@needs_cuda
def test_rejects_multipoles_out_of_range():
    p1, p2, _, _ = _setup(ells=np.array([0, 2]), los='z', n1=10, n2=9)
    ba = BinAttrs(s=EDGES, pole=(np.array([0, 9]), 'z'))
    ma = MeshAttrs(p1, p2, boxsize=BOX, battrs=ba, periodic=True)
    with pytest.raises(ValueError, match='multipoles up to ell'):
        count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF)


def test_soft_mu_weights_is_normalised_at_ell_zero():
    centres, sigma_mu, dmu = mu_grid()
    assert centres.size == 40
    assert sigma_mu == pytest.approx(0.5 * dmu)
    mu = np.linspace(-1.0, 1.0, 41)
    prof = soft_mu_weights(mu, np.array([0, 2]), None, 0, CUTOFF)
    assert prof.shape == (41, 2)
    assert np.allclose(prof[:, 0], 1.0, rtol=0, atol=1e-14)
    # even multipoles are even in mu, odd ones odd
    odd = soft_mu_weights(mu, np.array([1]), 0.05, 0, CUTOFF)[:, 0]
    assert np.allclose(odd, -odd[::-1], atol=1e-12)


def test_mu_grid_rejects_nothing_but_clamps():
    assert mu_grid(None, 0)[0].size == 40
    assert mu_grid(0.5, 0)[0].size == 8            # ceil(4 / 0.5)
    assert mu_grid(None, 1)[0].size == 4           # clamped below
    assert mu_grid(None, 100000)[0].size == 1024   # clamped above


# --------------------------------------------------------------------------
# Caller-supplied mu edges
# --------------------------------------------------------------------------

MU_EDGES_UNIFORM = np.linspace(-1.0, 1.0, 41)
MU_EDGES_COS = -np.cos(np.linspace(0.0, np.pi, 25))


@needs_cuda
def test_uniform_mu_edges_reproduce_the_internal_grid():
    """Arithmetic edges must be *identical* to the default grid, i.e. still run
    the two-exp recurrence path and not the general one."""
    p1, p2, ba, ma = _setup(ells=np.array([0, 2, 4]), n1=40, n2=35)
    internal = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF)
    edges = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                        soft_mu_edges=MU_EDGES_UNIFORM)
    # Same formula, same two-exp recurrence: equal to rounding (the edges carry a
    # 1-ulp difference in the derived mu width).
    assert np.allclose(np.asarray(edges['weight']), np.asarray(internal['weight']),
                       rtol=1e-13, atol=1e-14)
    meta = edges['soft']
    assert meta['mu_uniform'] is True
    assert meta['mu_bins'] == MU_EDGES_UNIFORM.size - 1
    assert meta['mu_taps'] == 2 * meta['mu_kmax'] + 1
    # a uniform grid needs no edge table: the metadata is the (spacing, count) pair
    assert 'mu_edges' not in meta


@needs_cuda
def test_non_uniform_mu_edges_match_brute_force():
    """A cos-spaced grid, width-scaled sigma (the default) and an absolute one."""
    ells = np.array([0, 2, 4])
    p1, p2, ba, ma = _setup(ells=ells, n1=40, n2=35)
    got = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                                 soft_mu_edges=MU_EDGES_COS)['weight'])
    want = _brute(p1, p2, EDGES, ells, 'z', soft_mu_edges=MU_EDGES_COS)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)

    meta = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                       soft_mu_edges=MU_EDGES_COS)['soft']
    assert meta['mu_uniform'] is False
    assert meta['mu_sigma_kind'] == 'width'
    assert meta['mu_taps'] == -1          # the window adapts, so no fixed count
    assert np.allclose(np.asarray(meta['mu_edges']), MU_EDGES_COS)

    got = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                                 soft_mu_edges=MU_EDGES_COS,
                                 soft_mu_sigma=0.2)['weight'])
    want = _brute(p1, p2, EDGES, ells, 'z', soft_mu_sigma=0.2,
                  soft_mu_edges=MU_EDGES_COS)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


@needs_cuda
def test_partial_mu_edges_leave_pairs_to_the_boundary_cell():
    """A grid covering only part of [-1, 1] is allowed: mu is never folded, and
    the pairs beyond it pile into the nearest boundary cell."""
    ells = np.array([0, 2])
    edges = np.linspace(0.0, 1.0, 11)
    p1, p2, ba, ma = _setup(ells=ells, n1=30, n2=25)
    got = np.asarray(count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF,
                                 soft_mu_edges=edges)['weight'])
    want = _brute(p1, p2, EDGES, ells, 'z', soft_mu_edges=edges)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)
    # folded mu would make M_1 vanish in an autocorrelation; unfolded it need not
    assert np.abs(got[:, 0]).sum() > 0


@needs_cuda
def test_adjoint_with_non_uniform_mu_edges_matches_central_differences():
    ells = np.array([0, 2])
    p1, p2, ba, ma = _setup(ells=ells, n1=12, n2=10)
    rng = np.random.default_rng(11)
    g = rng.normal(size=(EDGES.size - 1, ells.size))
    adj = count2_soft_vjp(p1, p2, battrs=ba, cotangent=g, mattrs=ma,
                          soft_cutoff=CUTOFF, soft_mu_edges=MU_EDGES_COS)
    pos1 = np.asarray(p1.positions)
    w1 = np.asarray(p1.values[0])
    h = 1e-6

    def loss(pos):
        hist = np.asarray(count2_soft(Particles(pos, w1), p2, battrs=ba,
                                      mattrs=ma, soft_cutoff=CUTOFF,
                                      soft_mu_edges=MU_EDGES_COS)['weight'])
        return float((hist * g).sum())

    # a few coordinates are enough: this pins the per-cell sigma chain rule
    for i, ax in ((0, 0), (3, 1), (7, 2)):
        plus, minus = pos1.copy(), pos1.copy()
        plus[i, ax] += h
        minus[i, ax] -= h
        fd = (loss(plus) - loss(minus)) / (2 * h)
        got = np.asarray(adj['grad_position1'])[i, ax]
        assert abs(got - fd) < 1e-5 * max(abs(fd), 1.0), (i, ax, got, fd)

    fd = np.zeros_like(w1)
    for i in range(w1.size):
        plus, minus = w1.copy(), w1.copy()
        plus[i] += h
        minus[i] -= h
        lp = float((np.asarray(count2_soft(Particles(pos1, plus), p2, battrs=ba,
                                           mattrs=ma, soft_cutoff=CUTOFF,
                                           soft_mu_edges=MU_EDGES_COS)['weight']) * g).sum())
        lm = float((np.asarray(count2_soft(Particles(pos1, minus), p2, battrs=ba,
                                           mattrs=ma, soft_cutoff=CUTOFF,
                                           soft_mu_edges=MU_EDGES_COS)['weight']) * g).sum())
        fd[i] = (lp - lm) / (2 * h)
    assert np.allclose(np.asarray(adj['grad_weight1']), fd, rtol=1e-5, atol=1e-7)


@needs_cuda
def test_mu_edges_are_rejected_when_malformed():
    p1, p2, ba, ma = _setup(ells=np.array([0, 2]), n1=10, n2=9)

    def call(**kw):
        return count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_cutoff=CUTOFF, **kw)

    with pytest.raises(ValueError, match='not both'):
        call(soft_mu_bins=40, soft_mu_edges=MU_EDGES_UNIFORM)
    with pytest.raises(ValueError, match='at least 2 mu edges'):
        call(soft_mu_edges=np.array([0.0]))
    with pytest.raises(ValueError, match='not folded'):
        call(soft_mu_edges=np.array([-1.0, 0.0, 1.5]))
    with pytest.raises(ValueError, match='strictly increasing'):
        call(soft_mu_edges=np.array([-1.0, 0.1, 0.1, 1.0]))
    with pytest.raises(ValueError, match='at most 1024'):
        call(soft_mu_edges=np.linspace(-1.0, 1.0, 1030))


def test_mu_spec_and_soft_mu_weights_accept_edges():
    spec = mu_spec(None, 0, MU_EDGES_COS)
    assert not spec['uniform']
    assert spec['nmu'] == MU_EDGES_COS.size - 1
    assert np.allclose(spec['sigmas'], 0.5 * np.diff(MU_EDGES_COS))
    assert np.allclose(spec['centres'],
                       0.5 * (MU_EDGES_COS[1:] + MU_EDGES_COS[:-1]))
    prof = soft_mu_weights(np.array([-0.5, 0.0, 0.5]), np.array([0, 2]), None, 0,
                           CUTOFF, soft_mu_edges=MU_EDGES_COS)
    assert np.allclose(prof[:, 0], 1.0, rtol=0, atol=1e-14)
