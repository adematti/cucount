"""Correctness of the soft-histogram adjoint (the VJP used by custom_vjp).

The adjoint is hand written, and its failure mode is silent: gradients that look
plausible and are wrong by a factor or a sign. So the reference here is central
differences of the *forward kernel itself*, particle by particle, plus two
structural checks that a wrong formula usually breaks -- zero cotangent gives
zero gradients, and the total gradient of an autocorrelation over a periodic box
vanishes, because each ordered pair contributes ``+e d`` and ``-e d``.

Residual mismatches at the 1e-6 level for a step of 1e-6 are finite-difference
noise (the loss is O(1e3), so 1e-16 relative roundoff divided by 2h is ~1e-7 of
a gradient of order one); the tests use a larger step where it stays small and
assert against the level that noise implies.
"""

import numpy as np
import pytest

from cucount.numpy import BinAttrs, MeshAttrs, Particles, cudalib
from cucount.numpy import count2_soft, count2_soft_vjp

needs_cuda = pytest.mark.skipif(
    cudalib is None or cudalib.device_count() == 0,
    reason='CUDA extension not built, or no GPU visible')

pytestmark = needs_cuda

BOX = 200.0
EDGES = np.linspace(1.0, 100.0, 11)
SIGMA = 3.0


def _setup(n=120, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, BOX, (n, 3))
    w = rng.uniform(0.5, 1.5, n)
    ba = BinAttrs(s=EDGES)
    ma = MeshAttrs(Particles(pos, w), Particles(pos, w), boxsize=BOX, battrs=ba, periodic=True)
    return pos, w, ba, ma


def _loss(pos, w, ba, g, sigma=SIGMA):
    p = Particles(pos, w)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    counts = count2_soft(p, p, battrs=ba, mattrs=ma, soft_sigma=sigma)['weight']
    return float(np.dot(g, counts))


def _vjp(pos, w, ba, g, sigma=SIGMA):
    p = Particles(pos, w)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    out = count2_soft_vjp(p, p, battrs=ba, cotangent=g, mattrs=ma, soft_sigma=sigma)
    gpos = np.asarray(out['grad_position1']) + np.asarray(out['grad_position2'])
    gw = np.asarray(out['grad_weight1']) + np.asarray(out['grad_weight2'])
    return gpos, gw, float(out['grad_sigma']), out


def test_grad_positions_matches_finite_difference():
    pos, w, ba, ma = _setup()
    g = np.random.default_rng(11).normal(size=EDGES.size - 1)
    gpos, _, _, _ = _vjp(pos, w, ba, g)

    h = 1e-5
    rng = np.random.default_rng(4)
    for i in rng.choice(pos.shape[0], size=4, replace=False):
        for axis in range(3):
            pp = pos.copy(); pp[i, axis] += h
            pm = pos.copy(); pm[i, axis] -= h
            fd = (_loss(pp, w, ba, g) - _loss(pm, w, ba, g)) / (2 * h)
            assert fd == pytest.approx(gpos[i, axis], rel=2e-5, abs=1e-7)


def test_grad_weights_matches_finite_difference():
    pos, w, ba, ma = _setup()
    g = np.random.default_rng(12).normal(size=EDGES.size - 1)
    _, gw, _, _ = _vjp(pos, w, ba, g)

    h = 1e-5
    for i in np.random.default_rng(5).choice(w.size, size=4, replace=False):
        wp = w.copy(); wp[i] += h
        wm = w.copy(); wm[i] -= h
        fd = (_loss(pos, wp, ba, g) - _loss(pos, wm, ba, g)) / (2 * h)
        assert fd == pytest.approx(gw[i], rel=2e-5, abs=1e-7)


def test_grad_sigma_matches_finite_difference():
    pos, w, ba, ma = _setup()
    g = np.random.default_rng(13).normal(size=EDGES.size - 1)
    _, _, gsigma, _ = _vjp(pos, w, ba, g)

    h = 1e-4 * SIGMA
    fd = (_loss(pos, w, ba, g, SIGMA + h) - _loss(pos, w, ba, g, SIGMA - h)) / (2 * h)
    assert fd == pytest.approx(gsigma, rel=1e-4, abs=1e-6)


def test_zero_cotangent_gives_zero_gradients():
    pos, w, ba, ma = _setup(n=60)
    g = np.zeros(EDGES.size - 1)
    gpos, gw, gsigma, out = _vjp(pos, w, ba, g)
    assert np.all(gpos == 0.0)
    assert np.all(gw == 0.0)
    assert gsigma == 0.0
    assert out['grad_position1'].shape == (pos.shape[0], 3)


def test_autocorrelation_total_gradient_vanishes():
    # Every ordered pair contributes -e d to i and +e d to j; summing over all
    # particles in a periodic box must cancel to roundoff, which a sign error in
    # ds/dr would not.
    pos, w, ba, ma = _setup(n=200)
    g = np.random.default_rng(14).normal(size=EDGES.size - 1)
    gpos, _, _, _ = _vjp(pos, w, ba, g)
    total = gpos.sum(axis=0)
    assert np.allclose(total, 0.0, atol=1e-9)


def test_cotangent_length_is_validated():
    pos, w, ba, ma = _setup(n=50)
    with pytest.raises(Exception, match='cotangent'):
        count2_soft_vjp(Particles(pos, w), Particles(pos, w), battrs=ba,
                        cotangent=np.zeros(EDGES.size), mattrs=ma)


def test_weight_gradients_absent_without_individual_weights():
    pos, w, ba, ma = _setup(n=50)
    g = np.ones(EDGES.size - 1)
    p = Particles(pos)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    out = count2_soft_vjp(p, p, battrs=ba, cotangent=g, mattrs=ma)
    assert out['grad_weight1'] is None
    assert out['grad_weight2'] is None
    assert np.asarray(out['grad_position1']).shape == (pos.shape[0], 3)


def test_non_uniform_grid_gradients_match_finite_difference():
    """The general (adaptive-window) adjoint must differentiate the same function.

    With ``sigma=None`` on a non-uniform grid the width of the kernel is a
    property of the binning, not a parameter, so ``grad_sigma`` is identically
    zero and only positions and weights carry a cotangent.
    """
    edges = np.geomspace(1.0, 100.0, 11)
    pos, w, _, _ = _setup(n=90, seed=17)
    ba = BinAttrs(s=edges)
    g = np.random.default_rng(21).normal(size=edges.size - 1)

    def loss(pp, ww):
        return _loss(pp, ww, ba, g, sigma=None)

    def vjp(pp, ww):
        p = Particles(pp, ww)
        ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
        out = count2_soft_vjp(p, p, battrs=ba, cotangent=g, mattrs=ma)
        return (np.asarray(out['grad_position1']) + np.asarray(out['grad_position2']),
                np.asarray(out['grad_weight1']) + np.asarray(out['grad_weight2']),
                float(out['grad_sigma']))

    gpos, gw, gs = vjp(pos, w)
    assert gs == 0.0

    h = 1e-5
    rng = np.random.default_rng(6)
    for i in rng.choice(pos.shape[0], size=3, replace=False):
        for axis in range(3):
            pp = pos.copy(); pp[i, axis] += h
            pm = pos.copy(); pm[i, axis] -= h
            fd = (loss(pp, w) - loss(pm, w)) / (2 * h)
            assert fd == pytest.approx(gpos[i, axis], rel=2e-4, abs=1e-6)
    for i in rng.choice(w.size, size=3, replace=False):
        wp = w.copy(); wp[i] += h
        wm = w.copy(); wm[i] -= h
        fd = (loss(pos, wp) - loss(pos, wm)) / (2 * h)
        assert fd == pytest.approx(gw[i], rel=2e-4, abs=1e-6)


def test_non_uniform_adjoint_fast_taps_tracks_exact():
    """soft_taps='fast' changes the exp implementation, not the differentiated function.

    The adjoint must follow the same taps as the forward it differentiates, so
    swapping __expf in may move gradients by no more than the ~1e-6 the
    approximation itself introduces.
    """
    edges = np.geomspace(1.0, 100.0, 11)
    rng = np.random.default_rng(41)
    n = 60
    pos = rng.uniform(0.0, BOX, (n, 3))
    w = rng.uniform(0.5, 1.5, n)
    ba = BinAttrs(s=edges)
    g = rng.normal(size=edges.size - 1)

    def run(kw):
        p = Particles(pos, w)
        ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
        out = count2_soft_vjp(p, p, battrs=ba, cotangent=g, mattrs=ma, soft_sigma=None, **kw)
        return np.asarray(out['grad_position1']), np.asarray(out['grad_weight1'])

    gpos_e, gw_e = run({})
    gpos_f, gw_f = run({'soft_taps': 'fast'})
    assert np.abs(gpos_f - gpos_e).max() <= 1e-4 * np.abs(gpos_e).max()
    assert np.abs(gw_f - gw_e).max() <= 1e-4 * np.abs(gw_e).max()
