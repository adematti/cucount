"""Correctness of the soft (Gaussian-softmax) separation binning.

The soft kernel is an extension beside the hard one, so these tests are
deliberately independent of the hard count2 machinery they compare against:
a numpy O(N^2) brute force defines the answer, hard binning supplies the
sigma -> 0 limit, and the invariants (weight conservation, monotone cutoff
truncation) are checked directly so a wrong kernel cannot pass by agreeing with
another wrong one.

Requests are made with backend='cuda' explicitly: soft binning is CUDA-only in
v1, and a request for another backend must raise rather than silently fall back
to hard binning.
"""

import numpy as np
import pytest

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           SplitAttrs, WeightAttrs, cudalib)
from cucount.numpy import count2, count2_soft, default_sigma, soft_weights

from cucount.numpy.soft import _bin_centres, is_uniform, sigmas

needs_cuda = pytest.mark.skipif(
    cudalib is None or cudalib.device_count() == 0,
    reason='CUDA extension not built, or no GPU visible')

BOX = 1000.0
EDGES = np.linspace(1.0, 100.0, 21)
DS = (EDGES[-1] - EDGES[0]) / (EDGES.size - 1)


def _setup(n=400, seed=5, edges=EDGES, box=BOX, periodic=True):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, (n, 3))
    w = rng.uniform(0.5, 1.5, n)
    p = Particles(pos, w)
    ba = BinAttrs(s=edges)
    ma = MeshAttrs(p, p, boxsize=box, battrs=ba, periodic=periodic)
    return p, ba, ma


def _brute_soft(p, edges, sigma, box=None):
    """Ordered-pair reference: minimum-image separation, full-grid softmax."""
    pos = np.asarray(p.positions)
    w = np.asarray(p.values[0])
    i, j = np.indices((pos.shape[0], pos.shape[0]))
    diff = pos[i] - pos[j]
    if box is not None:
        diff -= box * np.round(diff / box)
    s = np.sqrt((diff ** 2).sum(axis=-1))
    inside = (s >= edges[0]) & (s < edges[-1]) & (i != j)
    s = s[inside]
    ww = (w[i] * w[j])[inside]
    return soft_weights(s, edges, sigma).T @ ww


@needs_cuda
def test_conserves_total_weight():
    """Soft binning smooths a histogram, it does not reweight pairs."""
    p, ba, ma = _setup()
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cuda')['weight']
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda')
    assert soft['weight'].shape == hard.shape
    np.testing.assert_allclose(soft['weight'].sum(), hard.sum(), rtol=1e-12)


@needs_cuda
def test_matches_brute_force():
    p, ba, ma = _setup()
    sigma = 2.0
    got = count2_soft(p, p, battrs=ba, mattrs=ma,
                      soft_sigma=sigma, backend='cuda')['weight']
    want = _brute_soft(p, EDGES, sigma, box=BOX)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-8 * want.max())


@needs_cuda
def test_non_periodic_matches_brute_force():
    p, ba, ma = _setup(n=300, seed=11, periodic=False)
    sigma = 3.0
    got = count2_soft(p, p, battrs=ba, mattrs=ma,
                      soft_sigma=sigma, backend='cuda')['weight']
    want = _brute_soft(p, EDGES, sigma, box=None)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-8 * want.max())


@needs_cuda
def test_sigma_to_zero_is_hard_binning():
    """The limit the definition promises: a very narrow kernel is hard binning."""
    p, ba, ma = _setup()
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cuda')['weight']
    soft = count2_soft(p, p, battrs=ba, mattrs=ma,
                       soft_sigma=1e-6, backend='cuda')['weight']
    np.testing.assert_allclose(soft, hard, rtol=1e-10, atol=1e-10 * hard.max())


@needs_cuda
def test_cutoff_truncation_is_monotone():
    """Loosening the cutoff must cost accuracy, and never change the total."""
    p, ba, ma = _setup()
    sigma = 2.0
    ref = count2_soft(p, p, battrs=ba, mattrs=ma, soft_sigma=sigma,
                      soft_cutoff=1e-13, backend='cuda')['weight']
    errs = []
    for cutoff in (1e-6, 1e-3):
        got = count2_soft(p, p, battrs=ba, mattrs=ma, soft_sigma=sigma,
                          soft_cutoff=cutoff, backend='cuda')
        np.testing.assert_allclose(got['weight'].sum(), ref.sum(), rtol=1e-12)
        errs.append(np.abs(got['weight'] - ref).max())
    assert errs[0] < errs[1] < 1e-2 * ref.max()


@needs_cuda
def test_meta_reports_the_configuration_it_ran():
    p, ba, ma = _setup()
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda')
    meta = soft['soft']
    assert meta['sigma'] == pytest.approx(default_sigma(EDGES))
    # taps must cover the cutoff window, and be odd (centre + both sides)
    assert meta['taps'] == 2 * meta['kmax'] + 1
    assert meta['path'] in ('shared', 'global')
    assert meta['nbins'] == EDGES.size - 1


@needs_cuda
def test_bin_grid_of_arange_is_served():
    """BIN_CUSTOM with uniform edges must work: uniformity is what is required."""
    p, ba, ma = _setup(edges=np.arange(1.0, 101.0, 5.0))
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda')
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cuda')['weight']
    np.testing.assert_allclose(soft['weight'].sum(), hard.sum(), rtol=1e-12)


LOG_EDGES = np.geomspace(1.0, 100.0, 21)


@needs_cuda
def test_log_grid_width_scaled_sigma_matches_brute_force():
    """Non-uniform grid, default sigma: half of each bin, not half a mean bin."""
    p, ba, ma = _setup(edges=LOG_EDGES)
    res = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda')
    want = _brute_soft(p, LOG_EDGES, None, box=BOX)
    np.testing.assert_allclose(res['weight'], want, rtol=1e-9, atol=1e-9 * want.max())
    meta = res['soft']
    assert meta['sigma_kind'] == 'width'
    assert meta['uniform'] is False
    # The general path expands its window to the cutoff, so there is no single tap count.
    assert meta['taps'] == -1


@needs_cuda
def test_log_grid_absolute_sigma_matches_brute_force():
    """An explicit sigma is one absolute width on every bin, uniform grid or not."""
    p, ba, ma = _setup(edges=LOG_EDGES)
    sigma = 4.0
    res = count2_soft(p, p, battrs=ba, mattrs=ma, soft_sigma=sigma, backend='cuda')
    want = _brute_soft(p, LOG_EDGES, sigma, box=BOX)
    np.testing.assert_allclose(res['weight'], want, rtol=1e-9, atol=1e-9 * want.max())
    assert res['soft']['sigma_kind'] == 'absolute'
    assert res['soft']['uniform'] is False


@needs_cuda
def test_non_uniform_fast_taps_tracks_exact():
    """The opt-in __expf path trades ~1e-6 relative accuracy, nothing more."""
    p, ba, ma = _setup(edges=LOG_EDGES)
    exact = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda')['weight']
    fast = count2_soft(p, p, battrs=ba, mattrs=ma,
                       soft_taps='fast', backend='cuda')['weight']
    scale = exact.max()
    np.testing.assert_allclose(fast, exact, rtol=1e-5, atol=1e-5 * scale)
    assert not np.array_equal(fast, exact)  # it really took the other branch


@needs_cuda
def test_log_grid_conserves_total_weight():
    """The general path truncates its window, so it must renormalise inside it.

    This is the invariant that a wrong cutoff test would break: dropping taps and
    forgetting that the remaining sum has to be renormalised loses weight, and a
    brute-force comparison at loose tolerance would not notice a small leak.
    """
    p, ba, ma = _setup(edges=LOG_EDGES)
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cuda')['weight']
    for kw in ({}, {'soft_sigma': 2.0}, {'soft_taps': 'fast'}):
        soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda', **kw)
        np.testing.assert_allclose(soft['weight'].sum(), hard.sum(), rtol=1e-12)


@needs_cuda
def test_log_grid_tiny_absolute_sigma_snaps_to_the_nearest_centre():
    """A sigma far below every bin width leaves the pair on one bin -- the nearest centre.

    On a *uniform* grid the nearest centre is always the centre of the bin that
    contains s, so sigma -> 0 reproduces hard binning there. On a non-uniform grid
    it does not: a wide bin's centre can sit far from s while the centre of the
    neighbouring narrow bin is closer, so the pair lands one bin down. That is a
    property of centring a Gaussian on bins of unequal width, not a kernel bug,
    which is why this compares against the numpy softmax reference rather than
    against the hard histogram. The width-scaled sigma (the default here) keeps
    the weight in the containing bin because it shrinks with the local width.
    """
    p, ba, ma = _setup(edges=LOG_EDGES)
    for sigma in (1e-4, 1e-6):
        soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cuda', soft_sigma=sigma)
        np.testing.assert_allclose(soft['weight'],
                                   _brute_soft(p, LOG_EDGES, sigma, box=BOX),
                                   rtol=1e-11, atol=1e-9 * soft['weight'].max())


@needs_cuda
def test_uniform_grid_reports_recurrence_path():
    """A uniform grid keeps the two-exp recurrence, with the fixed window it implies."""
    p, ba, ma = _setup()
    res = count2_soft(p, p, battrs=ba, mattrs=ma, soft_sigma=2.0, backend='cuda')
    meta = res['soft']
    assert meta['uniform'] is True
    assert meta['sigma_kind'] == 'absolute'
    assert meta['taps'] == 2 * meta['kmax'] + 1


@needs_cuda
def test_rejects_bad_taps_mode():
    p, ba, ma = _setup()
    with pytest.raises(ValueError, match='soft_taps'):
        count2_soft(p, p, battrs=ba, mattrs=ma, soft_taps='turbo', backend='cuda')


@needs_cuda
def test_rejects_an_unsupported_second_axis():
    """Only s and a pole axis are softened; anything else must be refused.

    A real ``mu`` axis is a hard-binned coordinate the soft kernel does not
    serve, and quietly hard-binning it would be worse than refusing.
    """
    p, ba, ma = _setup()
    ba2 = BinAttrs(s=EDGES, mu=(np.linspace(-1, 1, 9), 'z'))
    ma2 = MeshAttrs(p, p, boxsize=BOX, battrs=ba2, periodic=True)
    with pytest.raises(ValueError, match='not served'):
        count2_soft(p, p, battrs=ba2, mattrs=ma2, backend='cuda')


@needs_cuda
def test_rejects_bad_cutoff():
    p, ba, ma = _setup()
    for cutoff in (0.0, 1.0, 2.0, -1e-3):
        with pytest.raises(ValueError, match='soft_cutoff'):
            count2_soft(p, p, battrs=ba, mattrs=ma, soft_cutoff=cutoff,
                        backend='cuda')


@needs_cuda
def test_rejects_non_cuda_backend():
    p, ba, ma = _setup()
    with pytest.raises(NotImplementedError, match='CUDA-only'):
        count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')


def test_soft_weights_is_a_normalised_softmax():
    s = np.array([5.0, 50.0, 99.0])
    w = soft_weights(s, EDGES, sigma=2.0)
    assert w.shape == (s.size, EDGES.size - 1)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-14)
    assert (w >= 0).all()
    # each row peaks in the bin holding s
    centres = _bin_centres(EDGES)
    np.testing.assert_array_equal(w.argmax(axis=1),
                                  np.abs(s[:, None] - centres[None, :]).argmin(axis=1))


def test_soft_weights_rejects_bad_sigma():
    with pytest.raises(ValueError, match='positive'):
        soft_weights([10.0], EDGES, sigma=0.0)


def test_is_uniform_recognises_arithmetic_progressions():
    assert is_uniform(EDGES)
    assert is_uniform(np.arange(0.0, 10.0))
    assert not is_uniform(LOG_EDGES)
    assert not is_uniform(np.array([1.0, 2.0, 4.0, 8.0]))
    # A grid uniform to 1e-12 relative counts as uniform; kernel and frontend must
    # agree, otherwise they take different paths for the same request.
    nearly = EDGES + np.linspace(0.0, 1e-10, EDGES.size)
    assert is_uniform(nearly)


def test_sigmas_scales_with_the_local_bin_width():
    """None means half a bin, per bin; a number is one absolute width."""
    np.testing.assert_allclose(sigmas(EDGES), 0.5 * DS, rtol=1e-14)
    widths = np.diff(LOG_EDGES)
    np.testing.assert_allclose(sigmas(LOG_EDGES), 0.5 * widths, rtol=1e-14)
    np.testing.assert_allclose(sigmas(LOG_EDGES, 4.0), 4.0, rtol=0)
    assert np.all(np.diff(sigmas(LOG_EDGES)) > 0)


def test_sigmas_rejects_non_positive_sigma():
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match='sigma'):
            sigmas(EDGES, bad)


def test_default_sigma_is_half_the_mean_bin_width():
    assert default_sigma(EDGES) == pytest.approx(0.5 * DS)
