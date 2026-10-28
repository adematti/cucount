"""CUDA tests for the differentiable soft-s jax frontend.

Two independent references, as in the rest of the suite:

* ``cucount.numpy.soft.count2_soft`` -- the validated CUDA forward -- for counts;
* central finite differences of that same forward, for the adjoint.

The jax frontend only exists when jax can see a GPU (the FFI targets are
registered for the CUDA platform), so these skip elsewhere.
"""

import numpy as np
import pytest

from cucount.numpy import BinAttrs, MeshAttrs, Particles
from cucount.numpy import soft as nsoft

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)  # counts and positions are float64

from cucount import jax as cjax  # noqa: E402


def _gpu_devices():
    try:
        return [d for d in jax.devices() if d.platform == 'gpu']
    except Exception:  # pragma: no cover - jax without a working plugin
        return []


needs_soft_jax = pytest.mark.skipif(
    not _gpu_devices() or not hasattr(cjax, 'count2_soft'),
    reason='the soft jax frontend needs jax on a GPU and the CUDA FFI module')

BOX = 300.0
SMAX = 100.0
EDGES = np.linspace(1.0, 100.0, 21)
SIGMA = 2.0
CUTOFF = 1e-13
N = 64


def _setup(seed=7, n=N, weighted=True):
    rng = np.random.default_rng(seed)
    pos1 = rng.uniform(0., BOX, (n, 3))
    pos2 = rng.uniform(0., BOX, (n, 3))
    w1 = rng.uniform(0.5, 1.5, n) if weighted else None
    w2 = rng.uniform(0.5, 1.5, n) if weighted else None
    battrs = BinAttrs(s=EDGES)
    p1 = Particles(pos1, w1)
    p2 = Particles(pos2, w2)
    mattrs = MeshAttrs(p1, p2, boxsize=BOX, battrs=battrs, periodic=True)
    return pos1, pos2, w1, w2, battrs, mattrs


def _reference_counts(pos1, pos2, w1, w2, battrs, mattrs, sigma=SIGMA):
    p1 = Particles(np.asarray(pos1), None if w1 is None else np.asarray(w1))
    p2 = Particles(np.asarray(pos2), None if w2 is None else np.asarray(w2))
    return nsoft.count2_soft(p1, p2, battrs=battrs, soft_sigma=sigma, soft_cutoff=CUTOFF,
                             mattrs=mattrs, backend='cuda')['weight']


def _jax_counts(pos1, w1, pos2, w2, battrs, mattrs, sigma=SIGMA):
    # mattrs is passed explicitly: MeshAttrs cannot infer box extents from traced
    # positions, the same restriction as cjax.count2.
    p1 = cjax.Particles(pos1, w1)
    p2 = cjax.Particles(pos2, w2)
    return cjax.count2_soft(p1, p2, battrs=battrs, soft_sigma=sigma, soft_cutoff=CUTOFF,
                            mattrs=mattrs, backend='cuda')['weight']


@needs_soft_jax
def test_counts_match_numpy_frontend():
    pos1, pos2, w1, w2, battrs, mattrs = _setup()
    expected = _reference_counts(pos1, pos2, w1, w2, battrs, mattrs)
    got = np.asarray(_jax_counts(pos1, w1, pos2, w2, battrs, mattrs))
    assert got.shape == expected.shape == (EDGES.size - 1,)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@needs_soft_jax
def test_position_gradient_matches_finite_difference():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=11)
    rng = np.random.default_rng(3)
    cot = rng.normal(size=EDGES.size - 1)

    def loss(p1, p2):
        return np.sum(_reference_counts(p1, p2, w1, w2, battrs, mattrs) * cot)

    gpos1, gpos2 = jax.grad(lambda a, b: (_jax_counts(a, w1, b, w2, battrs, mattrs) * cot).sum(),
                            argnums=(0, 1))(pos1, pos2)

    h = 1e-5
    fd1 = np.zeros_like(pos1)
    fd2 = np.zeros_like(pos2)
    for i in range(pos1.shape[0]):
        for d in range(3):
            plus, minus = pos1.copy(), pos1.copy()
            plus[i, d] += h
            minus[i, d] -= h
            fd1[i, d] = (loss(plus, pos2) - loss(minus, pos2)) / (2 * h)
    for i in range(pos2.shape[0]):
        for d in range(3):
            plus, minus = pos2.copy(), pos2.copy()
            plus[i, d] += h
            minus[i, d] -= h
            fd2[i, d] = (loss(pos1, plus) - loss(pos1, minus)) / (2 * h)

    scale = np.abs(fd1).max()
    assert np.abs(np.asarray(gpos1) - fd1).max() <= 2e-5 * scale + 1e-9
    scale = np.abs(fd2).max()
    assert np.abs(np.asarray(gpos2) - fd2).max() <= 2e-5 * scale + 1e-9


@needs_soft_jax
def test_weight_gradient_matches_finite_difference():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=13)
    rng = np.random.default_rng(5)
    cot = rng.normal(size=EDGES.size - 1)

    def loss(a, b):
        return np.sum(_reference_counts(pos1, pos2, a, b, battrs, mattrs) * cot)

    gw1, gw2 = jax.grad(lambda a, b: (_jax_counts(pos1, a, pos2, b, battrs, mattrs) * cot).sum(),
                        argnums=(0, 1))(w1, w2)

    h = 1e-5
    fd1 = np.array([(loss(w1 + h * e, w2) - loss(w1 - h * e, w2)) / (2 * h)
                    for e in np.eye(w1.size)])
    fd2 = np.array([(loss(w1, w2 + h * e) - loss(w1, w2 - h * e)) / (2 * h)
                    for e in np.eye(w2.size)])
    np.testing.assert_allclose(np.asarray(gw1).ravel(), fd1, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gw2).ravel(), fd2, rtol=2e-5, atol=1e-6)


@needs_soft_jax
def test_autocorrelation_gradient_is_symmetric():
    """One Particles: the sampled derivative must not privilege the first slot."""
    pos, _, w, _, battrs, _ = _setup(seed=17)
    part = Particles(pos, w)
    mattrs = MeshAttrs(part, part, boxsize=BOX, battrs=battrs, periodic=True)
    rng = np.random.default_rng(9)
    cot = rng.normal(size=EDGES.size - 1)

    def loss(p):
        part = Particles(np.asarray(p), w)
        return np.sum(nsoft.count2_soft(part, part, battrs=battrs, soft_sigma=SIGMA,
                                        soft_cutoff=CUTOFF, mattrs=mattrs,
                                        backend='cuda')['weight'] * cot)

    grad = jax.grad(lambda p: (_jax_counts(p, w, p, w, battrs, mattrs) * cot).sum())(pos)

    h = 1e-5
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[i, d] += h
            minus[i, d] -= h
            fd[i, d] = (loss(plus) - loss(minus)) / (2 * h)

    scale = np.abs(fd).max()
    assert np.abs(np.asarray(grad).reshape(fd.shape) - fd).max() <= 2e-5 * scale + 1e-9
    # Translation invariance in a periodic box: the gradient over all particles cancels.
    assert np.abs(np.asarray(grad).sum(axis=0)).max() <= 1e-9 * scale


@needs_soft_jax
def test_counts_conserve_total_weight_and_jit():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=19)
    jitted = jax.jit(lambda a, c: _jax_counts(a, w1, c, w2, battrs, mattrs))
    got = np.asarray(jitted(pos1, pos2))
    np.testing.assert_allclose(got.sum(), _reference_counts(pos1, pos2, w1, w2, battrs, mattrs).sum(),
                               rtol=1e-12)
    np.testing.assert_allclose(got, np.asarray(_jax_counts(pos1, w1, pos2, w2, battrs, mattrs)),
                               rtol=1e-12, atol=1e-12)


@needs_soft_jax
def test_unweighted_input_matches_unit_weights():
    """Particles without weights get a ones column; the FFI arity relies on it."""
    pos1, pos2, _, _, battrs, mattrs = _setup(seed=23, weighted=False)
    p1, p2 = cjax.Particles(pos1), cjax.Particles(pos2)
    got = np.asarray(cjax.count2_soft(p1, p2, battrs=battrs, soft_sigma=SIGMA,
                                      soft_cutoff=CUTOFF, mattrs=mattrs,
                                      backend='cuda')['weight'])
    expected = _reference_counts(pos1, pos2, np.ones(pos1.shape[0]), np.ones(pos2.shape[0]),
                                 battrs, mattrs)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@needs_soft_jax
def test_non_uniform_edges_match_numpy_and_gradient():
    """A log grid goes through the general path in the FFI too, forward and adjoint.

    ``soft_sigma=None`` there means half of each bin, so the smoothed kernel width
    is a property of the binning and ``jax.grad`` only has positions/weights to
    differentiate.
    """
    edges = np.geomspace(1.0, 100.0, 9)
    rng = np.random.default_rng(31)
    n = 32
    pos1 = rng.uniform(0., BOX, (n, 3))
    pos2 = rng.uniform(0., BOX, (n, 3))
    w1 = rng.uniform(0.5, 1.5, n)
    w2 = rng.uniform(0.5, 1.5, n)
    battrs = BinAttrs(s=edges)
    p1, p2 = Particles(pos1, w1), Particles(pos2, w2)
    mattrs = MeshAttrs(p1, p2, boxsize=BOX, battrs=battrs, periodic=True)

    expected = _reference_counts(pos1, pos2, w1, w2, battrs, mattrs, sigma=None)
    got = np.asarray(_jax_counts(pos1, w1, pos2, w2, battrs, mattrs, sigma=None))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    cot = rng.normal(size=edges.size - 1)
    grad = jax.grad(
        lambda a, b: (_jax_counts(a, w1, b, w2, battrs, mattrs, sigma=None) * cot).sum(),
        argnums=(0, 1))(pos1, pos2)

    h = 1e-5
    gpos = np.asarray(grad[0])
    for i in rng.choice(n, size=3, replace=False):
        for d in range(3):
            plus, minus = pos1.copy(), pos1.copy()
            plus[i, d] += h
            minus[i, d] -= h
            fd = (np.dot(_reference_counts(plus, pos2, w1, w2, battrs, mattrs, sigma=None), cot)
                  - np.dot(_reference_counts(minus, pos2, w1, w2, battrs, mattrs, sigma=None), cot)) / (2 * h)
            assert fd == pytest.approx(gpos[i, d], rel=2e-4, abs=1e-6)


def _n_devices():
    return len(_gpu_devices())


needs_two_gpus = pytest.mark.skipif(
    _n_devices() < 2 or not hasattr(cjax, 'count2_soft'),
    reason='the sharded soft path needs at least two GPUs')


@needs_two_gpus
def test_sharded_counts_and_gradients_match_single_device():
    """Under ``create_sharding_mesh`` the soft path splits and sums like count2.

    Catalogue 1 is split over the mesh axis and catalogue 2 replicated; each shard
    counts its own slice against the whole of catalogue 2 and the partial
    histograms are summed, so the sharded answer must equal the single-device one.
    The adjoint runs the same way: catalogue 1's gradient is per-shard, the
    replicated catalogue's partial gradients are summed back with ``psum``.
    """
    ndev = _n_devices()
    n = 8 * ndev  # divisible by the mesh size, as shard_map requires
    rng = np.random.default_rng(53)
    pos = rng.uniform(0., BOX, (n, 3))
    w = rng.uniform(0.5, 1.5, n)
    battrs = BinAttrs(s=EDGES)
    p = Particles(pos, w)
    mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True)
    cot = rng.normal(size=EDGES.size - 1)

    ref = np.asarray(_reference_counts(pos, pos, w, w, battrs, mattrs))
    vjp = nsoft.count2_soft_vjp(
        Particles(pos, w), Particles(pos, w), battrs=battrs, cotangent=cot,
        soft_sigma=SIGMA, soft_cutoff=CUTOFF, mattrs=mattrs, backend='cuda')
    ref_gpos = np.asarray(vjp['grad_position1']) + np.asarray(vjp['grad_position2'])
    ref_gw = (np.asarray(vjp['grad_weight1']).reshape(-1)
              + np.asarray(vjp['grad_weight2']).reshape(-1))

    with cjax.create_sharding_mesh() as mesh:
        assert mesh.devices.size == ndev
        pj = cjax.Particles(pos, w)

        def loss(pp):
            counts = cjax.count2_soft(pp, pp, battrs=battrs, soft_sigma=SIGMA,
                                      soft_cutoff=CUTOFF, mattrs=mattrs,
                                      backend='cuda')['weight']
            return (counts * cot).sum()

        value, grad = jax.value_and_grad(loss)(pj)

    np.testing.assert_allclose(float(value), float(np.dot(ref, cot)), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(grad.positions), ref_gpos,
                               rtol=1e-12, atol=1e-9 * np.abs(ref_gpos).max())
    np.testing.assert_allclose(np.asarray(grad.values[0]).reshape(-1), ref_gw,
                               rtol=1e-12, atol=1e-9 * np.abs(ref_gw).max())
