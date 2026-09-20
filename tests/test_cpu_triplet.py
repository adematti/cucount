"""Correctness of the factorized triplet counts on the CPU backend.

The oracle is a direct triple sum over (i1, i2, i3). That is deliberately not
the shape the backend computes: count3 histograms each leg separately and
contracts the two projections over m, so summing the triplets explicitly
checks the factorization and the angle-addition identity behind it, not just
the port. The CUDA backend is used as a second oracle where it is built.
"""

import numpy as np
import pytest
from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           cpu_available, cpulib, count3)

pytestmark = pytest.mark.skipif(
    not cpu_available(), reason='CPU backend not built (-DCUCOUNT_BUILD_CPU=ON)')


def _cuda_available():
    try:
        import cucountlib.cuda  # noqa: F401
        return True
    except ImportError:
        return False


CUDA = _cuda_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason='CUDA extension not built')

E12 = np.linspace(10., 400., 5)
E13 = np.linspace(10., 400., 5)


def sky_catalog(seed, n, rmin=1800., rmax=2200.):
    rng = np.random.default_rng(seed)
    ct = rng.uniform(-1., 1., n)
    phi = rng.uniform(0., 2 * np.pi, n)
    st = np.sqrt(1. - ct**2)
    r = rng.uniform(rmin, rmax, n)
    pos = np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])
    return pos, rng.uniform(0.5, 1.5, n)


def pbar(ell, m, x):
    """Normalized associated Legendre Pbar_ell^m, the closed forms up to ell = 5."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    s2 = np.maximum(0., 1. - x2)
    s = np.sqrt(s2)
    table = {
        (0, 0): lambda: np.ones_like(x),
        (1, 0): lambda: x,
        (1, 1): lambda: -0.70710678118654752440 * s,
        (2, 0): lambda: 0.5 * (3. * x2 - 1.),
        (2, 1): lambda: -1.22474487139158904910 * x * s,
        (2, 2): lambda: 0.61237243569579452455 * s2,
        (3, 0): lambda: 0.5 * (5. * x2 * x - 3. * x),
        (3, 1): lambda: -0.43301270189221932338 * (5. * x2 - 1.) * s,
        (3, 2): lambda: 1.36930639376291527536 * x * s2,
        (3, 3): lambda: -0.55901699437494742410 * s2 * s,
        (4, 0): lambda: 0.125 * (35. * x2 * x2 - 30. * x2 + 3.),
        (4, 1): lambda: -0.55901699437494742410 * x * (7. * x2 - 3.) * s,
        (4, 2): lambda: 0.39528470752104741743 * (7. * x2 - 1.) * s2,
        (4, 3): lambda: -0.93541434669348534640 * x * s2 * s,
        (4, 4): lambda: 0.52291251658379721705 * s2 * s2,
    }
    return table[(ell, m)]()


def build_los_frame(r1hat, los='firstpoint'):
    """The (ez, ex, ey) frame the projection uses, per build_los_frame."""
    if los == 'x':
        return np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    if los == 'y':
        return np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    if los == 'z':
        return np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    # firstpoint / endpoint / midpoint: an orthonormal frame with ez = r1hat
    ref = np.array([0., 0., 1.]) if abs(r1hat[2]) < 0.9 else np.array([1., 0., 0.])
    ez = r1hat
    tmp = ref - ref.dot(ez) * ez
    ex = tmp / np.linalg.norm(tmp)
    ey = np.cross(ez, ex)
    return np.array([ez, ex, ey])


def leg_geometry(pos1, frame, pos2):
    """(r, mu, phi) of every secondary seen from a primary in its frame."""
    diff = pos2 - pos1
    r = np.linalg.norm(diff, axis=-1)
    rhat = np.where(r[:, None] > 0., diff / np.where(r[:, None] > 0., r[:, None], 1.),
                    frame[0][None, :])
    mu = np.clip(rhat @ frame[0], -1., 1.)
    x = rhat @ frame[1]
    y = rhat @ frame[2]
    rho = np.sqrt(np.maximum(0., x * x + y * y))
    cphi = np.where(rho > 1e-12, x / np.where(rho > 1e-12, rho, 1.), 1.)
    sphi = np.where(rho > 1e-12, y / np.where(rho > 1e-12, rho, 1.), 0.)
    return r, mu, np.arctan2(sphi, cphi)


def _bin_index(v, edges):
    idx = np.searchsorted(edges, v, side='right') - 1
    return idx, (v >= edges[0]) & (v < edges[-1])


def proj_slots(ells1, ells2):
    """(ell1, ell2, m, is_imaginary, slot) for the flattened projection axis."""
    out, iproj = [], 0
    for ell1 in ells1:
        for ell2 in ells2:
            mmax = min(ell1, ell2)
            out.append((ell1, ell2, 0, False, iproj))
            for m in range(1, mmax + 1):
                out.append((ell1, ell2, m, False, iproj + m))
                out.append((ell1, ell2, m, True, iproj + mmax + m))
            iproj += 2 * mmax + 1
    return out, iproj


def triplet_reference(cats, e12, e13, ells=None, los='firstpoint'):
    """Direct sum over every (i1, i2, i3), with no factorization."""
    (pos1, w1), (pos2, w2), (pos3, w3) = cats
    nb12, nb13 = len(e12) - 1, len(e13) - 1

    if ells is None:
        out = np.zeros((nb12, nb13))
    else:
        slots, nprojs = proj_slots(*ells)
        out = np.zeros((nb12, nb13, nprojs))

    r1hat = pos1 / np.linalg.norm(pos1, axis=-1, keepdims=True)

    for i1 in range(len(pos1)):
        frame = build_los_frame(r1hat[i1], los)
        r2, mu2, phi2 = leg_geometry(pos1[i1], frame, pos2)
        r3, mu3, phi3 = leg_geometry(pos1[i1], frame, pos3)
        b2, ok2 = _bin_index(r2, e12)
        b3, ok3 = _bin_index(r3, e13)

        # Every (i2, i3) combination, weighted; no per-leg histogram.
        ww = w1[i1] * w2[:, None] * w3[None, :]
        pair_ok = ok2[:, None] & ok3[None, :]
        flat = np.where(pair_ok, b2[:, None] * nb13 + b3[None, :], 0)

        if ells is None:
            out += np.bincount(flat[pair_ok], weights=ww[pair_ok],
                               minlength=nb12 * nb13).reshape(nb12, nb13)
            continue

        dphi = phi2[:, None] - phi3[None, :]
        for ell1, ell2, m, imag, slot in slots:
            norm = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1))
            ang = np.sin(m * dphi) if imag else np.cos(m * dphi)
            term = (ww * pbar(ell1, m, mu2)[:, None] * pbar(ell2, m, mu3)[None, :]
                    * ang * norm)
            out[..., slot] += np.bincount(
                flat[pair_ok], weights=term[pair_ok],
                minlength=nb12 * nb13).reshape(nb12, nb13)
    return out


def _setup(n=60, seeds=(1, 2, 3), ells=None, los='firstpoint'):
    cats = [sky_catalog(s, n) for s in seeds]
    particles = [Particles(*c) for c in cats]
    if ells is None:
        b12, b13 = BinAttrs(s=E12), BinAttrs(s=E13)
    else:
        b12 = BinAttrs(s=E12, pole=(np.array(ells[0]), los))
        b13 = BinAttrs(s=E13, pole=(np.array(ells[1]), los))
    # One box for every leg, so the candidate window cannot clip a valid pair
    # and make the all-pairs reference disagree for a reason that is not a bug.
    kw = dict(battrs12=b12, battrs13=b13,
              mattrs1=MeshAttrs(*particles, battrs=b12),
              mattrs2=MeshAttrs(*particles, battrs=b12),
              mattrs3=MeshAttrs(*particles, battrs=b13))
    return particles, kw, cats


def test_plain_matches_triplet_sum():
    particles, kw, cats = _setup()
    got = count3(*particles, backend='cpu', **kw)['weight']
    want = triplet_reference(cats, E12, E13)
    assert np.any(want), 'the reference must not be empty'
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-9, atol=0)


@pytest.mark.parametrize('ells', [([0, 2], [0, 2]), ([0], [0]), ([1, 2], [0, 3])])
def test_poles_match_triplet_sum(ells):
    """The m-contraction must reproduce the explicit cos(m dphi) sum."""
    particles, kw, cats = _setup(ells=ells)
    got = count3(*particles, backend='cpu', **kw)['weight']
    want = triplet_reference(cats, E12, E13, ells=ells)
    assert got.shape == want.shape
    scale = np.abs(want).max()
    assert scale > 0, 'the reference must not be empty'
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


@pytest.mark.parametrize('los', ['z', 'x', 'firstpoint'])
def test_pole_los_matches_triplet_sum(los):
    ells = ([0, 2], [0, 2])
    particles, kw, cats = _setup(ells=ells, los=los)
    got = count3(*particles, backend='cpu', **kw)['weight']
    want = triplet_reference(cats, E12, E13, ells=ells, los=los)
    scale = np.abs(want).max()
    assert scale > 0, 'the reference must not be empty'
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


def test_selection_removes_triplets():
    """An s selection on one leg must change the answer, and match the oracle."""
    particles, kw, cats = _setup()
    full = count3(*particles, backend='cpu', **kw)['weight']
    sattrs12 = SelectionAttrs(s=(10., 200.))
    cut = count3(*particles, backend='cpu', sattrs12=sattrs12, **kw)['weight']
    assert not np.allclose(full, cut), 'the selection removed nothing'

    (pos1, w1), (pos2, w2), (pos3, w3) = cats
    keep = [(pos1, w1), (pos2, w2), (pos3, w3)]
    # The oracle applies the same cut by zeroing out-of-range (1, 2) pairs.
    nb12, nb13 = len(E12) - 1, len(E13) - 1
    want = np.zeros((nb12, nb13))
    r1hat = pos1 / np.linalg.norm(pos1, axis=-1, keepdims=True)
    for i1 in range(len(pos1)):
        frame = build_los_frame(r1hat[i1])
        r2, _, _ = leg_geometry(pos1[i1], frame, pos2)
        r3, _, _ = leg_geometry(pos1[i1], frame, pos3)
        b2, ok2 = _bin_index(r2, E12)
        b3, ok3 = _bin_index(r3, E13)
        ok2 = ok2 & (r2 >= 10.) & (r2 <= 200.)
        ww = w1[i1] * w2[:, None] * w3[None, :]
        pair_ok = ok2[:, None] & ok3[None, :]
        flat = np.where(pair_ok, b2[:, None] * nb13 + b3[None, :], 0)
        want.ravel()[:] += np.bincount(flat[pair_ok], weights=ww[pair_ok],
                                       minlength=nb12 * nb13)
    assert np.allclose(cut, want, rtol=1e-9, atol=0)
    del keep


def test_thread_count_invariance():
    particles, kw, _ = _setup(n=80, ells=([0, 2], [0, 2]))
    one = count3(*particles, backend='cpu', tuning={'nthreads': 1}, **kw)['weight']
    many = count3(*particles, backend='cpu', tuning={'nthreads': 8}, **kw)['weight']
    assert np.allclose(one, many, rtol=1e-12, atol=0)


UNSUPPORTED3 = ['rp binning', 'pole on one leg', 'ell above 5']


def _unsupported3(feature):
    particles = [Particles(*sky_catalog(s, 60)) for s in (1, 2, 3)]
    b12, b13 = BinAttrs(s=E12), BinAttrs(s=E13)
    if feature == 'rp binning':
        b12 = BinAttrs(rp=(E12, 'z'))
        match = 'rp'
    elif feature == 'pole on one leg':
        b12 = BinAttrs(s=E12, pole=(np.array([0, 2]), 'firstpoint'))
        match = 'multipole axis on one'
    else:
        b12 = BinAttrs(s=E12, pole=(np.array([0, 6]), 'firstpoint'))
        b13 = BinAttrs(s=E13, pole=(np.array([0, 6]), 'firstpoint'))
        match = 'ell'
    return particles, dict(battrs12=b12, battrs13=b13), match


@pytest.mark.parametrize('feature', UNSUPPORTED3)
def test_unsupported_triplets_are_declined(feature):
    particles, kw, match = _unsupported3(feature)
    with pytest.raises((NotImplementedError, AssertionError, ValueError), match=match):
        count3(*particles, backend='cpu', **kw)


@needs_cuda
@pytest.mark.parametrize('ells', [None, ([0, 2], [0, 2])])
def test_matches_cuda(ells):
    particles, kw, _ = _setup(n=120, ells=ells)
    count3(*particles, backend='compare', **kw)


@needs_cuda
def test_matches_cuda_with_selection():
    particles, kw, _ = _setup(n=120)
    count3(*particles, backend='compare', sattrs12=SelectionAttrs(s=(10., 200.)),
           veto13=SelectionAttrs(s=(0., 50.)), **kw)


# ---------------------------------------------------------------------------
# Close triplet counts: every (1, 2, 3) triplet formed and binned, with an
# optional (2, 3) axis and the 3-dimensional angular upweight. The oracle is
# again the direct triple sum.
# ---------------------------------------------------------------------------

from cucount.numpy import WeightAttrs, count3close  # noqa: E402

E23 = np.linspace(10., 600., 4)


def _costhetas(pos1, pos2, pos3, i1, i2, i3):
    """cos(theta) of the three legs, as add_weight3 orders them."""
    def unit(p):
        return p / np.linalg.norm(p, axis=-1, keepdims=True)
    u1, u2, u3 = unit(pos1)[i1], unit(pos2)[i2], unit(pos3)[i3]
    return np.array([u1 @ u2, u1 @ u3, u2 @ u3])


def close_reference(cats, e12, e13, e23=None, ells=None, los='firstpoint',
                    angular=None):
    """Direct sum over every (i1, i2, i3), with no search structure at all."""
    (pos1, w1), (pos2, w2), (pos3, w3) = cats
    nb12, nb13 = len(e12) - 1, len(e13) - 1
    nb23 = len(e23) - 1 if e23 is not None else 0

    shape = [nb12, nb13] + ([nb23] if e23 is not None else [])
    if ells is not None and e23 is None:
        slots, nprojs = proj_slots(*ells)
        out = np.zeros(shape + [nprojs])
    else:
        slots = None
        out = np.zeros(shape)

    r1hat = pos1 / np.linalg.norm(pos1, axis=-1, keepdims=True)

    for i1 in range(len(pos1)):
        frame = build_los_frame(r1hat[i1], los)
        r2, mu2, phi2 = leg_geometry(pos1[i1], frame, pos2)
        r3, mu3, phi3 = leg_geometry(pos1[i1], frame, pos3)
        b2, ok2 = _bin_index(r2, e12)
        b3, ok3 = _bin_index(r3, e13)

        for i2 in range(len(pos2)):
            if not ok2[i2]:
                continue
            for i3 in range(len(pos3)):
                if not ok3[i3]:
                    continue
                w = w1[i1] * w2[i2] * w3[i3]

                if angular is not None:
                    ct = _costhetas(pos1, pos2, pos3, i1, i2, i3)
                    deg = np.degrees(np.arccos(np.clip(ct, -1., 1.)))
                    w *= angular(tuple(deg))

                if e23 is not None:
                    r23 = np.linalg.norm(pos3[i3] - pos2[i2])
                    ib23, ok23 = _bin_index(np.array([r23]), e23)
                    if not ok23[0]:
                        continue
                    out[b2[i2], b3[i3], ib23[0]] += w
                    continue

                if slots is None:
                    out[b2[i2], b3[i3]] += w
                    continue

                dphi = phi2[i2] - phi3[i3]
                for ell1, ell2, m, imag, slot in slots:
                    norm = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1))
                    # exp(+i m dphi), dphi = phi12 - phi13: the phase of
                    # Y_{ell1 m}(rhat12) conj(Y_{ell2 m}(rhat13)), which is
                    # what count3's m-contraction produces and what
                    # add_weight3 now matches.
                    ang = np.sin(m * dphi) if imag else np.cos(m * dphi)
                    out[b2[i2], b3[i3], slot] += (
                        w * norm * pbar(ell1, m, mu2[i2]) * pbar(ell2, m, mu3[i3]) * ang)
    return out


def _close_setup(n=35, seeds=(11, 12, 13), ells=None, e23=None, los='firstpoint',
                 angular_mesh=False):
    """Objects for a close-triplet count, plus the raw catalogues.

    ``angular_mesh`` puts a wide theta selection on the (1, 2) leg so
    MeshAttrs builds an angular mesh for it. The CUDA count3close kernel walks
    the close pair's mesh with the angular candidate window unconditionally,
    so it needs that; this backend dispatches on the mesh's own type and is
    happy either way. The selection is wide enough to remove nothing, so the
    two setups must give the same answer.
    """
    cats = [sky_catalog(s, n) for s in seeds]
    particles = [Particles(*c) for c in cats]
    if ells is None:
        b12, b13 = BinAttrs(s=E12), BinAttrs(s=E13)
    else:
        b12 = BinAttrs(s=E12, pole=(np.array(ells[0]), los))
        b13 = BinAttrs(s=E13, pole=(np.array(ells[1]), los))
    b23 = BinAttrs(s=e23) if e23 is not None else None
    sattrs12 = SelectionAttrs(theta=(0., 180.)) if angular_mesh else None
    kw = dict(battrs12=b12, battrs13=b13, battrs23=b23,
              mattrs1=MeshAttrs(*particles, battrs=b12, sattrs=sattrs12),
              mattrs2=MeshAttrs(*particles, battrs=b12, sattrs=sattrs12),
              mattrs3=MeshAttrs(*particles, battrs=b13))
    if sattrs12 is not None:
        kw['sattrs12'] = sattrs12
    return particles, kw, cats


def test_close_plain_matches_triplet_sum():
    particles, kw, cats = _close_setup()
    got = count3close(*particles, backend='cpu', **kw)['weight']
    want = close_reference(cats, E12, E13)
    assert np.any(want), 'the reference must not be empty'
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-9, atol=0)


def test_close_with_third_axis_matches_triplet_sum():
    """A (2, 3) axis switches off the projection and adds a third bin axis."""
    particles, kw, cats = _close_setup(e23=E23)
    got = count3close(*particles, backend='cpu', **kw)['weight']
    want = close_reference(cats, E12, E13, e23=E23)
    assert np.any(want), 'the reference must not be empty'
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-9, atol=0)


@pytest.mark.parametrize('ells', [([0, 2], [0, 2]), ([1], [1])])
def test_close_poles_match_triplet_sum(ells):
    particles, kw, cats = _close_setup(ells=ells)
    got = count3close(*particles, backend='cpu', **kw)['weight']
    want = close_reference(cats, E12, E13, ells=ells)
    assert got.shape == want.shape
    scale = np.abs(want).max()
    assert scale > 0, 'the reference must not be empty'
    assert np.allclose(got, want, rtol=1e-8, atol=1e-11 * scale)


def test_close_3d_angular_weights_match_triplet_sum():
    """The 3-dimensional angular table, indexed by the triangle's three angles."""
    particles, kw, cats = _close_setup()
    sep = np.linspace(0., 180., 7)
    rng = np.random.default_rng(7)
    table = rng.uniform(0.5, 1.5, (sep.size, sep.size, sep.size))
    wattrs = WeightAttrs(angular=dict(sep=[sep, sep, sep], weight=table))

    got = count3close(*particles, backend='cpu', wattrs=wattrs, **kw)['weight']
    plain = count3close(*particles, backend='cpu', **kw)['weight']
    assert not np.allclose(got, plain), 'the angular table changed nothing'

    want = close_reference(cats, E12, E13, angular=wattrs.angular)
    assert np.any(want), 'the reference must not be empty'
    assert np.allclose(got, want, rtol=1e-9, atol=0)


def test_close_thread_count_invariance():
    particles, kw, _ = _close_setup(n=40, ells=([0, 2], [0, 2]))
    one = count3close(*particles, backend='cpu', tuning={'nthreads': 1}, **kw)['weight']
    many = count3close(*particles, backend='cpu', tuning={'nthreads': 8}, **kw)['weight']
    assert np.allclose(one, many, rtol=1e-12, atol=0)


@pytest.mark.parametrize('close_pair', [(1, 2), (1, 3), (2, 3)])
def test_close_pair_choice_does_not_change_the_result(close_pair):
    """close_pair is a performance hint; every choice must give one answer."""
    particles, kw, _ = _close_setup()
    ref = count3close(*particles, backend='cpu', close_pair=(1, 2), **kw)['weight']
    got = count3close(*particles, backend='cpu', close_pair=close_pair, **kw)['weight']
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize('close_pair', [(1, 2), (1, 3), (2, 3)])
def test_close_pair_choice_does_not_change_the_result_on_default_meshes(close_pair):
    """The same, with the meshes the frontend picks rather than explicit ones.

    This is the regression case. close_pair names a search strategy, and each
    strategy bounds a different leg, so the default meshes used to be sized
    for the close pair itself. This backend has one traversal, centred on
    particle 1, so a mesh sized for the (2, 3) leg left it sweeping a window
    narrower than the (1, 2) and (1, 3) separations it walks over: it dropped
    triplets instead of rejecting them and came back ~0.7% low. Every other
    close-triplet test here passes mattrs1/2/3 explicitly, which is what kept
    it hidden.

    Two things the catalogue has to supply for the bug to show at all. A
    selection tighter than the binning, or every strategy sizes its meshes
    from the same binning; and enough particles that the default mesh is
    finer than the shortfall, or the swept cells cover the missing window by
    accident. At n = 35 both fail and the reverted code still passes.
    """
    cats = [sky_catalog(s, 4000) for s in (11, 12, 13)]
    particles = [Particles(*c) for c in cats]
    kw = dict(battrs12=BinAttrs(s=E12), battrs13=BinAttrs(s=E13), battrs23=None)
    kw[f'sattrs{close_pair[0]:d}{close_pair[1]:d}'] = SelectionAttrs(s=(0., E12[len(E12) // 2]))
    ref = count3close(*particles, backend='cpu', close_pair=(1, 2), **kw)['weight']
    got = count3close(*particles, backend='cpu', close_pair=close_pair, **kw)['weight']
    assert np.any(ref), 'the comparison must not be between two empty results'
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_close_declines_a_mesh_too_narrow_for_the_leg_it_walks():
    """A hand-built mesh gets the check the defaults no longer need.

    Sizing mattrs2 for a short (2, 3) selection is exactly what the frontend
    used to do by itself. Walked over the (1, 2) leg it would drop triplets
    quietly, so it has to be refused rather than counted.
    """
    particles, kw, _ = _close_setup()
    narrow = SelectionAttrs(s=(0., E12[1]))
    kw['mattrs2'] = MeshAttrs(*particles, battrs=None, sattrs=narrow)
    with pytest.raises(ValueError, match='too narrow'):
        count3close(*particles, backend='cpu', close_pair=(2, 3),
                    sattrs23=narrow, **kw)


def test_close_mesh_type_does_not_change_the_result():
    """The mesh is an acceleration structure; cartesian and angular must agree."""
    particles, cartesian, _ = _close_setup()
    _, angular, _ = _close_setup(angular_mesh=True)
    assert cartesian['mattrs2'].type == 'cartesian'
    assert angular['mattrs2'].type == 'angular'
    a = count3close(*particles, backend='cpu', **cartesian)['weight']
    b = count3close(*particles, backend='cpu', **angular)['weight']
    assert np.any(a), 'the comparison must not be between two empty results'
    assert np.allclose(a, b, rtol=1e-9, atol=0)


@needs_cuda
@pytest.mark.parametrize('angular_mesh', [False, True])
@pytest.mark.parametrize('case', ['plain', 'third-axis', 'poles', 'angular3d'])
def test_close_matches_cuda(case, angular_mesh):
    """Both mesh types, on both backends.

    angular_mesh=False is the regression case: the CUDA kernel used to fix the
    close pair's candidate window to the angular one and exit the process on a
    cartesian mesh, so a close-triplet count with no theta selection anywhere
    could not run at all.
    """
    kwargs = {}
    if case == 'third-axis':
        particles, kw, _ = _close_setup(n=60, e23=E23, angular_mesh=angular_mesh)
    elif case == 'poles':
        particles, kw, _ = _close_setup(n=60, ells=([0, 2], [0, 2]),
                                        angular_mesh=angular_mesh)
    elif case == 'angular3d':
        particles, kw, _ = _close_setup(n=60, angular_mesh=angular_mesh)
        sep = np.linspace(0., 180., 7)
        rng = np.random.default_rng(7)
        kwargs['wattrs'] = WeightAttrs(
            angular=dict(sep=[sep, sep, sep],
                         weight=rng.uniform(0.5, 1.5, (sep.size,) * 3)))
    else:
        particles, kw, _ = _close_setup(n=60, angular_mesh=angular_mesh)
    count3close(*particles, backend='compare', **kw, **kwargs)


@needs_cuda
@pytest.mark.parametrize('close_pair', [(1, 2), (1, 3), (2, 3)])
def test_close_pair_choice_matches_cuda(close_pair):
    """Every close_pair, on a cartesian mesh: all three used to exit."""
    particles, kw, _ = _close_setup(n=60)
    count3close(*particles, backend='compare', close_pair=close_pair, **kw)


def test_count3_and_count3close_agree_on_every_coefficient():
    """count3 is the factorized form of count3close, so the two must agree.

    The imaginary coefficients used to come out exactly negated: add_weight3
    took sin(dphi) from the cross product of the two transverse parts the
    other way round, giving sin(phi13 - phi12), while count3's m-contraction
    forms Z12 conj(Z13) and so produces sin(phi12 - phi13). The real ones
    never showed it, cos being even. Both backends carried the flipped sign,
    so this test checks the two entry points really are interchangeable now,
    and the imaginary assertion is the one that regresses.
    """
    ells = ([0, 2], [0, 2])
    particles, kw, _ = _close_setup(n=40, ells=ells)
    factorized = count3(*particles, backend='cpu',
                        **{k: v for k, v in kw.items() if k != 'battrs23'})['weight']
    close = count3close(*particles, backend='cpu', **kw)['weight']

    slots, _ = proj_slots(*ells)
    checked_imag = 0
    for ell1, ell2, m, imag, slot in slots:
        a, b = factorized[..., slot], close[..., slot]
        scale = max(np.abs(a).max(), 1e-300)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12 * scale)
        if imag:
            assert np.abs(a).max() > 0, 'a vanishing coefficient proves nothing'
            checked_imag += 1
    assert checked_imag > 0, 'no imaginary coefficient was exercised'
