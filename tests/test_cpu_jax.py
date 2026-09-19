"""The JAX FFI entry points of the CPU backend, including multi-device.

The oracle is the numpy frontend on the same backend: the two go through
different bindings (pybind vs the FFI handler) but the same kernels, so any
disagreement is in the FFI plumbing -- the staged attrs, the value layout or
the output shape -- which is exactly what this file is for.

Multiple CPU devices need XLA told about them before jaxlib initializes, so
that happens at import time here; when another module got to jax first the
sharded test is skipped rather than run against one device.
"""

import os

# Must precede the first jax import. Appended, not assigned: the cosmodesi
# environment sets XLA_FLAGS already and dropping it breaks other things.
if 'xla_force_host_platform_device_count' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                               + ' --xla_force_host_platform_device_count=4')

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,  # noqa: E402
                           WeightAttrs, _cpu, count2, count3, count3close)

try:
    import cucount.jax as cj
    HAS_FFI_CPU = cj.ffi_cpulib is not None
except ImportError:
    cj = None
    HAS_FFI_CPU = False

pytestmark = pytest.mark.skipif(
    not (_cpu.available() and HAS_FFI_CPU),
    reason='the CPU backend and its JAX FFI module are both needed')

BOX = 1000.
SMAX = 100.
SEDGES = np.linspace(1., SMAX, 11)


def catalog(seed, n=600):
    rng = np.random.default_rng(seed)
    return rng.uniform(0., BOX, (n, 3)), rng.uniform(0.5, 1.5, n)


def sky_catalog(seed, n=80):
    rng = np.random.default_rng(seed)
    ct = rng.uniform(-1., 1., n)
    phi = rng.uniform(0., 2 * np.pi, n)
    st = np.sqrt(1. - ct**2)
    r = rng.uniform(1800., 2200., n)
    pos = np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])
    return pos, rng.uniform(0.5, 1.5, n)


CPU_DEVICES = jax.devices('cpu')


def on_cpu():
    """Place the computation on a jax CPU device.

    Without this the arrays land on whatever the default platform is, and on a
    machine with a GPU that is CUDA -- where the cpu FFI target is not
    registered, by design. XLA reports that as a missing handler rather than
    as a placement problem, so pin it explicitly.
    """
    return jax.default_device(CPU_DEVICES[0])


def cpu_mesh(n=None):
    """A sharding mesh over jax's CPU devices, whatever the default platform."""
    n = len(CPU_DEVICES) if n is None else n
    return jax.make_mesh((n,), ('x',), devices=CPU_DEVICES[:n])


def place(mesh, array, shard=True):
    """Put an array on the mesh, split along its first axis or replicated.

    Particles only places its arrays itself with exchange=True, which is the
    multi-process path; here the placement has to be explicit, and shard_map
    requires the argument's sharding to match the in_specs it is given.
    """
    array = jax.numpy.asarray(array)
    spec = (jax.sharding.PartitionSpec('x', *([None] * (array.ndim - 1))) if shard
            else jax.sharding.PartitionSpec())
    return jax.device_put(array, jax.sharding.NamedSharding(mesh, spec))


def close(got, want, rtol=1e-9):
    got = np.asarray(got)
    scale = np.abs(want).max()
    assert scale > 0, 'the reference must not be empty'
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=rtol, atol=1e-12 * scale)


@pytest.mark.parametrize('battrs_kind', ['s', 's-mu', 's-pole', 'rp-pi'])
def test_count2_ffi_matches_numpy(battrs_kind):
    """The vectorised kernel and the generic path, both through the FFI."""
    pos1, w1 = catalog(1)
    pos2, w2 = catalog(2)
    if battrs_kind == 's':
        battrs = BinAttrs(s=SEDGES)
    elif battrs_kind == 's-mu':
        battrs = BinAttrs(s=SEDGES, mu=(np.linspace(-1., 1., 5), 'midpoint'))
    elif battrs_kind == 's-pole':
        battrs = BinAttrs(s=SEDGES, pole=(np.array([0, 2]), 'firstpoint'))
    else:  # routes to the scalar generic path
        battrs = BinAttrs(rp=(np.linspace(1., 80., 6), 'z'),
                          pi=(np.linspace(-80., 80., 9), 'z'))

    nparticles = (Particles(pos1, w1), Particles(pos2, w2))
    mattrs = MeshAttrs(*nparticles, battrs=battrs)
    want = count2(*nparticles, battrs=battrs, mattrs=mattrs, backend='cpu')['weight']

    with on_cpu():
        jparticles = (cj.Particles(pos1, w1), cj.Particles(pos2, w2))
        got = cj.count2(*jparticles, battrs=battrs,
                        mattrs=cj.MeshAttrs(*jparticles, battrs=battrs),
                        backend="cpu")["weight"]
    close(got, want)


def test_count2_ffi_under_jit():
    pos1, w1 = catalog(3)
    pos2, w2 = catalog(4)
    battrs = BinAttrs(s=SEDGES)
    nparticles = (Particles(pos1, w1), Particles(pos2, w2))
    want = count2(*nparticles, battrs=battrs,
                  mattrs=MeshAttrs(*nparticles, battrs=battrs), backend='cpu')['weight']

    with on_cpu():
        jparticles = (cj.Particles(pos1, w1), cj.Particles(pos2, w2))
        mattrs = cj.MeshAttrs(*jparticles, battrs=battrs)
        # on_cpu() already pins the default device; jit's own device= is
        # deprecated.
        run = jax.jit(lambda: cj.count2(*jparticles, battrs=battrs, mattrs=mattrs,
                                        backend="cpu")["weight"])
        got = run()
    close(got, want)


def test_count2_ffi_with_selection_and_weights():
    """The staged attrs must survive the trip: selection, and a negative weight."""
    pos1, w1 = sky_catalog(5, n=300)
    pos2, w2 = sky_catalog(6, n=300)
    nw1 = np.random.default_rng(7).uniform(0., 0.1, len(w1))
    nw2 = np.random.default_rng(8).uniform(0., 0.1, len(w2))
    bits = [np.random.default_rng(9).integers(0, 0xffffffff, len(w1), dtype=np.uint64)]
    bits2 = [np.random.default_rng(10).integers(0, 0xffffffff, len(w2), dtype=np.uint64)]

    battrs = BinAttrs(s=np.linspace(10., 400., 9))
    sattrs = SelectionAttrs(theta=(0., 30.))
    nparticles = (Particles(pos1, [w1] + bits + [nw1]),
                  Particles(pos2, [w2] + bits2 + [nw2]))
    wattrs = WeightAttrs(bitwise=dict(weights=bits))
    mattrs = MeshAttrs(*nparticles, battrs=battrs, sattrs=sattrs)
    want = count2(*nparticles, battrs=battrs, mattrs=mattrs, sattrs=sattrs,
                  wattrs=wattrs, backend='cpu')['weight']

    with on_cpu():
        jparticles = (cj.Particles(pos1, [w1] + bits + [nw1]),
                      cj.Particles(pos2, [w2] + bits2 + [nw2]))
        got = cj.count2(*jparticles, battrs=battrs,
                        mattrs=cj.MeshAttrs(*jparticles, battrs=battrs, sattrs=sattrs),
                        sattrs=sattrs,
                        wattrs=cj.WeightAttrs(bitwise=dict(weights=bits)),
                        backend="cpu")["weight"]
    close(got, want)


@pytest.mark.parametrize('ells', [None, ([0, 2], [0, 2])])
def test_count3_ffi_matches_numpy(ells):
    cats = [sky_catalog(s, n=80) for s in (11, 12, 13)]
    e12 = np.linspace(10., 400., 5)
    if ells is None:
        b12, b13 = BinAttrs(s=e12), BinAttrs(s=e12)
    else:
        b12 = BinAttrs(s=e12, pole=(np.array(ells[0]), 'firstpoint'))
        b13 = BinAttrs(s=e12, pole=(np.array(ells[1]), 'firstpoint'))

    nparticles = [Particles(*c) for c in cats]
    mkw = dict(mattrs1=MeshAttrs(*nparticles, battrs=b12),
               mattrs2=MeshAttrs(*nparticles, battrs=b12),
               mattrs3=MeshAttrs(*nparticles, battrs=b13))
    want = count3(*nparticles, battrs12=b12, battrs13=b13, backend='cpu', **mkw)['weight']

    with on_cpu():
        jparticles = [cj.Particles(*c) for c in cats]
        jkw = dict(mattrs1=cj.MeshAttrs(*jparticles, battrs=b12),
                   mattrs2=cj.MeshAttrs(*jparticles, battrs=b12),
                   mattrs3=cj.MeshAttrs(*jparticles, battrs=b13))
        got = cj.count3(*jparticles, battrs12=b12, battrs13=b13, backend="cpu",
                        **jkw)["weight"]
    close(got, want)


@pytest.mark.parametrize('third', [False, True])
def test_count3close_ffi_matches_numpy(third):
    cats = [sky_catalog(s, n=40) for s in (14, 15, 16)]
    e12 = np.linspace(10., 400., 5)
    b12, b13 = BinAttrs(s=e12), BinAttrs(s=e12)
    b23 = BinAttrs(s=np.linspace(10., 600., 4)) if third else None

    nparticles = [Particles(*c) for c in cats]
    mkw = dict(mattrs1=MeshAttrs(*nparticles, battrs=b12),
               mattrs2=MeshAttrs(*nparticles, battrs=b12),
               mattrs3=MeshAttrs(*nparticles, battrs=b13))
    want = count3close(*nparticles, battrs12=b12, battrs13=b13, battrs23=b23,
                       backend='cpu', **mkw)['weight']

    with on_cpu():
        jparticles = [cj.Particles(*c) for c in cats]
        jkw = dict(mattrs1=cj.MeshAttrs(*jparticles, battrs=b12),
                   mattrs2=cj.MeshAttrs(*jparticles, battrs=b12),
                   mattrs3=cj.MeshAttrs(*jparticles, battrs=b13))
        got = cj.count3close(*jparticles, battrs12=b12, battrs13=b13, battrs23=b23,
                             backend="cpu", **jkw)["weight"]
    close(got, want)


def test_unknown_backend_is_rejected():
    pos, w = catalog(17, n=100)
    battrs = BinAttrs(s=SEDGES)
    p = (cj.Particles(pos, w), cj.Particles(pos, w))
    mattrs = cj.MeshAttrs(*p, battrs=battrs)
    with pytest.raises(ValueError, match='backend must be one of'):
        cj.count2(*p, battrs=battrs, mattrs=mattrs, backend='gpu')
    # compare mode belongs to the numpy frontend; a traced computation runs one
    with pytest.raises(ValueError, match='numpy-frontend mode'):
        cj.count2(*p, battrs=battrs, mattrs=mattrs, backend='compare')


NDEVICES = len(jax.devices('cpu')) if jax is not None else 0


@pytest.mark.skipif(NDEVICES < 2, reason='needs at least 2 jax CPU devices')
@pytest.mark.parametrize('battrs_kind', ['s', 'rp-pi'])
def test_count2_sharded_over_cpu_devices(battrs_kind):
    """shard_map splits catalogue 1 across devices and psums the counts.

    The result must not depend on the device count, for the vectorised kernel
    and for the generic path alike.
    """
    pos1, w1 = catalog(18)
    pos2, w2 = catalog(19)
    if battrs_kind == 's':
        battrs = BinAttrs(s=SEDGES)
    else:
        battrs = BinAttrs(rp=(np.linspace(1., 80., 6), 'z'),
                          pi=(np.linspace(-80., 80., 9), 'z'))

    nparticles = (Particles(pos1, w1), Particles(pos2, w2))
    want = count2(*nparticles, battrs=battrs,
                  mattrs=MeshAttrs(*nparticles, battrs=battrs), backend='cpu')['weight']

    mesh = cpu_mesh()
    with mesh:
        # Catalogue 1 is split across devices, catalogue 2 replicated: that is
        # the in_specs count2 builds its shard_map with.
        jparticles = (cj.Particles(place(mesh, pos1), place(mesh, w1),
                                   sharding_mesh=mesh),
                      cj.Particles(place(mesh, pos2, shard=False),
                                   place(mesh, w2, shard=False), sharding_mesh=mesh))
        mattrs = cj.MeshAttrs(*jparticles, battrs=battrs)
        got = cj.count2(*jparticles, battrs=battrs, mattrs=mattrs, backend="cpu",
                        sharding_mesh=mesh)["weight"]
    close(got, want)


@pytest.mark.skipif(NDEVICES < 2, reason='needs at least 2 jax CPU devices')
def test_count3_sharded_over_cpu_devices():
    cats = [sky_catalog(s, n=80) for s in (20, 21, 22)]
    e12 = np.linspace(10., 400., 5)
    b12, b13 = BinAttrs(s=e12), BinAttrs(s=e12)

    nparticles = [Particles(*c) for c in cats]
    want = count3(*nparticles, battrs12=b12, battrs13=b13, backend='cpu',
                  mattrs1=MeshAttrs(*nparticles, battrs=b12),
                  mattrs2=MeshAttrs(*nparticles, battrs=b12),
                  mattrs3=MeshAttrs(*nparticles, battrs=b13))['weight']

    mesh = cpu_mesh()
    with mesh:
        # shard_particle defaults to 1, so only catalogue 1 is split.
        jparticles = [cj.Particles(place(mesh, pos, shard=(i == 0)),
                                   place(mesh, w, shard=(i == 0)), sharding_mesh=mesh)
                      for i, (pos, w) in enumerate(cats)]
        got = cj.count3(*jparticles, battrs12=b12, battrs13=b13, backend='cpu',
                        mattrs1=cj.MeshAttrs(*jparticles, battrs=b12),
                        mattrs2=cj.MeshAttrs(*jparticles, battrs=b12),
                        mattrs3=cj.MeshAttrs(*jparticles, battrs=b13),
                        sharding_mesh=mesh)['weight']
    close(got, want)
