
from collections.abc import Callable
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax import tree_util
from jax.experimental.shard_map import shard_map
from jax import sharding
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

from cucountlib import ffi_cucount
from cucount.numpy import BinAttrs, SelectionAttrs, SplitAttrs, _make_list_weights, _format_positions, _format_values, _stack_values, count2_analytic, setup_logging, _setup_cucount_logging, triposh_to_poles, triposh_transform_matrix, _get_ells
from cucount import numpy


jax.ffi.register_ffi_target('count2', ffi_cucount.count2())


def create_sharding_mesh(device_mesh_shape=None):
    if device_mesh_shape is None:
        count = len(jax.devices())
        device_mesh_shape = (count,)
    device_mesh_shape = tuple(s for s in device_mesh_shape if s > 1)
    return jax.make_mesh(device_mesh_shape, axis_names=list('xyz'[:len(device_mesh_shape)]), axis_types=(jax.sharding.AxisType.Auto,) * len(device_mesh_shape))


def get_sharding_mesh():
    from jax._src import mesh as mesh_lib
    return mesh_lib.thread_resources.env.physical_mesh


def default_sharding_mesh(func: Callable):

    @functools.wraps(func)
    def wrapper(*args, sharding_mesh=None, **kwargs):
        if sharding_mesh is None:
            sharding_mesh = get_sharding_mesh()
        return func(*args, sharding_mesh=sharding_mesh, **kwargs)

    return wrapper


@default_sharding_mesh
def make_array_from_process_local_data(per_host_array, per_host_size=None, pad=0., sharding_mesh: jax.sharding.Mesh=None):

    if not len(sharding_mesh.axis_names):
        return per_host_array

    nlocal = len(jax.local_devices())
    sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))

    sizes = None
    def get_sizes():
        return jax.make_array_from_process_local_data(sharding, np.repeat(per_host_array.shape[0], nlocal))

    if not callable(pad):
        if isinstance(pad, str):
            if pad == 'global_mean':
                if sizes is None:
                    sizes = get_sizes()
                per_host_sum = np.repeat(per_host_array.sum(axis=0, keepdims=True), nlocal, axis=0)
                pad = jax.make_array_from_process_local_data(sharding, per_host_sum).sum(axis=0, keepdims=True) / sizes.sum()[None, ...]
            elif pad == 'mean':
                pad = np.mean(per_host_array, axis=0)[None, ...]
            else:
                raise ValueError('mean or global_mean supported only')
        constant_values = pad
        pad = lambda array, pad_width: np.concatenate([array, np.repeat(np.asarray(constant_values, dtype=array.dtype), pad_width[0][1], axis=0)], dtype=array.dtype, axis=0)

    if per_host_size is None:
        if sizes is None: sizes = get_sizes()
        per_host_size = (sizes.max().item() + nlocal - 1) // nlocal * nlocal

    pad_width = [(0, per_host_size - per_host_array.shape[0])] + [(0, 0)] * (per_host_array.ndim - 1)
    per_host_array = pad(per_host_array, pad_width=pad_width)
    return jax.make_array_from_process_local_data(sharding, per_host_array)


@tree_util.register_pytree_node_class
class AngularWeight(numpy.AngularWeight):

    pass
    #_np = jnp  # cannot set, else got "ValueError: array is not writeable"


@tree_util.register_pytree_node_class
class BitwiseWeight(numpy.BitwiseWeight):

    _np = jnp


@tree_util.register_pytree_node_class
class WeightAttrs(numpy.WeightAttrs):

    def __init__(self, spin=None, angular=None, bitwise=None):
        self.spin = spin
        self.angular = AngularWeight(**angular) if isinstance(angular, dict) else angular
        self.bitwise = BitwiseWeight(**bitwise) if isinstance(bitwise, dict) else bitwise


@tree_util.register_pytree_node_class
class MeshAttrs(numpy.MeshAttrs):

    _np = jnp


@tree_util.register_pytree_node_class
class IndexValue(numpy.IndexValue):

    pass



@tree_util.register_pytree_node_class
class Particles(numpy.Particles):

    @default_sharding_mesh
    def __init__(self, positions, weights=None, spin_values=None, splits=None, positions_type='pos', index_value=None, exchange=False, sharding_mesh=None):
        self.positions = _format_positions(positions, positions_type=positions_type, np=jnp)
        with_sharding = bool(sharding_mesh.axis_names)
        self.values, index_value = _format_values(weights=weights, spin_values=spin_values, splits=splits, index_value=index_value, np=jnp)
        self.index_value = IndexValue(**index_value)
        if not self.index_value('individual_weight'):
            # Let's add weights in any case (required arguments to count2)
            self.values = [jnp.ones_like(self.positions, shape=self.positions.shape[0])] + self.values
            index_value['individual_weight'] = 1
            self.index_value = IndexValue(**index_value)
        if with_sharding and exchange:
            self.positions = make_array_from_process_local_data(self.positions, pad='mean', sharding_mesh=sharding_mesh)
            self.values = [make_array_from_process_local_data(value, pad=0, sharding_mesh=sharding_mesh) for value in self.values]

    @classmethod
    def concatenate(cls, others):
        """Concatenate particles."""
        new = cls.__new__(cls)
        new.index_value = others[0].index_value.clone()
        new.values = [jnp.concatenate(values, axis=0) for values in zip(*[other.values for other in others])]
        new.positions = jnp.concatenate([other.positions for other in others], axis=0)
        return new


symmetrize_poles = partial(numpy.symmetrize_poles, np=jnp)


jax.ffi.register_ffi_target("count2", ffi_cucount.count2(), platform="CUDA")


def _count2_no_shard(*particles: Particles, mattrs: MeshAttrs, battrs: BinAttrs, wattrs: WeightAttrs = None,
                     sattrs: SelectionAttrs = None, spattrs: SplitAttrs = None):
    mattrs._to_c()
    ffi_cucount.set_count2_attrs(
        mattrs._to_c(), battrs, wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs
    )
    for i, p in enumerate(particles):
        ffi_cucount.set_index_value(i, **p.index_value._to_c())

    dtype = jnp.float64
    names, shape = ffi_cucount.get_count2_layout()
    shape = tuple(shape)
    size = int(np.prod(shape, dtype=int))

    res_type = jax.ShapeDtypeStruct((len(names) * size,), dtype)

    # Max values
    nblocks = 256

    # Two meshes
    meshsize = mattrs.meshsize.prod()
    meshsize2 = 2 * meshsize

    # Estimate buffer size
    bufsize = 0

    # Buffer for mesh
    # nparticles, cumnparticles
    bufsize += 2 * meshsize2

    # index, positions, spositions, values
    bufsize += sum(particle.positions.shape[0] for particle in particles)
    bufsize += 2 * sum(particle.positions.size for particle in particles)
    bufsize += sum(particle.index_value.size * particle.positions.shape[0] for particle in particles)

    # Buffer for recursive scan
    bufsize += 2 * 2 * (meshsize + 1024 - 1) // 1024

    # Buffer for angular upweights
    if wattrs.angular is not None:
        bufsize += 2 * wattrs.angular.weight.size + 1

    # Buffer for bitwise correction
    if wattrs.bitwise is not None and wattrs.bitwise.p_correction_nbits is not None:
        bufsize += wattrs.bitwise.p_correction_nbits.size

    # Buffer for count2: edges
    bufsize += sum(s + 1 for s in shape[-battrs.ndim:])

    # Buffer for count2: nblocks * counts
    bufsize += nblocks * len(names) * size

    buffer_type = jax.ShapeDtypeStruct((bufsize,), dtype)
    call = jax.ffi.ffi_call("count2", (res_type, buffer_type))

    args = sum(
        ([particle.positions, _stack_values(particle.values, np=jnp)] for particle in particles),
        start=[],
    )
    counts = call(*args)[0]

    return {
        name: counts[iweight * size:(iweight + 1) * size].reshape(shape)
        for iweight, name in enumerate(names)
    }


@default_sharding_mesh
def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs = None, sattrs: SelectionAttrs = None,
           spattrs: SplitAttrs = None, mattrs: MeshAttrs = None, sharding_mesh=None):
    """
    Perform two-point pair counts using the native cucount library.

    This is a thin frontend that prepares Python-side Particles and Weight/Selection
    attributes and calls the underlying cucountlib.cucount.count2 implementation
    (GPU-accelerated C/C++/CUDA).

    Parameters
    ----------
    *particles : Particles
        Exactly two Particles instances to correlate (positions, optional weights/spin/bitwise).
    battrs : BinAttrs
        Binning specification (edges/shape) for the pair counts.
    wattrs : WeightAttrs, optional
        Weight attributes (spin, angular, bitwise). If None, a default WeightAttrs()
        (no weights) is used.
    sattrs : SelectionAttrs, optional
        Selection attributes to restrict pairs. If None, defaults to SelectionAttrs().
    mattrs : MeshAttrs, optional
        Mesh attributes (periodic, cellsize). If None, defaults to MeshAttrs().

    Returns
    -------
    result : dict
        Output of the native count2 call. A dict of named arrays (e.g. weight, weight_plus, weight_cross, etc.).

    Note
    ----
    Computation is distributed if in the context of a sharding mesh:

    ```
    with create_sharding_mesh() as mesh:

        counts = count2(...)
    ```
    """
    assert jax.config.read("jax_enable_x64"), "for cucount you have to enable float64"
    assert len(particles) == 2
    _setup_cucount_logging()
    if wattrs is None:
        wattrs = WeightAttrs()
    if sattrs is None:
        sattrs = SelectionAttrs()
    if spattrs is None:
        spattrs = SplitAttrs()
    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs)
    wattrs.check(*particles)
    spattrs.check(*particles)
    count2 = _count2 = partial(
        _count2_no_shard, mattrs=mattrs, battrs=battrs, wattrs=wattrs, sattrs=sattrs, spattrs=spattrs
    )
    if sharding_mesh.axis_names:
        count2 = shard_map(
            lambda *particles: jax.lax.psum(_count2(*particles), sharding_mesh.axis_names),
            mesh=sharding_mesh,
            in_specs=(P(sharding_mesh.axis_names), P(None)),
            out_specs=P(None),
        )
    return count2(*particles)


jax.ffi.register_ffi_target("count3close", ffi_cucount.count3close(), platform="CUDA")


def _count3close_no_shard(
    *particles: Particles,
    mattrs1: MeshAttrs,
    mattrs2: MeshAttrs,
    mattrs3: MeshAttrs,
    battrs12: BinAttrs,
    battrs13: BinAttrs,
    battrs23: BinAttrs = None,
    wattrs: WeightAttrs = None,
    sattrs12: SelectionAttrs = None,
    sattrs13: SelectionAttrs = None,
    sattrs23: SelectionAttrs = None,
    veto12: SelectionAttrs = None,
    veto13: SelectionAttrs = None,
    veto23: SelectionAttrs = None,
    close_pair: tuple[int, int] = (1, 2),
):
    assert len(particles) == 3

    ffi_cucount.set_count3close_attrs(
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12,
        battrs13,
        battrs23,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        sattrs23=sattrs23,
        veto12=veto12,
        veto13=veto13,
        veto23=veto23,
        close_pair=close_pair,
    )

    for i, p in enumerate(particles):
        ffi_cucount.set_count3close_index_value(i, **p.index_value._to_c())

    dtype = jnp.float64

    names, shape = ffi_cucount.get_count3close_layout()
    shape = tuple(shape)
    size = int(np.prod(shape, dtype=int))

    res_type = jax.ShapeDtypeStruct((len(names) * size,), dtype)

    nblocks = 256

    meshsize1 = int(np.prod(mattrs1.meshsize))
    meshsize2 = int(np.prod(mattrs2.meshsize))
    meshsize3 = int(np.prod(mattrs3.meshsize))

    bufsize = 0
    bufsize += 2 * (meshsize1 + meshsize2 + meshsize3)

    bufsize += sum(particle.positions.shape[0] for particle in particles)
    bufsize += 2 * sum(particle.positions.size for particle in particles)
    bufsize += sum(particle.index_value.size * particle.positions.shape[0] for particle in particles)

    bufsize += 2 * ((meshsize1 + 1024 - 1) // 1024)
    bufsize += 2 * ((meshsize2 + 1024 - 1) // 1024)
    bufsize += 2 * ((meshsize3 + 1024 - 1) // 1024)

    if wattrs.angular is not None:
        bufsize += 2 * wattrs.angular.weight.size + 1

    if wattrs.bitwise is not None and wattrs.bitwise.p_correction_nbits is not None:
        bufsize += wattrs.bitwise.p_correction_nbits.size

    bufsize += sum(s + 1 for s in battrs12.shape)
    bufsize += sum(s + 1 for s in battrs13.shape)
    if battrs23 is not None:
        bufsize += sum(s + 1 for s in battrs23.shape)

    bufsize += nblocks * len(names) * size

    buffer_type = jax.ShapeDtypeStruct((bufsize,), dtype)
    call = jax.ffi.ffi_call("count3close", (res_type, buffer_type))

    args = sum(
        ([particle.positions, _stack_values(particle.values, np=jnp)] for particle in particles),
        start=[],
    )
    counts = call(*args)[0]

    return {
        name: counts[iweight * size:(iweight + 1) * size].reshape(shape)
        for iweight, name in enumerate(names)
    }


@default_sharding_mesh
def count3close(
    *particles: Particles,
    battrs12: BinAttrs,
    battrs13: BinAttrs,
    battrs23: BinAttrs = None,
    wattrs: WeightAttrs = None,
    sattrs12: SelectionAttrs = None,
    sattrs13: SelectionAttrs = None,
    sattrs23: SelectionAttrs = None,
    veto12: SelectionAttrs = None,
    veto13: SelectionAttrs = None,
    veto23: SelectionAttrs = None,
    close_pair: tuple[int, int] = (1, 2),
    mattrs1: MeshAttrs = None,
    mattrs2: MeshAttrs = None,
    mattrs3: MeshAttrs = None,
    sharding_mesh=None,
    shard_particle: int = 1,
):
    assert jax.config.read("jax_enable_x64"), "for cucount you have to enable float64"
    assert len(particles) == 3
    assert shard_particle in (1, 2, 3)

    _setup_cucount_logging()

    if wattrs is None:
        wattrs = WeightAttrs()

    if sattrs12 is None:
        sattrs12 = SelectionAttrs()
    if sattrs13 is None:
        sattrs13 = SelectionAttrs()
    if sattrs23 is None:
        sattrs23 = SelectionAttrs()

    if veto12 is None:
        veto12 = SelectionAttrs()
    if veto13 is None:
        veto13 = SelectionAttrs()
    if veto23 is None:
        veto23 = SelectionAttrs()

    assert close_pair in [(1, 2), (1, 3), (2, 3)]

    if mattrs1 is None:
        mattrs1 = MeshAttrs(
            particles[0],
            sattrs=sattrs13 if close_pair == (1, 3) else sattrs12,
            battrs=battrs13 if close_pair == (1, 3) else battrs12,
        )

    if mattrs2 is None:
        mattrs2 = MeshAttrs(
            particles[1],
            sattrs=sattrs23 if close_pair == (2, 3) else sattrs12,
            battrs=battrs23 if close_pair == (2, 3) else battrs12,
        )

    if mattrs3 is None:
        mattrs3 = MeshAttrs(
            particles[2],
            sattrs=sattrs23 if close_pair == (2, 3) else sattrs13,
            battrs=battrs23 if close_pair == (2, 3) else battrs13,
        )

    wattrs.check(*particles)

    _count3close = partial(
        _count3close_no_shard,
        mattrs1=mattrs1,
        mattrs2=mattrs2,
        mattrs3=mattrs3,
        battrs12=battrs12,
        battrs23=battrs23,
        battrs13=battrs13,
        wattrs=wattrs,
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        sattrs23=sattrs23,
        veto12=veto12,
        veto13=veto13,
        veto23=veto23,
        close_pair=close_pair,
    )

    count3close_fn = _count3close

    if sharding_mesh.axis_names:
        in_specs = [P(None), P(None), P(None)]
        in_specs[shard_particle - 1] = P(sharding_mesh.axis_names)

        count3close_fn = shard_map(
            lambda *particles: jax.lax.psum(
                _count3close(*particles),
                sharding_mesh.axis_names,
            ),
            mesh=sharding_mesh,
            in_specs=tuple(in_specs),
            out_specs=P(None),
        )

    return count3close_fn(*particles)


jax.ffi.register_ffi_target("count3", ffi_cucount.count3(), platform="CUDA")


def _count3_no_shard(
    *particles: Particles,
    mattrs1: MeshAttrs,
    mattrs2: MeshAttrs,
    mattrs3: MeshAttrs,
    battrs12: BinAttrs,
    battrs13: BinAttrs,
    wattrs: WeightAttrs = None,
    sattrs12: SelectionAttrs = None,
    sattrs13: SelectionAttrs = None,
    veto12: SelectionAttrs = None,
    veto13: SelectionAttrs = None,
):
    assert len(particles) == 3

    ffi_cucount.set_count3_attrs(
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12,
        battrs13,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        veto12=veto12,
        veto13=veto13,
    )

    for i, p in enumerate(particles):
        ffi_cucount.set_count3_index_value(i, **p.index_value._to_c())

    dtype = jnp.float64

    names, shape = ffi_cucount.get_count3_layout()
    shape = tuple(shape)
    size = int(np.prod(shape, dtype=int))

    res_type = jax.ShapeDtypeStruct((len(names) * size,), dtype)

    nblocks = 256
    nthreads_per_block = 256

    meshsize1 = int(np.prod(mattrs1.meshsize))
    meshsize2 = int(np.prod(mattrs2.meshsize))
    meshsize3 = int(np.prod(mattrs3.meshsize))

    bufsize = 0
    bufsize += 2 * (meshsize1 + meshsize2 + meshsize3)

    bufsize += sum(particle.positions.shape[0] for particle in particles)
    bufsize += 2 * sum(particle.positions.size for particle in particles)
    bufsize += sum(
        particle.index_value.size * particle.positions.shape[0]
        for particle in particles
    )

    bufsize += 2 * ((meshsize1 + 1024 - 1) // 1024)
    bufsize += 2 * ((meshsize2 + 1024 - 1) // 1024)
    bufsize += 2 * ((meshsize3 + 1024 - 1) // 1024)

    if wattrs.angular is not None:
        bufsize += 2 * wattrs.angular.weight.size + 1

    if wattrs.bitwise is not None and wattrs.bitwise.p_correction_nbits is not None:
        bufsize += wattrs.bitwise.p_correction_nbits.size

    bufsize += sum(s + 1 for s in battrs12.shape)
    bufsize += sum(s + 1 for s in battrs13.shape)

    # Buffer for the 12 and 12 histograms
    nthreads = nblocks * nthreads_per_block
    nells2, nells3 = [max(sum(2 * ell + 1 for ell in _get_ells(battrs)), 1) for battrs in [battrs12, battrs13]]
    hsize2 = battrs12.shape[0] * nells2
    hsize3 = battrs13.shape[0] * nells3

    bufsize += nthreads * hsize2
    bufsize += nthreads * hsize3
    bufsize += nblocks * len(names) * size

    buffer_type = jax.ShapeDtypeStruct((bufsize,), dtype)
    call = jax.ffi.ffi_call("count3", (res_type, buffer_type))

    args = sum(
        ([particle.positions, _stack_values(particle.values, np=jnp)] for particle in particles),
        start=[],
    )
    counts = call(*args)[0]

    return {
        name: counts[iweight * size:(iweight + 1) * size].reshape(shape)
        for iweight, name in enumerate(names)
    }


@default_sharding_mesh
def count3(
    *particles: Particles,
    battrs12: BinAttrs,
    battrs13: BinAttrs,
    wattrs: WeightAttrs = None,
    sattrs12: SelectionAttrs = None,
    sattrs13: SelectionAttrs = None,
    veto12: SelectionAttrs = None,
    veto13: SelectionAttrs = None,
    mattrs1: MeshAttrs = None,
    mattrs2: MeshAttrs = None,
    mattrs3: MeshAttrs = None,
    sharding_mesh=None,
    shard_particle: int = 1,
):
    """
    Compute factorized 3-point counts around each primary catalog D1.

    For each primary particle :math:`D_1(\\vec{r}_1)`:

    - particles in :math:`D_2` are binned as a function of
      :math:`r_{12} = |\\vec{r}_2 - \\vec{r}_1|` to obtain
      :math:`w_2(r_{12})`
    - particles in :math:`D_3` are binned as a function of
      :math:`r_{13} = |\\vec{r}_3 - \\vec{r}_1|` to obtain
      :math:`w_3(r_{13})`
    - the total contribution

      .. math::

          w_1 \\, w_2(r_{12}) \\, w_3(r_{13})

      is accumulated

    Unlike :func:`count3close`, there is no binning or selection in
    :math:`r_{23}`. The count is therefore factorized into two independent
    pair counts around the same primary object.

    If ``battrs12`` and ``battrs13`` include a ``pole`` dimension
    (``VAR_POLE``), local-frame spherical harmonic projections are computed
    using the LOS of the primary particle as the local z-axis, following the
    same convention as :func:`count3close`.

    Parameters
    ----------
    particles : Particles
        Three particle catalogs ``(D1, D2, D3)``.
    battrs12 : BinAttrs
        Binning definition for the ``(1, 2)`` pair.
    battrs13 : BinAttrs
        Binning definition for the ``(1, 3)`` pair.
    wattrs : WeightAttrs, optional
        Weight definitions (individual, bitwise, angular, etc.).
    sattrs12 : SelectionAttrs, optional
        Pair selection for the ``(1, 2)`` pair.
    sattrs13 : SelectionAttrs, optional
        Pair selection for the ``(1, 3)`` pair.
    veto12 : SelectionAttrs, optional
        Pair veto for the ``(1, 2)`` pair.
    veto13 : SelectionAttrs, optional
        Pair veto for the ``(1, 3)`` pair.
    mattrs1, mattrs2, mattrs3 : MeshAttrs, optional
        Mesh construction attributes for each catalog. If not provided,
        reasonable defaults are inferred from the corresponding binning and
        selection attributes.
    sharding_mesh : optional
        JAX sharding mesh used for distributed execution.
    shard_particle : int, default=1
        Which particle catalog to shard across devices (1, 2, or 3).

    Returns
    -------
    dict
        Output of the native ``count3`` call, typically ``{"weight": array}``.
    """
    assert jax.config.read("jax_enable_x64"), "for cucount you have to enable float64"
    assert len(particles) == 3
    assert shard_particle in (1, 2, 3)

    _setup_cucount_logging()

    if wattrs is None:
        wattrs = WeightAttrs()

    if sattrs12 is None:
        sattrs12 = SelectionAttrs()
    if sattrs13 is None:
        sattrs13 = SelectionAttrs()

    if veto12 is None:
        veto12 = SelectionAttrs()
    if veto13 is None:
        veto13 = SelectionAttrs()

    if mattrs1 is None:
        mattrs1 = MeshAttrs(particles[0], sattrs=sattrs12, battrs=battrs12)
    if mattrs2 is None:
        mattrs2 = MeshAttrs(particles[1], sattrs=sattrs12, battrs=battrs12)
    if mattrs3 is None:
        mattrs3 = MeshAttrs(particles[2], sattrs=sattrs13, battrs=battrs13)

    wattrs.check(*particles)

    _count3 = partial(
        _count3_no_shard,
        mattrs1=mattrs1,
        mattrs2=mattrs2,
        mattrs3=mattrs3,
        battrs12=battrs12,
        battrs13=battrs13,
        wattrs=wattrs,
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        veto12=veto12,
        veto13=veto13,
    )

    count3_fn = _count3

    if sharding_mesh.axis_names:
        in_specs = [P(None), P(None), P(None)]
        in_specs[shard_particle - 1] = P(sharding_mesh.axis_names)

        count3_fn = shard_map(
            lambda *particles: jax.lax.psum(
                _count3(*particles),
                sharding_mesh.axis_names,
            ),
            mesh=sharding_mesh,
            in_specs=tuple(in_specs),
            out_specs=P(None),
        )

    return count3_fn(*particles)