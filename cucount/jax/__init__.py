
from collections.abc import Callable
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax import tree_util
from jax.experimental.shard_map import shard_map
from jax import sharding, mesh_utils
from jax.sharding import PartitionSpec as P

from cucountlib import ffi_cucount
from cucount.numpy import BinAttrs, WeightAttrs, SelectionAttrs, _make_list_weights, _format_positions, _format_values, _concatenate_values, setup_logging
from cucount import numpy


jax.ffi.register_ffi_target('count2', ffi_cucount.count2())


def create_sharding_mesh(device_mesh_shape=None):

    if device_mesh_shape is None:
        count = len(jax.devices())
        device_mesh_shape = (count,)
    device_mesh_shape = tuple(s for s in device_mesh_shape if s > 1)
    devices = mesh_utils.create_device_mesh(device_mesh_shape)
    return sharding.Mesh(devices, axis_names=('x', 'y'))


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
                pad = jax.make_array_from_process_local_data(sharding, per_host_sum).sum(axis=0) / sizes.sum()[None, ...]
            elif pad == 'mean':
                pad = np.mean(per_host_array, axis=0)[None, ...]
            else:
                raise ValueError('mean or global_mean supported only')
        constant_values = pad
        pad = lambda array, pad_width: np.append(array, np.repeat(constant_values, pad_width[0][1], axis=0), axis=0)

    if per_host_size is None:
        if sizes is None: sizes = get_sizes()
        per_host_size = (sizes.max().item() + nlocal - 1) // nlocal * nlocal

    pad_width = [(0, per_host_size - per_host_array.shape[0])] + [(0, 0)] * (per_host_array.ndim - 1)
    per_host_array = pad(per_host_array, pad_width=pad_width)
    return jax.make_array_from_process_local_data(sharding, per_host_array)


IndexValue = tree_util.register_pytree_node_class(numpy.IndexValue)


@tree_util.register_pytree_node_class
class Particles(numpy.Particles):

    @default_sharding_mesh
    def __init__(self, positions, weights=None, spin_values=None, positions_type='pos', index_value=None, exchange=False, sharding_mesh=None):
        self.positions = _format_positions(positions, positions_type=positions_type, np=jnp)
        with_sharding = bool(sharding_mesh.axis_names)
        if with_sharding and exchange:  # weights required to pad with 0
            weights = [jnp.ones_like(self.positions, shape=self.positions.shape[0])] + _make_list_weights(weights)
        self.values, self.index_value = _format_values(weights=weights, spin_values=spin_values, index_value=index_value, np=jnp)
        self.exchanged = False
        if with_sharding and exchange:
            self.positions = make_array_from_process_local_data(positions, pad=0, sharding_mesh=sharding_mesh)
            self.values = [make_array_from_process_local_data(value, pad=0, sharding_mesh=sharding_mesh) for value in self.values]
            self.exchanged = True


jax.ffi.register_ffi_target("count2", ffi_cucount.count2(), platform="CUDA")


def _count2_no_shard(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None):
    ffi_cucount.set_attrs(battrs, wattrs=wattrs._to_c(), sattrs=sattrs)
    for i, p in enumerate(particles): ffi_cucount.set_index_value(i, **p.index_value._to_c())
    dtype = jnp.float64
    bsize, bshape = battrs.size, tuple(battrs.shape)
    names = ffi_cucount.get_count2_names()
    res_type = jax.ShapeDtypeStruct((len(names) * bsize,), dtype)
    ndim2 = sum(particle.positions.shape[1] for particle in particles)
    nvalues2 = sum(particle.index_value.size for particle in particles)
    # Max values
    nblocks = 256
    meshsize = sum(particle.size + 100 for particle in particles)
    # Estimate buffer size
    size = 0
    # Buffer for set_mesh_attrs
    size += nblocks * ndim2
    # Buffer for mesh
    size += 2 * meshsize
    size += (nvalues2 + ndim2) * sum(particle.size for particle in particles)
    # Buffer for count2: bins
    size += sum(s + 1 for s in bshape)
    # Buffer for count2: bins
    size += nblocks * bsize
    buffer_type = jax.ShapeDtypeStruct((size,), dtype)
    call = jax.ffi.ffi_call('count2', (res_type, buffer_type))

    args = sum(([particle.positions, _concatenate_values(particle.values, np=jnp)] for particle in particles), start=[])
    counts = call(*args)[0]
    return {name: counts[icount * bsize:(icount + 1) * bsize].reshape(bshape) for icount, name in enumerate(names)}


@default_sharding_mesh
def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None, sharding_mesh=None):
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
    assert jax.config.read('jax_enable_x64'), 'for cucount you have to enable float64'
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    wattrs.check(*particles)
    count2 = _count2 = partial(_count2_no_shard, battrs=battrs, wattrs=wattrs, sattrs=sattrs)
    if sharding_mesh.axis_names:
        #assert all(particle.exchanged for particle in particles), 'All input particles should be exchanged'
        count2 = shard_map(lambda *particles: jax.lax.psum(_count2(*particles), sharding_mesh.axis_names), mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names), P(None)), out_specs=P(None))
    return count2(*particles)
