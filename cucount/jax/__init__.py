import jax
from jax import numpy as jnp
from jax import tree_util

from cucountlib import ffi_cucount
from cucountlib.ffi_cucount import BinAttrs, WeightAttrs, SelectionAttrs, setup_logging
from cucount import numpy


jax.ffi.register_ffi_target('count2', ffi_cucount.count2())


IndexValue = tree_util.register_pytree_node_class(numpy.IndexValue)
Particles = tree_util.register_pytree_node_class(numpy.Particles)
jax.ffi.register_ffi_target("count2", ffi_cucount.count2(), platform="CUDA")


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs, sattrs: SelectionAttrs=SelectionAttrs()):
    assert len(particles) == 2
    assert jax.config.read('jax_enable_x64'), 'for cucount you have to enable float64'
    ffi_cucount.set_attrs(battrs, wattrs=wattrs, sattrs=sattrs)
    for i, p in enumerate(particles): ffi_cucount.set_index_value(i, **p.index_value)
    dtype = jnp.float64
    bshape = tuple(battrs.shape)
    bsize = battrs.size
    res_type = jax.ShapeDtypeStruct(bsize, dtype)
    ndim2 = sum(p.positions.shape[1] for p in particles)
    nvalues2 = sum(p.index_value.size for p in particles)
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
    args = sum(([particle.positions, particle.values] for particle in particles), start=[])
    return call(*args)[0]
