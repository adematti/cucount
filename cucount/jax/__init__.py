import jax
from jax import numpy as jnp
from jax import tree_util

from cucountlib import ffi_cucount
from cucount.numpy import BinAttrs, WeightAttrs, SelectionAttrs, setup_logging
from cucount import numpy


jax.ffi.register_ffi_target('count2', ffi_cucount.count2())


IndexValue = tree_util.register_pytree_node_class(numpy.IndexValue)
Particles = tree_util.register_pytree_node_class(numpy.Particles)

jax.ffi.register_ffi_target("count2", ffi_cucount.count2(), platform="CUDA")


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None):
    assert jax.config.read('jax_enable_x64'), 'for cucount you have to enable float64'
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    wattrs.check(*particles)
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

    def concatenate(values):
        if len(values) == 0:
            return None
        cvalues = []
        for value in values:
            if value.ndim == 1: value = value[:, jnp.newaxis]
            if jnp.issubdtype(value.dtype, jnp.integer):
                value = value.view(jnp.float64)
            cvalues.append(value)
        return jnp.concatenate(cvalues, axis=1)

    args = sum(([particle.positions, concatenate(particle.values)] for particle in particles), start=[])
    counts = call(*args)[0]
    return {name: counts[icount * bsize:(icount + 1) * bsize].reshape(bshape) for icount, name in enumerate(names)}
