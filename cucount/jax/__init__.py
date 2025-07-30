import jax
from jax import numpy as jnp

from cucountlib import ffi_cucount
from cucountlib.ffi_cucount import BinAttrs, SelectionAttrs, setup_logging


jax.ffi.register_ffi_target('count2', ffi_cucount.count2())


from collections import namedtuple
from jax import tree_util


@tree_util.register_pytree_node_class
class Particles(namedtuple('Particles', ['positions', 'weights'])):

    def tree_flatten(self):
        # Return flattenable children and auxiliary data (non-flattenable)
        children = (self.positions, self.weights)
        aux_data = None  # no auxiliary data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def size(self):
        return self.positions.shape[0]


jax.ffi.register_ffi_target("count2", ffi_cucount.count2(), platform="CUDA")


def count2(*particles: Particles, battrs: BinAttrs, sattrs: SelectionAttrs=SelectionAttrs()):
    assert len(particles) == 2
    assert jax.config.read('jax_enable_x64'), 'for cucount you have to enable float64'
    ffi_cucount.set_attrs(battrs, sattrs=sattrs)
    dtype = jnp.float64
    bshape = tuple(battrs.shape)
    bsize = battrs.size
    res_type = jax.ShapeDtypeStruct(bshape, dtype)
    ndim = particles[0].positions.shape[1]
    # Max values
    nblocks = 256
    meshsize = sum(particle.size + 100 for particle in particles)
    # Estimate buffer size
    size = 0
    # Buffer for set_mesh_attrs
    size += nblocks * 2 * ndim
    # Buffer for mesh
    size += 2 * meshsize
    size += (2 + 2 * ndim) * sum(particle.size for particle in particles)
    # Buffer for count2: bins
    size += sum(s + 1 for s in bshape)
    # Buffer for count2: bins
    size += nblocks * bsize
    buffer_type = jax.ShapeDtypeStruct((size,), dtype)
    call = jax.ffi.ffi_call('count2', (res_type, buffer_type))
    args = sum(([particle.positions, particle.weights] for particle in particles), start=[])
    return call(*args)[0]
