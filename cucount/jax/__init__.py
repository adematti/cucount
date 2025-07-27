import jax
from cucountlib import ffi_cucount
from cucountlib.cucount import BinAttrs, SelectionAttrs


for name, target in ffi_cucount.registrations().items():
    jax.ffi.register_ffi_target(name, target)


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


def count2(particles1: Particles, particles2: Particles, battrs: BinAttrs, sattrs: SelectionAttrs=SelectionAttrs()):
    pass