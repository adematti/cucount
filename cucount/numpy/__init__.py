from dataclasses import dataclass, asdict

import numpy as np

import cucountlib
from cucountlib.cucount import BinAttrs, WeightAttrs, SelectionAttrs, setup_logging


@dataclass
class IndexValue(object):
    # To check/modify when adding new weighting scheme

    size_spin: int = 0
    size_individual_weight: int = 0
    size_bitwise_weight: int = 0

    def tree_flatten(self):
        # Only used by JAX; kept here for API consistency
        # Return flattenable children and auxiliary data (non-flattenable)
        children = tuple()
        aux_data = asdict(self)  # no auxiliary data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def keys(self):
        return asdict(self).keys()

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def size(self):
        return self.size_spin + self.size_individual_weight + self.size_bitwise_weight


@dataclass(init=False)
class Particles(object):

    positions: np.ndarray
    values: np.ndarray
    index_value: IndexValue

    def __init__(self, positions, weights=None, spin_values=None):
        # To check/modify when adding new weighting scheme
        values, kwargs = [], {}
        if spin_values is not None:
            if spin_values.ndim == 1:
                spin_values = spin_values[:, np.newaxis]
            values.append(spin_values)
            kwargs.update(size_spin=spin_values.shape[1])
        if weights is not None:
            if weights.ndim == 1:
                weights = weights[:, np.newaxis]
            values.append(weights)
            kwargs.update(size_individual_weight=weights.shape[1])
        self.positions = positions
        self.values = self._get_values(values)
        self.index_value = IndexValue(**kwargs)

    def _get_values(self, values):
        return np.concatenate(values, axis=1) if values else None

    @property
    def size(self):
        return self.positions.shape[0]

    def tree_flatten(self):
        # Only used by JAX; kept here for API consistency
        # Return flattenable children and auxiliary data (non-flattenable)
        children = (self.positions, self.values, self.index_value)
        aux_data = None  # no auxiliary data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=WeightAttrs(), sattrs: SelectionAttrs=SelectionAttrs()):
    assert len(particles) == 2
    particles = [cucountlib.cucount.Particles(p.positions, values=p.values, **p.index_value) for p in particles]
    return cucountlib.cucount.count2(*particles, battrs=battrs, wattrs=wattrs, sattrs=sattrs)
