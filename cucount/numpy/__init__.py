import numbers
from dataclasses import dataclass, asdict

import numpy as np

import cucountlib
from cucountlib.cucount import setup_logging


@dataclass
class AngularWeight(object):

    sep: np.ndarray
    weight: np.ndarray

    def tree_flatten(self):
        children = (self.sep, self.weight)
        aux_data = dict()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def keys(self):
        return asdict(self).keys()

    def __getitem__(self, key):
        return getattr(self, key)

    def _to_c(self):
        return dict(self)


@dataclass(init=False)
class BitwiseWeight(object):

    default: float = 0.
    nrealizations: float = 0.
    noffset: int = 0
    nalways: int = 0
    p_correction_nbits: np.ndarray = None

    def __init__(self, nrealizations, default=0., noffset=None, nalways=0, p_correction_nbits=True):
        if not isinstance(nrealizations, numbers.Number):
            nrealizations = get_nrealizations_from_bitwise_weights(nrealizations)
        self.default = default
        self.nrealizations = nrealizations
        self.noffset = 1 if noffset is None else noffset
        self.nalways = nalways
        if size_bitwise_weights is None:
            size_bitwise_weights = nrealizations
        if isinstance(p_correction_nbits, bool):
            if p_correction_nbits:
                joint = joint_occurences(nrealizations, noffset=noffset + nalways, default=default)
                p_correction_nbits = np.ones((1 + size_bitwise_weights,) * 2, dtype=np.float64)
                cmin, cmax = nalways, min(nrealizations - noffset, size_bitwise_weights* 8)
                for c1 in range(cmin, 1 + cmax):
                    for c2 in range(cmin, 1 + cmax):
                        p_correction_nbits[c1, c2] = joint[c1 - nalways][c2 - nalways] if c2 <= c1 else joint[c2 - nalways][c1 - nalways]
                        p_correction_nbits[c1, c2] /= (nrealizations / (noffset + c1) * nrealizations / (noffset + c2))
            else:
                p_correction_nbits = None
        self.p_correction_nbits = p_correction_nbits

    def tree_flatten(self):
        """
        JAX pytree flatten: put array-like child(ren) in `children` and scalars in `aux_data`.
        If p_correction_nbits is not None it must be returned as a child, otherwise children is empty.
        """
        # children must be array-like objects that JAX can handle
        children = (self.p_correction_nbits,)
        # aux_data must be a pure-Python (picklable) structure with the remaining fields
        aux_data = {name: getattr(self, name) for name in ['default', 'num', 'noffset']}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct BitwiseWeight from aux_data and children produced by tree_flatten.
        """
        p_correction_nbits = children[0]
        return cls(**aux_data, p_correction_nbits=p_correction_nbits)

    def weight_iip(self, bitwise_weights):
        denom = self.noffset + sum(popcount(weight) for weight in bitwise_weights)
        mask = denom == 0
        denom[mask] = 1
        toret = np.empty_like(denom, dtype=np.float64)
        toret[...] = self.nrealizations / denom
        toret[mask] = self.default
        return toret

    def _to_c(self):
        state = dict(self)
        state.pop('nalways')
        return state


@dataclass(init=False)
class WeightAttrs(object):

    spin = None
    angular = None
    bitwise = None

    def __init__(self, spin=None, angular=None, bitwise=None):
        self.spin = spin
        self.angular = AngularWeight(**angular) if isinstance(angular, dict) else angular
        self.bitwise = BitwiseWeight(**bitwise) if isinstance(bitwise, dict) else bitwise

    def tree_flatten(self):
        children = (self.angular, self.bitwise)
        aux_data = dict(spin=self.spin)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        angular, bitwise = children
        return cls(*aux_data, angular=angular, bitwise=bitwise)

    def _to_c(self):
        state = asdict(self)
        for name in ['angular', 'bitwise']:
            state[name] = state[name]._to_c()
        for name in ['spin']:
            if state[name] is None: state.pop(name)
        return cucountlib.cucount.WeightAttrs(**state)


@dataclass(init=False)
class SelectionAttrs(object):

    def __init__(self, **kwargs):
        """
        Provide selection:
        - theta = (min, max)
        """
        self.__dict__.udpate(**kwargs)

    def _to_c(self):
        state = asdict(self)
        return cucountlib.cucount.SelectionAttrs(**state)


@dataclass(init=False)
class BinAttrs(object):

    def __init__(self, **kwargs):
        """
        Provide binning:
        - s = edge array or (min, max, step)
        - mu = (edge array or (min, max, step), line-of-sight (midpoint, firstpoint, endpoint, x, y, z))
        - theta = edge array or (min, max, step)
        """
        self.__dict__.udpate(**kwargs)

    def _to_c(self):
        state = asdict(self)
        return cucountlib.cucount.BinAttrs(**state)



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
        aux_data = asdict(self)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def _to_c(self):
        return asdict(self)

    @property
    def size(self):
        return self.size_spin + self.size_individual_weight + self.size_bitwise_weight + self.size_negative_weight


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
            values.append(spin_values.astype(np.float64))
            kwargs.update(size_spin=spin_values.shape[1])
        if weights is not None:
            if not isinstance(weights, (tuple, list)): # individual weights, bitwise weights
                weights = [weights]
            individual_weights, bitwise_weights = [], []
            for weight in weights:
                if np.issubdtype(weight.dtype, np.integer):
                    bitwise_weights += reformat_bitarrays(weight, dtype=np.uint64, copy=True)
                else:
                    individual_weights.append(weight.astype(np.float64))
            values += individual_weights
            values += bitwise_weights
            kwargs.update(size_individual_weight=len(individual_weights),
                          size_bitwise_weight=len(bitwise_weights))
        self.positions = positions.astype(np.float64)
        self.values = values
        self.index_value = IndexValue(**kwargs)

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

    def concatenate(values):
        if len(values) == 0:
            return None
        cvalues = []
        for value in values:
            if value.ndim == 1: value = value[:, np.newaxis]
            if np.issubdtype(value.dtype, np.integer):
                value = value.view(np.float64)
            cvalues.append(value)
        return np.concatenate(cvalues, axis=1)

    particles = [cucountlib.cucount.Particles(p.positions, values=concatenate(p.values), **p.index_value._to_c()) for p in particles]
    return cucountlib.cucount.count2(*particles, battrs=battrs._to_c(), wattrs=wattrs._to_c(), sattrs=sattrs._to_c())



def get_nrealizations_from_bitwise_weights(bitwise_weights):
    return sum(weight.dtype.itemsize for weight in bitwise_weights) * 8 + 1


# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def reformat_bitarrays(*arrays, dtype=np.uint64, copy=True):
    """
    Reformat input integer arrays into list of arrays of type ``dtype``.
    If, e.g. 6 arrays of type ``np.uint8`` are input, and ``dtype`` is ``np.uint32``,
    a list of 2 arrays is returned.

    Parameters
    ----------
    arrays : integer arrays
        Arrays of integers to reformat.

    dtype : string, dtype
        Type of output integer arrays.

    copy : bool, default=True
        If ``False``, avoids copy of input arrays if ``dtype`` is uint8.

    Returns
    -------
    arrays : list
        List of integer arrays of type ``dtype``, representing input integer arrays.
    """
    dtype = np.dtype(dtype)
    toret = []
    nremainingbytes = 0
    for array in arrays:
        # first bits are in the first byte array
        arrayofbytes = array.view((np.uint8, (array.dtype.itemsize,)))
        arrayofbytes = np.moveaxis(arrayofbytes, -1, 0)
        for arrayofbyte in arrayofbytes:
            if nremainingbytes == 0:
                toret.append([])
                nremainingbytes = dtype.itemsize
            newarray = toret[-1]
            nremainingbytes -= 1
            newarray.append(arrayofbyte[..., None])
    for iarray, array in enumerate(toret):
        npad = dtype.itemsize - len(array)
        if npad: array += [np.zeros_like(array[0])] * npad
        if len(array) > 1 or copy:
            toret[iarray] = np.squeeze(np.concatenate(array, axis=-1).view(dtype), axis=-1)
        else:
            toret[iarray] = array[0][..., 0]
    return toret


def pascal_triangle(n_rows):
    """
    Compute Pascal triangle.
    Taken from https://stackoverflow.com/questions/24093387/pascals-triangle-for-python.

    Parameters
    ----------
    n_rows : int
        Number of rows in the Pascal triangle, i.e. maximum number of elements :math:`n`.

    Returns
    -------
    triangle : list
        List of list of binomial coefficients.
        The binomial coefficient :math:`(k, n)` is ``triangle[n][k]``.
    """
    toret = [[1]]  # a container to collect the rows
    for _ in range(1, n_rows + 1):
        row = [1]
        last_row = toret[-1]  # reference the previous row
        # this is the complicated part, it relies on the fact that zip
        # stops at the shortest iterable, so for the second row, we have
        # nothing in this list comprension, but the third row sums 1 and 1
        # and the fourth row sums in pairs. It's a sliding window.
        row += [sum(pair) for pair in zip(last_row, last_row[1:])]
        # finally append the final 1 to the outside
        row.append(1)
        toret.append(row)  # add the row to the results.
    return toret


from functools import lru_cache

@lru_cache(maxsize=10, typed=False)
def joint_occurences(nrealizations=128, max_occurences=None, noffset=1, default=0):
    """
    Return expected value of inverse counts, i.e. eq. 21 of arXiv:1912.08803.

    Parameters
    ----------
    nrealizations : int
        Number of realizations (including current realization).

    max_occurences : int, default=None
        Maximum number of occurences (including ``noffset``).
        If ``None``, defaults to ``nrealizations``.

    noffset : int, default=1
        The offset added to the bitwise count, typically 0 or 1.
        See "zero truncated estimator" and "efficient estimator" of arXiv:1912.08803.

    default : float, default=0.
        The default value of pairwise weights if the denominator is zero (defaulting to 0).

    Returns
    -------
    occurences : list
        Expected value of inverse counts.
    """
    # gk(c1, c2)
    if max_occurences is None: max_occurences = nrealizations

    binomial_coeffs = pascal_triangle(nrealizations)

    def prob(c12, c1, c2):
        return binomial_coeffs[c1 - noffset][c12 - noffset] * binomial_coeffs[nrealizations - c1][c2 - c12] / binomial_coeffs[nrealizations - noffset][c2 - noffset]

    def fk(c12):
        if c12 == 0:
            return default
        return nrealizations / c12

    toret = []
    for c1 in range(noffset, max_occurences + 1):
        row = []
        for c2 in range(noffset, c1 + 1):
            # we have c12 <= c1, c2 and nrealizations >= c1 + c2 + c12
            row.append(sum(fk(c12) * prob(c12, c1, c2) for c12 in range(max(noffset, c1 + c2 - nrealizations), min(c1, c2) + 1)))
        toret.append(row)

    return toret