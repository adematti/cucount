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

    def _to_c(self):
        sep = np.cos(np.radians(self.sep))
        argsort = np.argsort(sep)
        state = {'sep': sep[argsort].copy(), 'weight': self.weight[argsort].copy()}
        return state

    def __call__(self, sep):
        """Return value of weights for given separation in degrees."""
        costheta = np.cos(np.radians(sep))
        state = self._to_c()
        return np.interp(costheta, state['sep'], state['weight'], left=1., right=1.)


@dataclass(init=False)
class BitwiseWeight(object):

    default_value: float = 0.
    nrealizations: float = 0.
    noffset: int = 0
    nalways: int = 0
    p_correction_nbits: np.ndarray = None

    def __init__(self, weights=None, nrealizations=None, default_value=0., noffset=None, nalways=0, p_correction_nbits=True):
        if weights is not None:
            max_bits = sum(weight.dtype.itemsize for weight in weights) * 8
            if nrealizations is None: nrealizations = 1 + max_bits
        else:
            assert nrealizations is not None
            max_bits = nrealizations
        self.default_value = default_value
        self.nrealizations = nrealizations
        self.noffset = 1 if noffset is None else noffset
        self.nalways = nalways
        if isinstance(p_correction_nbits, bool):
            if p_correction_nbits:
                joint = joint_occurences(self.nrealizations, noffset=self.noffset + self.nalways, default_value=self.default_value)
                p_correction_nbits = np.ones((1 + max_bits,) * 2, dtype=np.float64)
                cmin, cmax = self.nalways, min(self.nrealizations - self.noffset, max_bits)
                for c1 in range(cmin, 1 + cmax):
                    for c2 in range(cmin, 1 + cmax):
                        p_correction_nbits[c1, c2] = joint[c1 - self.nalways][c2 - self.nalways] if c2 <= c1 else joint[c2 - self.nalways][c1 - self.nalways]
                        p_correction_nbits[c1, c2] /= (self.nrealizations / (self.noffset + c1) * self.nrealizations / (self.noffset + c2))
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
        aux_data = {name: getattr(self, name) for name in ['nrealizations', 'default_value', 'noffset', 'nalways']}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct BitwiseWeight from aux_data and children produced by tree_flatten.
        """
        p_correction_nbits = children[0]
        return cls(**aux_data, p_correction_nbits=p_correction_nbits)

    def __call__(self, *bitwise_weights):
        """Return value of weights."""
        denom = self.noffset + sum(popcount(np.bitwise_and.reduce(weights)) for weights in zip(*bitwise_weights))
        mask = denom == 0
        toret = self.nrealizations / np.where(mask, 1., denom)
        if len(bitwise_weights) > 1:
            c = tuple(sum(popcount(weight) for weight in weights) for weights in bitwise_weights)
            toret /= self.p_correction_nbits[c]
        toret = np.where(mask, self.default_value, toret)
        return toret

    def _to_c(self):
        state = asdict(self)
        state.pop('nalways')
        return state


@dataclass(init=False)
class WeightAttrs(object):

    spin: tuple = None
    angular: AngularWeight = None
    bitwise: BitwiseWeight = None

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
        return cls(**aux_data, angular=angular, bitwise=bitwise)

    def _to_c(self):
        state = {}
        for name in ['angular', 'bitwise']:
            value = getattr(self, name)
            if value is not None:
                state[name] = value._to_c()
        for name in ['spin']:
            value = getattr(self, name)
            if value is not None:
                state[name] = value
        return cucountlib.cucount.WeightAttrs(**state)

    def check(self, *particles):
        if not particles:
            return
        for particle in particles:
            assert all(value.shape[0] == particle.size for value in particle.values), "All input value arrays should be of same length as positions"
        if self.spin is not None:
            assert all(particle.get('spin') for particle in particles), 'WeightAttrs.spin is not None, so Particles must have spin values'
        nbitwises = [len(particle.get('bitwise_weight')) for particle in particles]
        if any(nbitwises):
            assert self.bitwise is not None, 'Particles have bitwise weights, so provide bitwise to WeightAttrs'
        if self.bitwise is not None:
            assert all(nbitwise == nbitwises[0] for nbitwise in nbitwises), 'WeightAttrs.bitwise is not None, so Particles must have same bitwise weights'

    def __call__(self, *particles):
        """Return value of all weights."""
        weight = 1.
        for particle in particles:
            for value in particle.get('individual_weight'): weight *= value
        if self.bitwise:
            weight *= self.bitwise(*[particle.get('bitwise_weight') for particle in particles])
        if self.angular:
            angular = self.angular(0.)
            weight *= angular
        negatives = [particle.get('negative_weight') for particle in particles]
        if all(negatives):
            negative = 1.
            for _ in negatives:
                for value in _: negative *= value
            weight2 -= negative
        return weight


class SelectionAttrs(cucountlib.cucount.SelectionAttrs):
    """
    Provide selection:
    - theta = (min, max)  # in degrees
    """



class BinAttrs(cucountlib.cucount.BinAttrs):
    """
    Provide binning:
    - s = edge array or (min, max, step)
    - mu = (edge array or (min, max, step), line-of-sight (midpoint, firstpoint, endpoint, x, y, z))
    - theta = edge array or (min, max, step)
    """
    def edges(self, name=None):
        if name is None:
            return {coord: self.array[icoord] for icoord, coord in enumerate(self.varnames)}
        index = self.varnames.index(name)
        return self.array[index]

    def coords(self, name=None):
        def mid(array, name):
            if name in ['pole', 'k']:
                return array
            return (array[:-1] + array[1:]) / 2.
        if name is None:
            return {coord: mid(self.array[icoord], coord) for icoord, coord in enumerate(self.varnames)}
        index = self.varnames.index(name)
        return mid(self.array[index], name)


@dataclass(init=False)
class IndexValue(object):
    # To check/modify when adding new weighting scheme
    _fields = ['spin', 'individual_weight', 'bitwise_weight', 'negative_weight']

    sizes: dict = None

    def __init__(self, **kwargs):
        sizes = {name: 0 for name in self._fields}
        for name, size in kwargs.items():
            if name not in sizes:
                raise ValueError(f'{name} is not supported; options are {list(sizes)}')
            sizes[name] = size
        self._sizes = sizes

    def clone(self, **kwargs):
        """Copy and update."""
        return self.__class__(**(self._sizes | kwargs))

    def tree_flatten(self):
        # Only used by JAX; kept here for API consistency
        # Return flattenable children and auxiliary data (non-flattenable)
        children = tuple()
        aux_data = self._sizes
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def _to_c(self):
        return {f'size_{name}': value for name, value in self._sizes.items()}

    @property
    def size(self):
        return sum(self._sizes.values())

    def get(self, name, return_type=list):
        sizes = [self._sizes[name] for name in self._fields]
        cumsum = np.insert(np.cumsum(sizes, axis=0), 0, 0)
        sls = {name: slice(cumsum[i], cumsum[i + 1], 1) for i, name in enumerate(self._fields)}
        sl = sls[name]
        if return_type is list:
            return list(range(sl.start, sl.stop))
        return sl


def _make_list_weights(weights):
    if weights is None:
        return []
    if not isinstance(weights, (tuple, list)): # individual weights, bitwise weights
        weights = [weights]
    return list(weights)


def _format_values(weights=None, spin_values=None, index_value=None, np=np):
    values, kwargs = [], {}
    if spin_values is not None:
        if spin_values.ndim == 1:
            spin_values = spin_values[:, np.newaxis]
        values.append(spin_values.astype(np.float64))
        kwargs.update(spin=spin_values.shape[1])
    if weights is not None:
        weights = _make_list_weights(weights)
        individual_weights, bitwise_weights, negative_weights = [], [], []
        for weight in weights:
            if np.issubdtype(weight.dtype, np.integer):
                bitwise_weights += reformat_bitarrays(weight, dtype=np.uint64, copy=True, np=np)
            else:
                weight = weight.astype(np.float64)
                if bitwise_weights:
                    negative_weights.append(weight)  # if coming afer bitwise weight, assumed to be negative weight
                else:
                    individual_weights.append(weight)  # else, positive weight
        values += individual_weights
        values += bitwise_weights
        values += negative_weights
        kwargs.update(individual_weight=len(individual_weights),
                      bitwise_weight=len(bitwise_weights),
                      negative_weight=len(negative_weights))
    if isinstance(index_value, dict):
        kwargs.update(**index_value)
    if not isinstance(index_value, IndexValue):
        index_value = IndexValue(**index_value)
    return values, index_value


def _concatenate_values(values, np=np):
    if len(values) == 0:
        return None
    cvalues = []
    for value in values:
        if value.ndim == 1: value = value[:, np.newaxis]
        if np.issubdtype(value.dtype, np.integer):
            value = value.view(np.float64)
        cvalues.append(value)
    return np.concatenate(cvalues, axis=1)


def sky_to_cartesian(rdd, degree=True, dtype=np.float64, np=np):
    """
    Transform RA, Dec, distance into Cartesian coordinates.

    Parameters
    ----------
    rdd : array of shape (3, N), list of 3 arrays
        Right ascension, declination and distance.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    positions : list of 3 arrays
        Positions x, y, z in cartesian coordinates.
    """
    conversion = 1.
    if degree: conversion = np.pi / 180.
    ra, dec, dist = rdd
    cos_dec = np.cos(dec * conversion)
    x = dist * cos_dec * np.cos(ra * conversion)
    y = dist * cos_dec * np.sin(ra * conversion)
    z = dist * np.sin(dec * conversion)
    return [np.asarray(xx, dtype=dtype) for xx in [x, y, z]]


def _format_positions(positions, positions_type='pos', np=np):
    if positions_type == 'pos':
        positions = positions.astype(np.float64)
    elif positions_type == 'rdd':
        positions = np.column_stack(sky_to_cartesian(positions, np=np))
    elif positions_type == 'xyz':
        positions = np.column_stack(positions)
    return positions


@dataclass(init=False)
class Particles(object):

    positions: np.ndarray
    values: np.ndarray
    index_value: IndexValue

    def __init__(self, positions, weights=None, spin_values=None, positions_type='pos', index_value=None):
        # To check/modify when adding new weighting scheme
        self.values, self.index_value = _format_values(weights=weights, spin_values=spin_values, index_value=index_value, np=np)
        self.positions = _format_positions(positions, positions_type=positions_type, np=np)

    @property
    def size(self):
        return self.positions.shape[0]

    def clone(self, **kwargs):
        """Copy and replace positions, weights, spin_values, etc."""
        kwargs.setdefault('positions', self.positions)
        if 'weights' not in kwargs and 'spin_values' not in kwargs:
            kwargs.setdefault('index_value', self.index_value)  # preserve index_value
        kwargs.setdefault('weights', self.weights)
        kwargs.setdefault('spin_values', self.spin_values)
        return self.__class__(**kwargs)

    def get(self, name):
        return self.values[self.index_value.get(name, return_type=slice)]

    def tree_flatten(self):
        # Only used by JAX; kept here for API consistency
        # Return flattenable children and auxiliary data (non-flattenable)
        children = (self.positions, self.values, self.index_value)
        aux_data = None  # no auxiliary data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None):
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
    """
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    wattrs.check(*particles)
    particles = [cucountlib.cucount.Particles(p.positions, values=_concatenate_values(p.values, np=np), **p.index_value._to_c()) for p in particles]
    return cucountlib.cucount.count2(*particles, battrs=battrs, wattrs=wattrs._to_c(), sattrs=sattrs)


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


def reformat_bitarrays(*arrays, dtype=np.uint64, copy=True, np=np):
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
def joint_occurences(nrealizations=128, max_occurences=None, noffset=1, default_value=0):
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

    default_value : float, default=0.
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
            return default_value
        return nrealizations / c12

    toret = []
    for c1 in range(noffset, max_occurences + 1):
        row = []
        for c2 in range(noffset, c1 + 1):
            # we have c12 <= c1, c2 and nrealizations >= c1 + c2 + c12
            row.append(sum(fk(c12) * prob(c12, c1, c2) for c12 in range(max(noffset, c1 + c2 - nrealizations), min(c1, c2) + 1)))
        toret.append(row)

    return toret