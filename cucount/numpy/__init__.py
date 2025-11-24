import sys
import time
import numbers
import logging
import functools
import operator
from dataclasses import dataclass, asdict

import numpy as np

import cucountlib.cucount


logger = logging.getLogger('cucount')


def _setup_cucount_logging():
    level = logging.getLogger('cucount').getEffectiveLevel()
    level = logging.getLevelName(level)
    cucountlib.cucount.setup_logging(level.lower())


def setup_logging(level=logging.INFO, stream=sys.stdout,  **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : str, int, default=logging.INFO
        Logging level.
    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.
    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    _setup_cucount_logging()


@dataclass
class AngularWeight(object):

    sep: np.ndarray
    weight: np.ndarray
    _np = np

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
        state = {'sep': np.array(sep[argsort], dtype=np.float64), 'weight': np.array(self.weight[argsort], dtype=np.float64)}
        return state

    def __call__(self, sep):
        """Return value of weights for given separation in degrees."""
        costheta = self._np.cos(self._np.radians(sep))
        state = self._to_c()
        return self._np.interp(costheta, state['sep'], state['weight'], left=1., right=1.)


@dataclass(init=False)
class BitwiseWeight(object):

    default_value: float = 0.
    nrealizations: float = 0.
    noffset: int = 0
    nalways: int = 0
    p_correction_nbits: np.ndarray = None
    _np = np

    def __init__(self, weights=None, nrealizations=None, default_value=0., noffset=None, nalways=0, p_correction_nbits=True):
        if weights is not None:
            assert all(np.issubdtype(weight.dtype, np.integer) for weight in weights)
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
        bitwise_weights = [reformat_bitarrays(*weights, dtype=np.uint64, copy=True, np=self._np) for weights in bitwise_weights]
        denom = self.noffset + sum(popcount(functools.reduce(operator.and_, weights), np=self._np) for weights in zip(*bitwise_weights))
        mask = denom == 0
        toret = self.nrealizations / self._np.where(mask, 1., denom)
        if len(bitwise_weights) > 1 and self.p_correction_nbits is not None:
            c = tuple(sum(popcount(weight, np=self._np) for weight in weights) for weights in bitwise_weights)
            toret = toret / self._np.asarray(self.p_correction_nbits)[c]
        toret = self._np.where(mask, self.default_value, toret)
        return toret

    def _to_c(self):
        state = asdict(self)
        state.pop('nalways')
        for name in ['p_correction_nbits']:
            if state[name] is None: state.pop(name)
            else: state[name] = np.array(state[name], dtype=np.float64)
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
            assert len(particle.index_value('individual_weight', return_type=list)) <= 1, "Only one individual weight is supported"
            assert len(particle.index_value('negative_weight', return_type=list)) <= 1, "Only one negative weight is supported"
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
            weight -= negative
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
    - rp = (edge array or (min, max, step), line-of-sight (midpoint, firstpoint, endpoint, x, y, z))
    - pi = (edge array or (min, max, step), line-of-sight (midpoint, firstpoint, endpoint, x, y, z))
    - theta = edge array or (min, max, step)  # in degrees
    """
    def edges(self, name=None):
        def edge(array, name):
            if name in ['pole', 'k']:
                return array
            return np.column_stack([array[:-1], array[1:]])
        if name is None:
            return {coord: edge(self.array[icoord], coord) for icoord, coord in enumerate(self.varnames)}
        index = self.varnames.index(name)
        return edge(self.array[index], name)

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
class MeshAttrs(object):

    boxsize: np.ndarray
    boxcenter: np.ndarray
    meshsize: np.ndarray
    type: str
    periodic: bool
    smax: float
    _np = np

    def __init__(self, *positions, boxsize=None, boxcenter=None, meshsize=None, battrs=None, sattrs=None, periodic=False):
        """
        Determine mesh attributes from input positions and other attributes.

        Parameters
        ----------
        *positions : array-like or Particles
            List of positions arrays.

        boxsize : array-like of 3 floats, optional
            Box size along each axis. If None, determined from positions.

        meshsize : array-like of 3 floats, optional
            Size of the mesh along each axis. If None, determined from battrs or sattrs.

        battrs : BinAttrs, optional
            Binning attributes. Used to determine cellsize if cellsize is None.

        sattrs : SelectionAttrs, optional
            Selection attributes. Used to determine boxsize if boxsize is None.

        periodic : bool, default=False
            Whether to use periodic boundary conditions.
        """
        positions = [p.positions if isinstance(p, Particles) else p for p in positions]
        nparticles = sum(p.shape[0] for p in positions) // len(positions)

        def _get_extent(*positions):
            """Return minimum physical extent (min, max) corresponding to input positions."""
            if not positions:
                raise ValueError('positions must be provided if boxsize and boxcenter are not specified, or check is True')
            nonempty_positions = [pos for pos in positions if pos.size]
            if not nonempty_positions:
                raise ValueError('<= 1 particles found; cannot infer boxsize')
            axis = tuple(range(len(nonempty_positions[0].shape[:-1])))

            def cartesian_to_sphere(pos, np=self._np):
                """Convert cartesian to spherical coordinates (r, theta, phi)."""
                x, y, z = pos.T
                r = np.sqrt(x**2 + y**2 + z**2)
                cth = np.clip(z / r, -1.0, 1.0)  # polar angle
                phi = np.arctan2(y, x) % (2. * np.pi)  # azimuthal angle
                return np.column_stack((cth, phi))

            if mesh_type == 'angular':
                # angular: compute extent in theta, phi
                nonempty_positions = [cartesian_to_sphere(pos) for pos in nonempty_positions]

            pos_min = self._np.array([self._np.min(p, axis=axis) for p in nonempty_positions]).min(axis=0)
            pos_max = self._np.array([self._np.max(p, axis=axis) for p in nonempty_positions]).max(axis=0)
            return pos_min, pos_max

        mesh_type, mesh_smax = None, None
        limits = {}
        for attrs in [sattrs, battrs]:
            if attrs is None: continue
            for name, lim in zip(attrs.varnames, attrs.max):
                if name in limits: limits[name] = min(limits[name], lim)
                else: limits[name] = lim

        for name, lim in limits.items():
            if name == 'theta':
                mesh_type = 'angular'
                mesh_smax = np.cos(np.radians(lim))
                break
            elif name == 's':
                mesh_type = 'cartesian'
                mesh_smax = lim
                break
            elif name in ['rp', 'pi', 'k']:
                mesh_type = 'cartesian'

        assert mesh_type is not None, 'cannot determine mesh type from sattrs or battrs; provide at least one'
        if mesh_smax is None and all(name in limits for name in ['rp', 'pi']):
            mesh_smax = (limits['rp']**2 + limits['pi']**2)**0.5
        ndim = {'angular': 2, 'cartesian': 3}[mesh_type]

        if periodic:
            assert mesh_type == 'cartesian'
            assert boxsize is not None, 'if periodic=True, boxsize must be provided'
            boxcenter = 0.

        elif boxsize is None or boxcenter is None:
            extent = _get_extent(*positions)
            if boxsize is None:
                boxsize = 1.00001 * (extent[1] - extent[0])
            if boxcenter is None:
                boxcenter = 0.5 * (extent[1] + extent[0])

        boxsize = self._np.asarray(boxsize, dtype=np.float64) * np.ones(ndim, dtype=np.float64)
        boxcenter = self._np.asarray(boxcenter, dtype=np.float64) * np.ones(ndim, dtype=np.float64)
        if mesh_smax is None:
            mesh_smax = sum(bb**2 for bb in boxsize)**0.5

        # Now set up resolution meshsize
        if mesh_type == 'angular':
            if meshsize is None:
                theta_max = np.arccos(mesh_smax)
                nside1 = 5 * np.rint(np.pi / theta_max).astype(int)
                fsky = boxsize.prod() / (4 * np.pi)
                nside2 = np.minimum(self._np.sqrt(self._np.rint((0.25 * nparticles / fsky)).astype(int)), 2048)
                meshsize = np.maximum(np.minimum(nside1, nside2), 1)
                meshsize = [meshsize, 2 * meshsize]
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            pixel_resolution = np.degrees(np.sqrt(4 * np.pi / meshsize.prod()))
            logger.info("Mesh size is %d = %d x %d.", meshsize.prod(), meshsize[0], meshsize[1])
            logger.info("Pixel resolution is %.4lf deg.", pixel_resolution)
        elif mesh_type == 'cartesian':
            nside2 = int((0.5 * nparticles)**(1. / 3.))
            if meshsize is None:
                nside1 = np.rint(4.0 * boxsize / mesh_smax).astype(int)
                meshsize = np.maximum(np.minimum(nside1, nside2), 1)
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            cellsize = boxsize / meshsize
            logger.info("Mesh size is %d = %d x %d x %d.", meshsize.prod(), meshsize[0], meshsize[1], meshsize[2])
            logger.info("Cell size is (%.4lf, %.4lf, %.4lf).", cellsize[0], cellsize[1], cellsize[2])
        self.meshsize = meshsize
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.smax = mesh_smax
        self.type = mesh_type
        self.periodic = bool(periodic)

    def tree_flatten(self):
        children = (self.meshsize, self.boxsize, self.boxcenter, self.smax)
        aux_data = dict(type=self.type, periodic=self.periodic)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.meshsize, new.boxsize, new.boxcenter, new.smax = children
        new.__dict__.update(aux_data)
        return new

    def _to_c(self):
        return cucountlib.cucount.MeshAttrs(**asdict(self))


@dataclass(init=False)
class IndexValue(object):
    # To check/modify when adding new weighting scheme
    _fields = ['spin', 'individual_weight', 'bitwise_weight', 'negative_weight']

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

    def __call__(self, name=None, return_type=list):
        sizes = [self._sizes[name] for name in self._fields]
        cumsum = np.insert(np.cumsum(sizes, axis=0), 0, 0)
        sls = {name: slice(cumsum[i], cumsum[i + 1], 1) for i, name in enumerate(self._fields)}
        if return_type is list:
            sls = {name: list(range(sl.start, sl.stop)) for name, sl in sls.items()}
        if name is None:
            return sls
        return sls[name]

    def __repr__(self):
        s = ', '.join(f'{k}={v}' for k, v in self._sizes.items())
        return f'{self.__class__.__name__}({s})'


def _make_list_weights(weights):
    if weights is None:
        return []
    if not isinstance(weights, (tuple, list)): # individual weights, bitwise weights
        weights = [weights]
    return list(weights)


def _format_values(weights=None, spin_values=None, index_value=None, np=np):
    values, kwargs = [], {}
    if spin_values is not None:
        spin_values = _make_list_weights(spin_values)
        _spin_values = []
        for value in spin_values:
            value = value.astype(np.float64)
            if value.ndim == 2:
                _spin_values += list(value.T)
            else:
                assert value.ndim == 1, 'Only 1D or 2D arrays are supported for spin values'
                _spin_values.append(value)
        values += _spin_values
        kwargs.update(spin=len(_spin_values))
    if weights is not None:
        weights = _make_list_weights(weights)
        individual_weights, bitwise_weights, negative_weights = [], [], []
        for weight in weights:
            if np.issubdtype(weight.dtype, np.integer):
                bitwise_weights += reformat_bitarrays(weight, dtype=np.uint64, copy=True, np=np)
            else:
                weight = weight.astype(np.float64)
                assert weight.ndim == 1, 'Only 1D arrays are supported for weights'
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
    if index_value is not None:
        kwargs.update(**(index_value if isinstance(index_value, dict) else index_value._sizes))
    return values, kwargs


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
    elif positions_type == 'rdd':  # RA, Dec, distance
        positions = np.column_stack(sky_to_cartesian(positions, np=np))
    elif positions_type == 'rd':
        positions = np.column_stack(sky_to_cartesian(list(positions) + [np.ones_like(positions[0])], np=np))
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
        self.values, index_value = _format_values(weights=weights, spin_values=spin_values, index_value=index_value, np=np)
        self.index_value = IndexValue(**index_value)
        self.positions = _format_positions(positions, positions_type=positions_type, np=np)

    @property
    def size(self):
        return self.positions.shape[0]

    def clone(self, **kwargs):
        """Copy and replace positions, weights, spin_values, etc."""
        kwargs.setdefault('positions', self.positions)
        if 'weights' not in kwargs and 'spin_values' not in kwargs:
            kwargs.setdefault('index_value', self.index_value)  # preserve index_value
        kwargs.setdefault('weights', self.get('weights'))
        kwargs.setdefault('spin_values', self.get('spin') or None)
        return self.__class__(**kwargs)

    def get(self, name):
        if name == 'positions':
            return self.positions
        if name == 'weights':
            weights = []
            for name, sl in self.index_value(return_type=slice).items():
                weights += self.values[sl]
            return weights
        return self.values[self.index_value(name, return_type=slice)]

    def tree_flatten(self):
        # Only used by JAX; kept here for API consistency
        # Return flattenable children and auxiliary data (non-flattenable)
        children = (self.positions, self.values, self.index_value)
        aux_data = None  # no auxiliary data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.positions, new.values, new.index_value = children
        return new


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None, mattrs: MeshAttrs=None, nthreads: int=1):
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
    nthreads : int, optional
        Number of GPUs (within the same node) to run in parallel on.

    Returns
    -------
    result : dict
        Output of the native count2 call. A dict of named arrays (e.g. weight, weight_plus, weight_cross, etc.).
    """
    _setup_cucount_logging()
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    wattrs.check(*particles)
    if sattrs is None: sattrs = SelectionAttrs()
    if mattrs is None: mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs)
    particles = [cucountlib.cucount.Particles(p.positions, values=_concatenate_values(p.values, np=np), **p.index_value._to_c()) for p in particles]
    return cucountlib.cucount.count2(*particles, mattrs._to_c(), battrs=battrs, wattrs=wattrs._to_c(), sattrs=sattrs, nthreads=nthreads)


# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays, np=np):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    try:
        _popcount = np.bitwise_count
    except AttributeError:
        def _popcount(array):
            return _popcount_lookuptable[array.view((np.uint8, (array.dtype.itemsize,)))].sum(axis=-1)
    return sum(_popcount(array) for array in arrays)


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
        if np.__name__.startswith('jax'):
            import jax
            arrayofbytes = jax.lax.bitcast_convert_type(array, np.uint8)
        else:
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