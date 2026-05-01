import sys
import time
import itertools
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
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING, 'error': logging.ERROR}[level.lower()]
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
class AngularWeight:
    sep: list | None = None
    edges: list | None = None
    weight: np.ndarray | None = None
    _np = np

    def __init__(self, weight=None, **kwargs):
        self.weight = self._np.asarray(weight, dtype=self._np.float64)
        self.sep = None
        self.edges = None

        sep = kwargs.get("sep", None)
        edges = kwargs.get("edges", None)

        msg = "provide exactly one of sep or edges"
        assert (sep is None) != (edges is None), msg

        if sep is not None:
            sep = _make_list_weights(sep)
            self.sep = [self._np.asarray(arr, dtype=self._np.float64) for arr in sep]
            assert len(self.sep) == self.weight.ndim, (
                "provide a list of sep arrays, one for each dimension of weight"
            )
            for idim, arr in enumerate(self.sep):
                assert arr.ndim == 1, f"sep[{idim}] must be 1D"
                assert arr.shape[0] == self.weight.shape[idim], (
                    f"sep[{idim}] must have length weight.shape[{idim}]"
                )

        else:
            edges = _make_list_weights(edges)
            self.edges = [self._np.asarray(arr, dtype=self._np.float64) for arr in edges]
            assert len(self.edges) == self.weight.ndim, (
                "provide a list of edges arrays, one for each dimension of weight"
            )
            for idim, arr in enumerate(self.edges):
                assert arr.ndim == 1, f"edges[{idim}] must be 1D"
                assert arr.shape[0] == self.weight.shape[idim] + 1, (
                    f"edges[{idim}] must have length weight.shape[{idim}] + 1"
                )

    @property
    def tabulation(self):
        return "edges" if self.edges is not None else "sep"

    def tree_flatten(self):
        children = (getattr(self, self.tabulation), self.weight)
        aux_data = dict(tabulation=self.tabulation)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tabulation, weight = children
        return cls(weight=weight, **{aux_data["tabulation"]: tabulation})

    def _to_c(self):
        """
        Return a C-friendly representation:
        - convert angular coordinates from degrees to cos(theta)
        - sort each axis independently
        - permute the weight array consistently across all dimensions
        """
        state = {"weight": np.array(self.weight, dtype=np.float64, copy=True)}

        axes = getattr(self, self.tabulation)
        converted = [np.cos(np.radians(axis)) for axis in axes]
        argsorts = [np.argsort(axis) for axis in converted]
        sorted_axes = [axis[idx] for axis, idx in zip(converted, argsorts)]

        weight = state["weight"]
        for idim in range(weight.ndim):
            idx = np.argsort(converted[idim] if self.tabulation == 'sep' else converted[idim][:-1])
            weight = np.take(weight, idx, axis=idim)

        state[self.tabulation] = [np.array(axis, dtype=np.float64) for axis in sorted_axes]
        state["weight"] = np.array(weight, dtype=np.float64)
        return state

    @property
    def ndim(self):
        return self.weight.ndim

    def __call__(self, sep):
        """
        Return angular weight for given separation(s) in degrees.

        Parameters
        ----------
        sep : scalar or sequence
            - If weight is 1D, sep may be a scalar or array.
            - If weight is ND, sep must provide one coordinate per dimension:
              e.g. (sep0, sep1, ..., sepN-1), where each entry may be scalar
              or broadcastable array.

        Returns
        -------
        weight : scalar or ndarray
        """
        state = self._to_c()
        weight = state["weight"]

        # Normalize input into one entry per angular dimension
        if self.weight.ndim == 1:
            if not isinstance(sep, (tuple, list)):
                sep = [sep]
        else:
            assert isinstance(sep, (tuple, list)), (f"for {self.weight.ndim}D angular weights, sep must be a tuple/list "
                f"with one entry per dimension")
            assert len(sep) == self.weight.ndim, (
                f"expected {self.weight.ndim} separation arrays, got {len(sep)}"
            )
        coords_in = list(sep)

        coords = [self._np.cos(self._np.radians(self._np.asarray(coord))) for coord in coords_in]
        coords = self._np.broadcast_arrays(*coords)

        if self.tabulation == "edges":
            edges = state["edges"]

            idxs = []
            mask = self._np.ones(coords[0].shape, dtype=bool)
            for coord, edge in zip(coords, edges):
                idx = self._np.digitize(coord, edge, right=False) - 1
                valid = (idx >= 0) & (idx < len(edge) - 1)
                idxs.append(self._np.where(valid, idx, 0))
                mask &= valid

            values = weight[tuple(idxs)]
            return self._np.where(mask, values, 1.0)

        else:
            seps = state["sep"]

            if self.weight.ndim == 1:
                return self._np.interp(coords[0], seps[0], weight, left=1.0, right=1.0)

            # Multilinear interpolation on a rectilinear grid
            idx0 = []
            frac = []
            mask = self._np.ones(coords[0].shape, dtype=bool)

            for coord, grid in zip(coords, seps):
                i0 = self._np.searchsorted(grid, coord, side="right") - 1
                valid = (i0 >= 0) & (i0 < len(grid) - 1)
                i0_safe = self._np.clip(i0, 0, len(grid) - 2)

                g0 = grid[i0_safe]
                g1 = grid[i0_safe + 1]
                t = (coord - g0) / (g1 - g0)

                idx0.append(i0_safe)
                frac.append(t)
                mask &= valid

            out = self._np.zeros(coords[0].shape, dtype=weight.dtype)

            # Sum over 2**ndim cell corners
            for corner in range(1 << self.weight.ndim):
                indices = []
                coeff = 1.0
                for idim in range(self.weight.ndim):
                    upper = (corner >> idim) & 1
                    indices.append(idx0[idim] + upper)
                    coeff = coeff * (frac[idim] if upper else (1.0 - frac[idim]))
                out = out + coeff * weight[tuple(indices)]

            return self._np.where(mask, out, 1.0)


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
            assert len(self.spin) == len(particles), "Provide as many WeightAttrs.spin as Particles catalogs"
            assert all(bool(particle.get('spin')) == bool(spin) for spin, particle in zip(self.spin, particles)), "Provide spin_values whenever WeightAttrs.spin != 0"
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
            angular = self.angular((0.,) * self.angular.ndim)
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


class SplitAttrs(cucountlib.cucount.SplitAttrs):
    """
    Provide split attributes:
    - mode = 'jackknife'
    - nsplits = total number of splits
    """
    def check(self, *particles):
        for particle in particles:
            if self.nsplits:
                assert len(particle.index_value('split', return_type=list)) == 1, 'splits must be provided when SplitAttrs is set'
                #assert particle.get('split').max() < self.nsplits, 'particle.get("split") must be less than SplitAttrs.nsplits'
            else:
                assert len(particle.index_value('split', return_type=list)) == 0, 'splits provided but SplitAttrs is not set'


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
        if isinstance(name, list):
            return [self.edges(name) for name in name]
        index = self.varnames.index(name)
        return edge(self.array[index], name)

    def coords(self, name=None):
        def mid(array, name):
            if name in ['pole', 'k']:
                return array
            return (array[:-1] + array[1:]) / 2.
        if name is None:
            return {coord: mid(self.array[icoord], coord) for icoord, coord in enumerate(self.varnames)}
        if isinstance(name, list):
            return [self.coords(name) for name in name]
        index = self.varnames.index(name)
        return mid(self.array[index], name)

    @property
    def shape(self):
        return tuple(super().shape)


@dataclass(init=False)
class MeshAttrs(object):

    boxsize: np.ndarray
    boxcenter: np.ndarray
    meshsize: np.ndarray
    type: str
    periodic: bool
    smax: float
    _np = np

    def __init__(self, *positions, boxsize=None, boxcenter=None, meshsize=None, refine=1., battrs=None, sattrs=None, periodic=False):
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
        refine : float, default=1.
            Refine mesh by this factor; > 1 to increase the resolution of the mesh used to speed-up pair counting;
            < 1 to decrease the resolution (only impact running time).
        battrs : BinAttrs, optional
            Binning attributes. Used to determine cellsize if cellsize is None.
        sattrs : SelectionAttrs, optional
            Selection attributes. Used to determine boxsize if boxsize is None.
        periodic : bool, default=False
            Whether to use periodic boundary conditions.
        """
        positions = [p.positions if isinstance(p, Particles) else p for p in positions]
        nparticles = sum(p.shape[0] for p in positions) // len(positions) if len(positions) else 1

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
                r = np.sqrt((pos**2).sum(axis=-1, keepdims=True))
                pos = pos / r
                cth = np.clip(pos[..., 2], -1.0, 1.0)  # polar angle
                phi = np.arctan2(pos[..., 1], pos[..., 0]) % (2. * np.pi)  # azimuthal angle
                return np.column_stack((cth, phi))

            if mesh_type == 'angular':
                # angular: compute extent in theta, phi
                nonempty_positions = [cartesian_to_sphere(pos) for pos in nonempty_positions]

            pos_min = np.array([self._np.min(p, axis=axis) for p in nonempty_positions]).min(axis=0)
            pos_max = np.array([self._np.max(p, axis=axis) for p in nonempty_positions]).max(axis=0)
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

        boxsize = np.asarray(boxsize, dtype=np.float64) * np.ones(ndim, dtype=np.float64)
        boxcenter = np.asarray(boxcenter, dtype=np.float64) * np.ones(ndim, dtype=np.float64)
        if mesh_smax is None:
            mesh_smax = sum(bb**2 for bb in boxsize)**0.5

        # Now set up resolution meshsize
        if mesh_type == 'angular':
            if meshsize is None:
                theta_max = np.arccos(mesh_smax)
                nside1 = 5 * (np.pi / theta_max)
                fsky = boxsize.prod() / (4 * np.pi)
                nside2 = np.minimum(self._np.sqrt(0.25 * nparticles / fsky), 2048)
                meshsize = np.maximum(np.minimum(nside1, nside2) * refine, 1).astype(int)
                meshsize = [meshsize, 2 * meshsize]
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            pixel_resolution = np.degrees(np.sqrt(4 * np.pi / meshsize.prod()))
            logger.debug("Mesh size is %d = %d x %d.", meshsize.prod(), meshsize[0], meshsize[1])
            logger.debug("Pixel resolution is %.4lf deg.", pixel_resolution)
        elif mesh_type == 'cartesian':
            nside2 = (0.5 * nparticles)**(1. / 3.)
            if meshsize is None:
                nside1 = 6.0 * boxsize / mesh_smax
                meshsize = np.maximum(np.minimum(nside1, nside2) * refine, 1).astype(int)
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            cellsize = boxsize / meshsize
            logger.debug("Mesh size is %d = %d x %d x %d.", meshsize.prod(), meshsize[0], meshsize[1], meshsize[2])
            logger.debug("Cell size is (%.4lf, %.4lf, %.4lf).", cellsize[0], cellsize[1], cellsize[2])
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
        state = asdict(self)
        return cucountlib.cucount.MeshAttrs(**state)


@dataclass(init=False)
class IndexValue(object):
    # To check/modify when adding new weighting scheme
    _fields = ['split', 'spin', 'individual_weight', 'bitwise_weight', 'negative_weight']

    def __init__(self, **kwargs):
        sizes = {name: 0 for name in self._fields}
        for name, size in kwargs.items():
            if name not in sizes:
                raise ValueError(f'{name} is not supported; options are {list(sizes)}')
            sizes[name] = size
        self._sizes = sizes

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new._sizes = dict(self._sizes)
        return new

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


def _format_values(weights=None, spin_values=None, splits=None, index_value=None, np=np):
    values, kwargs = [], {}
    if splits is not None:
        values += [splits]
        kwargs.update(split=1)
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


def _stack_values(values, np=np):
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

    def __init__(self, positions, weights=None, spin_values=None, splits=None, positions_type='pos', index_value=None):
        # To check/modify when adding new weighting scheme
        self.values, index_value = _format_values(weights=weights, spin_values=spin_values, splits=splits, index_value=index_value, np=np)
        self.index_value = IndexValue(**index_value)
        self.positions = _format_positions(positions, positions_type=positions_type, np=np)

    @property
    def size(self):
        return self.positions.shape[0]

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.index_value = self.index_value.copy()
        new.values = list(self.values)
        new.positions = self.positions
        return new

    @classmethod
    def concatenate(cls, others):
        """Concatenate particles."""
        others = list(others)
        new = others[0].copy()
        new.values = [np.concatenate(values, axis=0) for values in zip(*[other.values for other in others])]
        new.positions = np.concatenate([other.positions for other in others], axis=0)
        return new

    def clone(self, **kwargs):
        """Copy and replace positions, weights, spin_values, etc."""
        kwargs.setdefault('positions', self.positions)
        if not any(name in kwargs for name in ['weights', 'spin_values', 'splits']):
            kwargs.setdefault('index_value', self.index_value)  # preserve index_value
        kwargs.setdefault('weights', self.get('weights'))
        kwargs.setdefault('spin_values', self.get('spin') or None)
        kwargs.setdefault('splits', (self.get('split') or [None])[0])
        return self.__class__(**kwargs)

    def get(self, name):
        """Get positions, weights, etc."""
        if name == 'positions':
            return self.positions
        if name == 'weights':
            weights = []
            for name, sl in self.index_value(return_type=slice).items():
                if name not in ['split', 'spin']: weights += self.values[sl]
            return weights
        return self.values[self.index_value(name, return_type=slice)]

    def __getitem__(self, name):
        if isinstance(name, str):
            return self.get(name)
        mask = name
        new = self.copy()
        new.index_value = self.index_value.clone()
        new.values = [value[mask] for value in self.values]
        new.positions = self.positions[mask]
        return new

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


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None,
           spattrs: SplitAttrs=None, mattrs: MeshAttrs=None, nthreads: int=1):
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
    spattrs : SplitAttrs, optional
        Split attributes (for jackknife). If None, defaults to SplitAttrs().
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
    if sattrs is None: sattrs = SelectionAttrs()
    if spattrs is None: spattrs = SplitAttrs()
    wattrs.check(*particles)
    spattrs.check(*particles)
    if mattrs is None: mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs)
    particles = [cucountlib.cucount.Particles(p.positions, values=_stack_values(p.values, np=np), **p.index_value._to_c()) for p in particles]
    return cucountlib.cucount.count2(*particles, mattrs._to_c(), battrs=battrs, wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs, nthreads=nthreads)


def wigner_3j(*ells):
    from sympy.physics.wigner import wigner_3j
    ells = map(int, ells)
    return float(wigner_3j(*ells))


def _triposh_transform_matrix_sub(ell1, ell2, ell3, tol=1e-12):
    """
    Matrix M such that, for fixed (ell1, ell2, ell3),

        c_triposh = M @ c_ylm

    with Eq. 30 of arXiv:1803.02132:

        zeta_{ell1 ell2 ell3}
        = (2 ell3 + 1) H_{ell1 ell2 ell3}
          sum_m (-1)^m
          ( ell1 ell2 ell3 ; m -m 0 )
          zeta^m_{ell1 ell2}

    where c_ylm is stored as

        [Re(m=0), Re(m=1), ..., Re(mmax),
         Im(m=1), ..., Im(mmax)].
    """
    ell1, ell2, ell3 = int(ell1), int(ell2), int(ell3)
    mmax = min(ell1, ell2)

    M = np.zeros((2 * mmax + 1,), dtype=float)

    H = float(wigner_3j(ell1, ell2, ell3, 0, 0, 0))
    if abs(H) < tol:
        return M

    prefactor = (2 * ell3 + 1) * H
    # m = 0
    W0 = float(wigner_3j(ell1, ell2, ell3, 0, 0, 0))
    M[0] = prefactor * W0
    # m > 0: fold ±m into real/imag stored coefficients.
    # For Eq. 30, H != 0 implies even parity, so only Re/cos contributes.
    for m in range(1, mmax + 1):
        Wm = float(wigner_3j(ell1, ell2, ell3, m, -m, 0))
        if abs(Wm) < tol:
            continue
        coeff = prefactor * ((-1) ** m) * Wm
        # contribution from +m and -m gives 2 Re[zeta^m]
        M[m] = 2.0 * coeff
        # sine block remains zero for Eq. 30 allowed rows
    return M


def triposh_to_poles(ells):
    """Return ells1, ells2 sufficient to cover input ells."""
    ells1, ells2 = [], []
    for ell1, ell2, ell3 in ells:
        ells1.append(ell1)
        ells2.append(ell2)
    return np.unique(ells1).tolist(), np.unique(sorted(ells2)).tolist()


def _get_ells(battrs):
    if isinstance(battrs, BinAttrs):
        try:
            ells = battrs.coords('pole')
        except (ValueError, IndexError):
            ells = []
    else:
        ells = battrs
    return [int(ell) for ell in ells]


def poles_to_ells(ells1, ells2):
    """Return (factor, ell1, ell2, m) for the stored pole axis."""
    ells1, ells2 = _get_ells(ells1), _get_ells(ells2)
    ells = []
    for ell1 in ells1:
        for ell2 in ells2:
            mmax = min(ell1, ell2)
            for m in range(mmax + 1):
                ells.append((1, ell1, ell2, m))   # Re
            for m in range(1, mmax + 1):
                ells.append((1j, ell1, ell2, m))  # Im
    return ells


def symmetrize_poles(poles, ells1, ells2, axis=-1, np=np):
    """
    Symmetrize pole coefficients following Eq. 9 of https://arxiv.org/pdf/1709.10150
    2017, retaining only real-valued positive-m coefficients.

    Returns
    -------
    sym : array
        Array with the pole axis replaced by the real-only symmetrized
        coefficients.
    ells : list
        Output labels ``(ell1, ell2, m)``.
    """
    labels = poles_to_ells(ells1, ells2)

    keep = []
    factors = []
    out_labels = []

    for ipole, (part, ell1, ell2, m) in enumerate(labels):
        if part == 1:
            keep.append(ipole)
            factors.append(1 if m == 0 else 2)
            out_labels.append((ell1, ell2, m))

    keep = np.asarray(keep)
    factors = np.asarray(factors, dtype=poles.dtype)

    sym = np.take(poles, keep, axis=axis)

    shape = [1] * sym.ndim
    shape[axis % sym.ndim] = factors.size
    sym = sym * factors.reshape(shape)

    return sym, out_labels


def triposh_transform_matrix(ells1, ells2, ells=None):
    """
    Build the linear transform from the CUDA Ylm-product basis to the
    tripoSH basis.

    This constructs a matrix ``M`` such that

    ``c_triposh = M @ counts``

    where ``c_cuda`` contains the coefficients produced by the GPU
    ``count3close`` kernel in its native packed basis, i.e. concatenated
    blocks of

    ``[cos(m=0), cos(1), ..., cos(mmax), sin(1), ..., sin(mmax)]``

    for each ``(ell1, ell2)`` pair in the CUDA projection layout.

    In the triplet counts, the line-of-sight direction was fixed to the z-axis.

    Parameters
    ----------
    ells1 : BinAttrs, list
        Bin attributes for the first leg.
        The ``'pole'`` coordinate determines the ordered list of ``ell1`` values.
    ells2 : BinAttrs, list
        Bin attributes for the second leg.
        The ``'pole'`` coordinate determines the ordered list of ``ell2`` values.
    ells : list[tuple], optional
        Explicit list of requested ``(ell1, ell2, ell3)`` modes.

        Each entry should be a tuple ``(ell1, ell2, ell3)``.
        If ``ell3`` is ``None``, all valid triangle-compatible values are
        included for that ``(ell1, ell2)`` pair.

        If None, all ``(ell1, ell2)`` pairs from
        ``itertools.product(battrs12.coords('pole'), battrs23.coords('pole'))``
        are included with all allowed ``ell3``.

    Returns
    -------
    out_ells : list[tuple]
        Flattened list of output ``(ell1, ell2, ell3)`` modes, one per row
        of the returned matrix.
    matrix : ndarray
        Dense transformation matrix of shape
        ``(n_triposh_modes, len(counts))``.
        Left-multiplying this matrix by the obtained counts yields the corresponding tripoSH coefficients.

    Notes
    -----
    The projection layout is assumed to match the native kernel packing
    order:

    ``for ell1 in battrs12.coords('pole'):``
    ``    for ell2 in battrs13.coords('pole'):``

    with each pair block occupying
    ``2 * min(ell1, ell2) + 1`` consecutive coefficients.
    """
    def pad(M, ell1, ell2, ells1, ells2):
        """
        Pad a local (ell1, ell2) tripoSH transform block into the full CUDA
        projection layout.

        Parameters
        ----------
        M : ndarray, shape (nrow, 2*min(ell1, ell2)+1)
            Local transform matrix for one (ell1, ell2) block.
        ell1, ell2 : int
            The multipole pair corresponding to M.
        bells1, bells2 : sequence[int]
            Full ordered multipole lists used in CUDA packing.

        Returns
        -------
        Mpad : ndarray, shape (nrow, nproj_total)
            Matrix padded into the full concatenated CUDA basis.
        """
        def block_size(l1, l2):
            return 2 * min(l1, l2) + 1

        # Total number of CUDA projections
        total = sum(block_size(l1, l2) for l1 in ells1 for l2 in ells2)

        # Find start offset of this block in CUDA ordering
        offset = 0
        found = False
        for l1 in ells1:
            for l2 in ells2:
                if l1 == ell1 and l2 == ell2:
                    found = True
                    break
                offset += block_size(l1, l2)
            if found:
                break

        if not found:
            raise ValueError(f"(ell1, ell2)=({ell1}, {ell2}) not found in provided pole coordinates")

        out = np.zeros(total, dtype=M.dtype)
        out[offset:offset + M.shape[0]] = M
        return out

    ells1, ells2 = _get_ells(ells1), _get_ells(ells2)
    if ells is None:
        ells = list(itertools.product(ells1, ells2))
        ells = [tuple(ell) + (None,) for ell in ells]
    matrix = []
    out_ells = []
    for ell1, ell2, ell3 in ells:
        ells3 = [ell3] if ell3 is not None else list(range(abs(ell1 - ell2), ell1 + ell2 + 1))
        for ell3 in ells3:
            M = _triposh_transform_matrix_sub(ell1, ell2, ell3)
            matrix.append(pad(M, ell1, ell2, ells1, ells2)[None, :])
            out_ells.append((ell1, ell2, ell3))
    return out_ells, np.concatenate(matrix, axis=0)


def count3close(*particles: Particles,
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
                mattrs1: MeshAttrs = None,
                mattrs2: MeshAttrs = None,
                mattrs3: MeshAttrs = None,
                close_pair: tuple = (1, 2),
                nthreads: int = 1):
    """
    Perform close-triplet counts using the native cucount library.

    This is a thin frontend that prepares Python-side ``Particles`` and
    weight/selection attributes and calls the underlying
    ``cucountlib.cucount.count3close`` implementation.

    Parameters
    ----------
    *particles : Particles
        Exactly three ``Particles`` instances corresponding to catalogs
        1, 2, and 3.
    battrs12 : BinAttrs
        Binning specification for pair (1, 2).
    battrs13 : BinAttrs
        Binning specification for pair (1, 3).
    battrs23 : BinAttrs, optional
        Binning specification for pair (2, 3).
    wattrs : WeightAttrs, optional
        Weight attributes. If ``None``, defaults to ``WeightAttrs()``.
    sattrs12 : SelectionAttrs, optional
        Selection attributes for pair (1, 2).
        If ``None``, defaults to ``SelectionAttrs()``.
    sattrs13 : SelectionAttrs, optional
        Selection attributes for pair (1, 3).
        If ``None``, defaults to ``SelectionAttrs()``.
    sattrs23 : SelectionAttrs, optional
        Selection attributes for pair (2, 3).
        If ``None``, defaults to ``SelectionAttrs()``.
    veto12 : SelectionAttrs, optional
        Veto selection for pair (1, 2).
        If this selection is satisfied, the pair (1, 2) is ignored.
    veto13 : SelectionAttrs, optional
        Veto selection for pair (1, 3).
        If this selection is satisfied, the pair (1, 3) is ignored.
    veto23 : SelectionAttrs, optional
        Veto selection for pair (2, 3).
        If this selection is satisfied, the pair (2, 3) is ignored.
    mattrs1 : MeshAttrs, optional
        Mesh attributes used for catalog 1.
        If ``None``, defaults to a mesh built from the selection and
        binning associated with the relevant close-pair search.
    mattrs2 : MeshAttrs, optional
        Mesh attributes used for catalog 2.
        If ``None``, defaults to a mesh built from the selection and
        binning associated with the relevant close-pair search.
    mattrs3 : MeshAttrs, optional
        Mesh attributes used for catalog 3.
        If ``None``, defaults to a mesh built from the selection and
        binning associated with the relevant close-pair search.
    close_pair : tuple, optional
        Close pair specification: ``(1, 2)``, ``(1, 3)``, or ``(2, 3)``.
        This only affects performance, not the final result.
        It is generally best to choose the pair with the tightest
        angular selection.
    nthreads : int, optional
        Number of GPUs (within the same node) to run in parallel on.

    Returns
    -------
    dict
        Output of the native ``count3close`` call.
        Typically a dictionary such as::

            {"weight": array}
    """
    _setup_cucount_logging()
    assert len(particles) == 3

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

    wattrs.check(*particles)

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

    particles = [
        cucountlib.cucount.Particles(
            p.positions,
            values=_stack_values(p.values, np=np),
            **p.index_value._to_c(),
        )
        for p in particles
    ]

    return cucountlib.cucount.count3close(
        *particles,
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12=battrs12,
        battrs13=battrs13,
        battrs23=battrs23,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        sattrs23=sattrs23,
        veto12=veto12,
        veto13=veto13,
        veto23=veto23,
        close_pair=close_pair,
        nthreads=nthreads,
    )


def count3(*particles: Particles,
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
           nthreads: int = 1):
    """
    Perform factorized triplet counts using the native cucount library.

    For each primary particle in catalog 1, catalog 2 is binned as a
    function of the (1, 2) separation and catalog 3 is binned as a function
    of the (1, 3) separation. The accumulated contribution is

    .. math::

        w_1 \\, w_2(r_{12}) \\, w_3(r_{13})

    There is no binning or selection in terms of the (2, 3) separation.

    Parameters
    ----------
    *particles : Particles
        Exactly three ``Particles`` instances corresponding to catalogs
        1, 2, and 3.
    battrs12 : BinAttrs
        Binning specification for pair (1, 2).
    battrs13 : BinAttrs
        Binning specification for pair (1, 3).
    wattrs : WeightAttrs, optional
        Weight attributes. If ``None``, defaults to ``WeightAttrs()``.
    sattrs12 : SelectionAttrs, optional
        Selection attributes for pair (1, 2).
    sattrs13 : SelectionAttrs, optional
        Selection attributes for pair (1, 3).
    veto12 : SelectionAttrs, optional
        Veto selection for pair (1, 2).
    veto13 : SelectionAttrs, optional
        Veto selection for pair (1, 3).
    mattrs1, mattrs2, mattrs3 : MeshAttrs, optional
        Mesh attributes used for catalogs 1, 2, and 3.
    nthreads : int, optional
        Number of GPUs within the same node to run in parallel on.

    Returns
    -------
    dict
        Output of the native ``count3`` call, typically ``{"weight": array}``.
    """
    _setup_cucount_logging()
    assert len(particles) == 3

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

    wattrs.check(*particles)

    if mattrs1 is None:
        mattrs1 = MeshAttrs(particles[0], sattrs=sattrs12, battrs=battrs12)
    if mattrs2 is None:
        mattrs2 = MeshAttrs(particles[1], sattrs=sattrs12, battrs=battrs12)
    if mattrs3 is None:
        mattrs3 = MeshAttrs(particles[2], sattrs=sattrs13, battrs=battrs13)

    particles = [
        cucountlib.cucount.Particles(
            p.positions,
            values=_stack_values(p.values, np=np),
            **p.index_value._to_c(),
        )
        for p in particles
    ]

    return cucountlib.cucount.count3(
        *particles,
        mattrs1._to_c(),
        mattrs2._to_c(),
        mattrs3._to_c(),
        battrs12=battrs12,
        battrs13=battrs13,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs12,
        sattrs13=sattrs13,
        veto12=veto12,
        veto13=veto13,
        nthreads=nthreads,
    )


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
        for ibyte in range(arrayofbytes.shape[0]):  # for JAX-sharding-friendliness
            arrayofbyte = arrayofbytes[ibyte]
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


def count2_analytic(battrs: BinAttrs, mattrs: MeshAttrs=None):
    """
    Perform two-point pair counts analytically for periodic boxes.

    Parameters
    ----------
    battrs : BinAttrs
        Binning specification (edges/shape) for the pair counts.
    mattrs : MeshAttrs or array, optional
        Mesh attributes (boxsize).

    Returns
    -------
    counts : array
        Normalized analytical pair counts in each bin.
    """
    boxsize = getattr(mattrs, 'boxsize', mattrs) * np.ones(3, dtype=np.float64)
    edges = battrs.edges()
    mode = tuple(edges)
    shape = battrs.shape
    if mode == ('s',):
        v = 4. / 3. * np.pi * edges['s']**3
        dv = np.diff(v, axis=-1)
    elif mode == ('s', 'mu'):
        # we bin in mu
        v = 2. / 3. * np.pi * edges['s'][..., None, None]**3 * edges['mu']
        dv = np.diff(np.diff(v, axis=1), axis=-1)
    elif mode == ('s', 'pole'):
        v = 4. / 3. * np.pi * edges['s']**3
        dv = np.diff(v, axis=-1)
        dv = np.concatenate([(ell == 0) * dv[..., None] for ell in battrs.coords('pole')], axis=-1)
    elif mode == ('rp', 'pi'):
        v = np.pi * edges['rp'][..., None, None]**2 * edges['pi']
        dv = np.diff(np.diff(v, axis=1), axis=-1)
    elif mode == ('rp',):
        los = battrs.losnames[0]
        v = np.pi * edges['rp']**2 * boxsize['xyz'.index(los)]
        dv = np.diff(v, axis=-1)
    else:
        raise NotImplementedError('No analytic pair counter provided for binning {}'.format(mode))
    return np.squeeze(dv).reshape(shape) / boxsize.prod()