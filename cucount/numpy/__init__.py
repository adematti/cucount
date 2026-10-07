import sys
import time
import itertools
import logging
import contextlib
import functools
import operator
import os
import warnings
from dataclasses import dataclass, asdict

import numpy as np

# Import names mirror the source layout: cucountlib.attrs / .cuda / .cpu
# / .ffi_cuda; local aliases keep the frontend text stable.
from cucountlib import attrs as cucount_attrs

logger = logging.getLogger('cucount')


def _log_level_name():
    return logging.getLevelName(logger.getEffectiveLevel()).lower()


BACKENDS = ('cuda', 'cpu', 'compare')
"""Backend selection, via a backend= keyword or the CUCOUNT_BACKEND variable.

cuda     CUDA
cpu      the portable CPU backend, raising if it cannot serve the request
compare  run both and raise if they disagree

With neither given: 'cuda' when this build has it and the process sees a GPU,
else 'cpu'.
"""

# The two backends accumulate in different orders (per-thread histograms on the
# GPU, SIMD lanes + threads on the CPU), so bitwise equality is not the
# contract; this is. One knob, applied everywhere compare mode compares.
COMPARE_RTOL = float(os.environ.get('CUCOUNT_COMPARE_RTOL', 1e-9))

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
# Both backends against the same three entry points, side by side rather than
# in an adapter module each: after the two converged, what is left per backend
# is the extension handle, its tuning keys and the call itself, and everything
# else -- what a request may ask for, how it is dispatched, how a disagreement
# is reported -- is backend-neutral and lives below.
#
# Either extension may be absent, and this module imports and serves whichever
# is there: a -DCUCOUNT_BUILD_CUDA=OFF build still serves backend='cpu'.

# Both extensions are imported here, either one optional. Importing
# cucountlib.cuda costs about 10 ms and pulls in no CUDA libraries -- nvcc
# links the runtime statically and the driver is dlopened at the first CUDA
# call, not at import -- so there is nothing to defer. (cucountlib.cpu, the
# multi-ISA Highway module, is the expensive one at ~300 ms.) cucount.jax
# imports its two the same way.

try:
    from cucountlib import cuda as cudalib
except ImportError:  # -DCUCOUNT_BUILD_CUDA=OFF, or the CUDA libraries are missing
    cudalib = None

try:
    from cucountlib import cpu as cpulib
except ImportError:  # -DCUCOUNT_BUILD_CPU=OFF
    cpulib = None


def cuda_available():
    return cudalib is not None


def cpu_available():
    return cpulib is not None


def cpu_nthreads():
    """CPU threads, which is not what CUDA's nthreads means (that is GPUs)."""
    n = os.environ.get('CUCOUNT_CPU_NTHREADS')
    return int(n) if n else len(os.sched_getaffinity(0))


CUDA_TUNING_KEYS = ('nthreads',)
"""Tuning keys the CUDA backend accepts through the public tuning= keyword.

nthreads  number of GPUs (within the same node) to run in parallel on

Kernel launch geometry is chosen by occupancy (CONFIGURE_KERNEL_LAUNCH) and
is not exposed.
"""

CPU_TUNING_KEYS = ('nthreads', 'isa', 'scatter', 'float32')
"""Tuning keys the CPU backend accepts through the public tuning= keyword.

nthreads  CPU threads (default: CUCOUNT_CPU_NTHREADS, else the affinity mask)
isa       Highway target to pin for this call, e.g. 'AVX2' (default: automatic)
scatter   'scalar' (default) or 'binmajor' accumulation strategy
float32   run the geometry in single precision (default: False), for twice the
          lanes per vector; the accumulators stay double either way. Pairs near
          a bin edge can land in the neighbouring bin, so results differ at
          roughly the per-cent level, and count2 alone serves it. There is no
          CUDA counterpart: FLOAT is a compile-time choice there, not a
          runtime one.
"""

_TUNING_KEYS = {'cuda': CUDA_TUNING_KEYS, 'cpu': CPU_TUNING_KEYS}


# Mirrors ELLMAX in cuda/include/count3close.h. The kernel applies it as
# `ellmax = MIN(ellmax, ELLMAX)`, i.e. it clamps silently: asking for higher
# orders returns fewer poles than the binning describes, with no error. Keep in
# sync with the header (raising it also needs MMAX_SIZE = ELLMAX + 1).
KERNEL_ELLMAX = 5


def _check_tuning(backend, tuning):
    """Validate a tuning dict for one backend, rejecting unknown keys by name."""
    tuning = dict(tuning or {})
    unknown = set(tuning) - set(_TUNING_KEYS[backend])
    if unknown:
        extra = (' (kernel launch geometry is chosen by occupancy and is not exposed yet)'
                 if backend == 'cuda' else '')
        raise ValueError(f'{backend.upper()} backend tuning: unknown keys {sorted(unknown)}; '
                         f'accepted: {list(_TUNING_KEYS[backend])}{extra}')
    return tuning


def _unavailable(backend):
    """Return a reason string if a backend cannot run at all, else None.

    Nothing else is declined per backend. What neither kernel can serve is the
    same for both, so it is checked once in _check_count2 / _check_count3
    before either is chosen -- otherwise one backend would report a limit by
    name while the other went ahead and computed something else.
    """
    if backend == 'cpu' and cpulib is None:
        return 'not built (-DCUCOUNT_BUILD_CPU=ON)'
    if backend == 'cuda' and cudalib is None:
        return ('not built (-DCUCOUNT_BUILD_CUDA=ON), or the CUDA libraries are '
                "missing; only backend='cpu' can be served")
    return None


def _log_timings(backend, mesh_seconds, count_seconds):
    """The mesh/count split both bindings return, logged the same way.

    On CUDA the devices run concurrently, so these are the slowest of them
    rather than the sum; on the CPU the mesh build is still serial, so its
    share grows with the thread count.
    """
    logger.debug('%s backend: mesh %.1f ms, count %.1f ms',
                 backend, mesh_seconds * 1e3, count_seconds * 1e3)


def check_kernel_ells(*battrs_or_ells):
    """
    Raise if any requested multipole exceeds what the count3 kernel can compute.

    Without this the kernel clamps silently, and the mismatch surfaces far downstream -- as an
    opaque IndexError while packing the poles, or, worse, as counts quietly built from the wrong
    multipoles.
    """
    ells = [_get_ells(x) for x in battrs_or_ells if x is not None]  # battrs23 is optional
    ells = [ell for ell in ells if ell is not None and len(ell)]
    if not ells: return
    requested = max(max(np.ravel(ell)) for ell in ells)
    if requested > KERNEL_ELLMAX:
        raise ValueError(
            f'requested multipoles up to ell = {requested}, but the count3/count3close cucount kernel supports only '
            f'ell <= {KERNEL_ELLMAX} (ELLMAX in cuda/include/count3close.h) and would clamp silently. '
            f'Lower the requested multipoles, or rebuild cucount with a larger ELLMAX '
            f'(and MMAX_SIZE = ELLMAX + 1).')


# --- the calls themselves --------------------------------------------------
# ``cparticles`` are already-converted native Particles, one conversion serving
# both backends; the attrs cross into either extension through pybind's foreign
# module_local loading.

def _lib_and_nthreads(backend, tuning):
    """The extension to call, and the one tuning key whose meaning differs:
    nthreads is GPUs on one backend and CPU threads on the other."""
    if backend == 'cuda':
        return cudalib, tuning.get('nthreads', 1)
    return cpulib, int(tuning.get('nthreads') or cpu_nthreads())


@contextlib.contextmanager
def _pinned_isa(tuning):
    """Pin Highway to one ISA for the call, and restore it afterwards.

    The target is process-global state inside the extension, so it has to be
    put back even if the call raises. A CUDA request never arrives here with an
    isa: _check_tuning rejects the key for that backend.
    """
    isa = tuning.get('isa')
    if isa is None:
        yield
        return
    if cpulib.set_target(isa) is None:
        cpulib.set_target('')
        raise ValueError(f'CPU backend tuning: ISA {isa!r} unknown or unavailable; '
                         f'available: {cpulib.available_targets()}')
    try:
        yield
    finally:
        cpulib.set_target('')


def _count2(backend, cparticles, battrs, mattrs, wattrs=None, sattrs=None,
            spattrs=None, tuning=None):
    tuning = _check_tuning(backend, tuning)
    lib, nthreads = _lib_and_nthreads(backend, tuning)
    # Selections and splits go through too: the bindings lower them, and the
    # kernels apply a selection as a per-pair veto.
    kwargs = dict(wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
                  nthreads=nthreads, return_timings=True)
    if backend == 'cpu':
        kwargs['scatter'] = str(tuning.get('scatter', 'scalar'))
        kwargs['float32'] = bool(tuning.get('float32', False))
    with _pinned_isa(tuning):
        result, timings = lib.count2(*cparticles, mattrs._to_c(), battrs, **kwargs)
    _log_timings(backend, *timings)
    return result


def _count3(backend, cparticles, mattrs, battrs12, battrs13, wattrs=None,
            sattrs=None, vetos=None, tuning=None):
    tuning = _check_tuning(backend, tuning)
    lib, nthreads = _lib_and_nthreads(backend, tuning)
    result, timings = lib.count3(
        *cparticles, *[m._to_c() for m in mattrs], battrs12, battrs13,
        wattrs=wattrs._to_c(),
        sattrs12=sattrs[0], sattrs13=sattrs[1], veto12=vetos[0], veto13=vetos[1],
        nthreads=nthreads, return_timings=True)
    _log_timings(backend, *timings)
    return result


def _count3close(backend, cparticles, mattrs, battrs, wattrs=None, sattrs=None,
                 vetos=None, close_pair=(1, 2), tuning=None):
    tuning = _check_tuning(backend, tuning)
    lib, nthreads = _lib_and_nthreads(backend, tuning)
    # battrs[2] is the one that really can be None: the (2, 3) axis is optional.
    result, timings = lib.count3close(
        *cparticles, *[m._to_c() for m in mattrs], battrs[0], battrs[1], battrs[2],
        wattrs=wattrs._to_c(),
        sattrs12=sattrs[0], sattrs13=sattrs[1], sattrs23=sattrs[2],
        veto12=vetos[0], veto13=vetos[1], veto23=vetos[2],
        close_pair=close_pair, nthreads=nthreads, return_timings=True)
    _log_timings(backend, *timings)
    return result


def _resolve_backend(backend=None):
    """Return the backend to run: the argument, else $CUCOUNT_BACKEND, else 'cuda' when this
    build has the CUDA backend and the process sees a GPU, else 'cpu'.

    Mirrors cucount.jax's _resolve_ffi_backend, except that this one returns a
    name rather than an extension handle, because 'compare' needs both. Naming
    a backend is separate from requiring it (_require_backend): MeshAttrs
    resolves a name only to pick its cell size, and a mesh outlives the choice
    of kernel, so it must keep working on a build without that backend.
    """
    mode = backend or os.environ.get('CUCOUNT_BACKEND')
    if not mode:
        mode = 'cuda' if cudalib is not None and (cpulib is None or cudalib.device_count()) else 'cpu'
    mode = mode.lower()
    if mode not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}, got {mode!r}')
    return mode


def _require_backend(mode):
    """Raise unless every backend ``mode`` would run is actually built.

    Deliberately later than the tuning check in _resolve_tuning: a misspelled
    tuning key is a programming error and is reported by name on any machine,
    including one where that backend was never built.
    """
    for name in (('cuda', 'cpu') if mode == 'compare' else (mode,)):
        why = _unavailable(name)
        if why:
            raise NotImplementedError(f'{name.upper()} backend: {why}')


def _resolve_tuning(mode, tuning, nthreads=1):
    """Return (cuda_tuning, cpu_tuning) for the resolved backend mode.

    ``tuning`` is a flat dict addressed to the selected backend; ``compare``
    runs both backends, so there it must be nested as
    ``{'cpu': {...}, 'cuda': {...}}``. Only the nesting is decided here; each
    backend rejects its own unknown keys by name, eagerly, so a bad key raises
    before any kernel runs (and on a machine without that backend built).
    """
    tuning = dict(tuning or {})
    if mode == 'compare':
        unknown = set(tuning) - {'cuda', 'cpu'}
        if unknown:
            raise ValueError(
                f"backend='compare' runs both backends, so tuning must be nested as "
                f"{{'cpu': {{...}}, 'cuda': {{...}}}}; got extra keys {sorted(unknown)}")
        cuda, cpu = dict(tuning.get('cuda') or {}), dict(tuning.get('cpu') or {})
    elif mode == 'cuda':
        cuda, cpu = tuning, {}
    else:
        cuda, cpu = {}, tuning
    if nthreads != 1:
        warnings.warn("nthreads= is deprecated: pass tuning={'nthreads': ...} instead "
                      "(addressed to the selected backend, so it no longer only means GPUs)",
                      DeprecationWarning, stacklevel=3)
        cuda.setdefault('nthreads', nthreads)
    _check_tuning('cuda', cuda)
    _check_tuning('cpu', cpu)
    return cuda, cpu


# Limits of the kernels themselves, not of one backend. Both count2 kernels
# index a Legendre cache of MAX_POLE + 1 entries by ell and fill an ell list of
# MAX_POLE + 2; both look the count2 angular upweight up against a single
# cos(theta); both project spin from exactly two components; and neither
# triplet kernel weights by splits, by bitwise columns, or -- for count3 -- by
# an angular table.
#
# They are checked here, once, before a backend is chosen, so that the two
# decline the same requests for the same reasons. Previously only the CPU
# backend checked them and reported by name, while CUDA went ahead: it dropped
# a one-sided multipole projection silently, and with ell > MAX_POLE it indexed
# the Legendre cache out of bounds.

MAX_POLE = 8
"""Mirrors MAX_POLE in include/common.h: the size of both count2 kernels' Legendre cache."""


def _pole_values(battrs):
    """The ell values of a pole axis, or None when there is none."""
    for name, array in zip(battrs.varnames, battrs.array):
        if name == 'pole':
            return np.asarray(array)
    return None


def _check_count2(particles, battrs, wattrs, spattrs):
    """Raise for a count2 request neither kernel can serve."""
    ells = _pole_values(battrs)
    if ells is not None and ells.size:
        if ells.size > MAX_POLE + 2:
            raise ValueError(
                f'{ells.size} multipoles requested, but both count2 kernels hold the ell list '
                f'in a buffer of {MAX_POLE + 2} (MAX_POLE + 2 in include/common.h)')
        if ells.max() > MAX_POLE:
            raise ValueError(
                f'multipoles up to ell = {int(ells.max())} requested, but both count2 kernels '
                f'index a Legendre cache of {MAX_POLE + 1} entries by ell, so ell <= {MAX_POLE} '
                f'(MAX_POLE in include/common.h). Raise MAX_POLE and rebuild to go higher.')

    for p in particles:
        sizes = dict(p.index_value._sizes)
        if sizes.get('spin') and sizes['spin'] != 2:
            raise ValueError(
                f'spin with {sizes["spin"]} components, but the projection both kernels share '
                'takes exactly 2')
    nbitwise = [dict(p.index_value._sizes).get('bitwise_weight', 0) for p in particles]
    if len(set(nbitwise)) > 1:
        raise ValueError(
            f'catalogues carry {nbitwise} bitwise weights; both kernels pair them column by '
            'column, so the counts must match')

    angular = getattr(wattrs, 'angular', None)
    ndim = 0 if angular is None or angular.weight is None else angular.weight.ndim
    if ndim and ndim != 1:
        raise ValueError(
            f'{ndim}-dimensional angular weights, but count2 has one angle per pair and both '
            'kernels look the upweight up against a single cos(theta)')

    if getattr(spattrs, 'size', 0) > 1 and not all(
            dict(p.index_value._sizes).get('split') for p in particles):
        raise ValueError('jackknife splits need a split label on both catalogues')


def _check_count3(particles, battrs12, battrs13, battrs23, wattrs, close):
    """Raise for a count3 / count3close request neither kernel can serve."""
    poles = []
    for name, battrs in [('battrs12', battrs12), ('battrs13', battrs13)]:
        names = list(battrs.varnames)
        if names[:1] not in (['s'], ['theta']):
            raise ValueError(
                f'{name} bins in {names}, but both triplet kernels bin a leg in s or theta')
        if len(names) > 2 or (len(names) == 2 and names[1] != 'pole'):
            raise ValueError(
                f'{name} bins in {names}, but a triplet leg takes one separation axis and an '
                'optional multipole axis')
        poles.append(len(names) == 2)
    if poles[0] != poles[1]:
        # CUDA computed nprojs = 0 here and dropped the projection without a word.
        raise ValueError(
            'a multipole axis on one triplet leg only: both legs carry one, or neither')
    if poles[0]:
        check_kernel_ells(battrs12, battrs13)

    if battrs23 is not None and list(battrs23.varnames) not in (['s'], ['theta']):
        raise ValueError(
            f'battrs23 bins in {list(battrs23.varnames)}, but the (2, 3) axis bins in s or theta')

    angular = getattr(wattrs, 'angular', None)
    ndim = 0 if angular is None or angular.weight is None else angular.weight.ndim
    if ndim and not close:
        raise ValueError('neither count3 kernel applies angular weights; they belong to count3close')
    if ndim and ndim != 3:
        raise ValueError(
            f'{ndim}-dimensional angular weights, but count3close indexes the table by the '
            "triangle's three cos(theta)")

    bitwise = getattr(wattrs, 'bitwise', None)
    if bitwise is not None and getattr(bitwise, 'weights', None):
        raise ValueError('neither triplet kernel applies bitwise weights')
    for p in particles:
        if dict(p.index_value._sizes).get('split'):
            raise ValueError('neither triplet kernel splits; splits are a count2 feature')


def _dispatch(mode, call, cuda_tuning, cpu_tuning):
    """Run a count on the selected backend.

    ``call(backend, tuning)`` runs one backend; one callable rather than two
    means the two cannot drift apart. No backend is ever chosen implicitly:
    an unbuilt one raises here rather than silently falling back.
    """
    _require_backend(mode)
    if mode in ('cuda', 'cpu'):
        return call(mode, cuda_tuning if mode == 'cuda' else cpu_tuning)

    # compare: wall clock on both sides
    start = time.perf_counter()
    cpu = call('cpu', cpu_tuning)
    cpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reference = call('cuda', cuda_tuning)
    cuda_seconds = time.perf_counter() - start
    ratio = cuda_seconds / cpu_seconds if cpu_seconds > 0 else float('inf')
    logger.info('compare: cpu %.4f s, cuda %.4f s -- cpu %.2fx %s',
                cpu_seconds, cuda_seconds,
                ratio if ratio >= 1 else 1. / ratio,
                'faster' if ratio >= 1 else 'slower')

    from numpy.testing import assert_allclose

    for key, want in reference.items():
        assert_allclose(cpu[key], want, rtol=COMPARE_RTOL, atol=0,
                        err_msg=f'CPU and CUDA backends disagree on {key!r}')
    return reference


def _setup_cucount_logging():
    # Each extension holds its own copy of the log level; the backends sync
    # theirs if (and only if) they are loaded.
    level = _log_level_name()
    cucount_attrs.setup_logging(level)
    # Each extension holds its own copy; sync the ones that are built.
    for lib in (cudalib, cpulib):
        if lib is not None:
            lib.setup_logging(level)


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
        # A backend-neutral instance: the CUDA extensions load it through
        # pybind's foreign module_local casting.
        return cucount_attrs.WeightAttrs(**state)

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


class SelectionAttrs(cucount_attrs.SelectionAttrs):
    """
    Provide selection:
    - theta = (min, max)  # in degrees
    """


class SplitAttrs(cucount_attrs.SplitAttrs):
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


class BinAttrs(cucount_attrs.BinAttrs):
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


def _mesh_reach(sattrs=None, battrs=None):
    """Return the (mesh type, smax) a leg bounded by these attrs needs.

    The rule MeshAttrs sizes itself by, lifted out so a caller can ask what a
    leg requires without building a mesh for it. The binning and the selection
    each bound the leg, so on any variable they share the tighter wins. smax
    is a distance for a cartesian mesh and cos(theta_max) for an angular one,
    matching what MeshAttrs.smax carries; it is None when nothing here bounds
    the leg, and the mesh then has to reach across the whole box.
    """
    limits = {}
    for attrs in [sattrs, battrs]:
        if attrs is None: continue
        for name, lim in zip(attrs.varnames, attrs.max):
            if name in limits: limits[name] = min(limits[name], lim)
            else: limits[name] = lim

    mesh_type, mesh_smax = None, None
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

    if mesh_smax is None and all(name in limits for name in ['rp', 'pi']):
        mesh_smax = (limits['rp']**2 + limits['pi']**2)**0.5
    return mesh_type, mesh_smax


def _check_meshsize(meshsize):
    """Raise unless every axis has at least one cell.

    Only a caller-supplied meshsize can be short: both derived branches clamp
    what they compute. A mesh of no cells is refused rather than repaired, so
    the two backends meet it the same way -- neither kernel carries a
    resolution heuristic to fall back on, and the cell size divides by it.
    Checked before pixel_resolution and cellsize, which would divide by zero
    first and warn.
    """
    if np.any(np.asarray(meshsize) < 1):
        raise ValueError(f'meshsize must be at least 1 on every axis, got {list(meshsize)}')


def _check_mesh_reaches(mattrs, name, leg, sattrs=None, battrs=None):
    """Raise unless ``mattrs`` sweeps wide enough to bound ``leg``.

    A window narrower than the leg it is walked over drops pairs rather than
    rejecting them, so the count comes back quietly low. The defaults built
    below are always wide enough; this is for a mesh passed in by hand.
    Nothing is checked when the leg is unbounded (the mesh spans the box
    anyway) or when it is bounded in a variable this mesh type cannot express,
    which leaves the walk complete by being maximally wide.
    """
    want_type, want_smax = _mesh_reach(sattrs, battrs)
    if want_smax is None or want_type != mattrs.type:
        return
    # An angular mesh carries cos(theta_max), so there wider means smaller.
    # The slack is additive on that side: a cosine is bounded, and scaling it
    # would tighten rather than loosen the whole southern half of the range.
    wide_enough = (mattrs.smax <= want_smax + 1e-9 if want_type == 'angular'
                   else mattrs.smax >= want_smax * (1. - 1e-9))
    if not wide_enough:
        unit = 'cos(theta)' if want_type == 'angular' else 'separation'
        raise ValueError(
            f'{name} sweeps to {unit} {mattrs.smax!r}, too narrow for the {leg} leg it is '
            f'walked over, which reaches {want_smax!r}; pairs would be dropped rather than '
            f'rejected. Pass a mesh built from the {leg} binning and selection, or let '
            f'{name}=None build one.')


@dataclass(init=False)
class MeshAttrs(object):

    boxsize: np.ndarray
    boxcenter: np.ndarray
    meshsize: np.ndarray
    type: str
    periodic: bool
    smax: float
    _np = np

    def __init__(self, *positions, boxsize=None, boxcenter=None, meshsize=None, refine=1., backend=None, battrs=None, sattrs=None, periodic=False):
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
        backend : str, optional
            Backend the mesh is built for, 'cuda' or 'cpu' (default: the CUCOUNT_BACKEND
            variable, else 'cuda' if a GPU is visible, else 'cpu'). This sets the default resolution only, which ``refine``
            then scales and an explicit ``meshsize`` overrides; it is not stored on the
            instance. A mesh built for one backend stays correct on the other, only slower.
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

        mesh_type, mesh_smax = _mesh_reach(sattrs, battrs)
        assert mesh_type is not None, 'cannot determine mesh type from sattrs or battrs; provide at least one'
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

        # Target number of cells per maximum separation, smax / cellsize, before the
        # O(nparticles) cap and refine are applied. Only running time depends on it: the
        # candidate window widens as the cells shrink, so every value gives the same counts.
        # The trade-off is wasted distance evaluations against per-cell overhead. Finer
        # cells fit the swept volume to the sphere of radius smax more tightly -- at 6
        # cells per smax, 41% of the candidates examined lie within smax, against 15% at
        # 1 -- but there are more cells to walk and fewer particles in each. The CUDA
        # kernel wants the tighter fit, since each candidate costs a load and the per-cell
        # bookkeeping is amortised across a warp; the CPU kernel wants the opposite, its
        # inner loop being fast only while it can load full SIMD vectors.
        # The cartesian CPU value is measured, on a 1000 box with smax = 100 and 32
        # threads, timing the kernel against resolution with meshsize set explicitly so
        # the O(nparticles) cap does not confound it (cpu/bench_mesh.py). Against c = 1, the
        # mesh the kernel used to pick for itself:
        #   n = 50k    c=2 0.53x   c=3 0.64x   c=6 1.07x
        #   n = 200k   c=2 0.68x   c=3 0.67x   c=6 1.44x
        #   n = 500k   c=2 0.63x   c=3 0.67x   c=6 0.84x
        # 2 is the only value that wins at every size: 6 is good only once the cells are
        # dense enough to fill a vector, and 1 leaves a factor of 1.5 on the table
        # everywhere. The angular CPU value is untested and keeps the CUDA one.
        # 'compare' runs both on one mesh and so cannot suit either: anchor it on the
        # reference implementation, which costs the comparison nothing but timings.
        _backend = _resolve_backend(backend)
        cells_per_smax = {'cartesian': {'cuda': 6., 'cpu': 2.},
                          'angular': {'cuda': 5., 'cpu': 5.}}[mesh_type][
                              'cuda' if _backend == 'compare' else _backend]

        # Now set up resolution meshsize
        if mesh_type == 'angular':
            if meshsize is None:
                theta_max = np.arccos(mesh_smax)
                nside1 = cells_per_smax * (np.pi / theta_max)
                fsky = boxsize.prod() / (4 * np.pi)
                nside2 = np.minimum(self._np.sqrt(0.25 * nparticles / fsky), 2048)
                meshsize = np.maximum(np.minimum(nside1, nside2) * refine, 1).astype(int)
                meshsize = [meshsize, 2 * meshsize]
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            _check_meshsize(meshsize)
            pixel_resolution = np.degrees(np.sqrt(4 * np.pi / meshsize.prod()))
            logger.debug("Mesh size is %d = %d x %d.", meshsize.prod(), meshsize[0], meshsize[1])
            logger.debug("Pixel resolution is %.4lf deg.", pixel_resolution)
        elif mesh_type == 'cartesian':
            nside2 = (0.5 * nparticles)**(1. / 3.)
            if meshsize is None:
                nside1 = cells_per_smax * boxsize / mesh_smax
                meshsize = np.maximum(np.minimum(nside1, nside2) * refine, 1).astype(int)
            meshsize = np.array(meshsize, dtype=np.int64) * np.ones(ndim, dtype=np.int64)
            _check_meshsize(meshsize)
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
        return cucount_attrs.MeshAttrs(**state)


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
            value = np.asarray(value).astype(np.float64)
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
            weight = np.asarray(weight)
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


def _to_c_particles(p):
    """Backend-neutral native Particles: the identical packed-values layout is
    consumed by the CUDA and CPU extensions alike (through pybind's foreign
    module_local loading)."""
    return cucount_attrs.Particles(p.positions, values=_stack_values(p.values, np=np), **p.index_value._to_c())


def count2(*particles: Particles, battrs: BinAttrs, wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None,
           spattrs: SplitAttrs=None, mattrs: MeshAttrs=None, nthreads: int=1, backend: str=None,
           tuning: dict=None):
    """
    Perform two-point pair counts using the native cucount library.

    This is a thin frontend that prepares Python-side Particles and Weight/Selection
    attributes and calls the underlying cucountlib.cuda.count2 implementation
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
        Deprecated: number of GPUs (within the same node) to run in parallel on.
        Use ``tuning={'nthreads': ...}`` instead.
    backend : str, optional
        Override the CUCOUNT_BACKEND environment variable for this call:
        'cuda', 'cpu' or 'compare' (default: 'cuda' if a GPU is visible, else 'cpu'). See BACKENDS.
    tuning : dict, optional
        Tuning options for the selected backend, passed through opaquely;
        unknown keys are rejected by name. CUDA accepts ``nthreads`` (number of
        GPUs); the CPU backend accepts ``nthreads`` (CPU threads), ``isa``
        (e.g. 'AVX2', pinning Highway for this call) and ``scatter``
        ('scalar' or 'binmajor'). With ``backend='compare'``, nest per backend:
        ``{'cpu': {...}, 'cuda': {...}}``. None of them changes the result.

        On ``scatter``: SIMD can compute a vector of bin indices but cannot
        scatter-accumulate into a histogram, since two lanes may fall in the
        same bin. 'scalar' (the default) spills the vector to a buffer and adds
        the lanes one at a time, costing O(lanes) per vector whatever the bin
        count. 'binmajor' instead keeps a per-lane replica of the whole
        histogram and sweeps it with masked vector adds, costing O(nbins) per
        vector and never leaving SIMD. So 'binmajor' pays only when there are
        fewer bins than lanes -- a handful of ``s`` bins and nothing else --
        and loses on anything wider, which is why it is not the default. It
        also applies to the plain ``w1 * w2`` path only: spin, PIP, negative or
        angular weights, a selection and multipoles all give a pair its own
        weight, which one number per (bin, lane) cannot hold, so those go to
        'scalar' regardless of what is asked for.

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

    # Before the mesh: a default MeshAttrs picks its resolution for the backend
    # that will run.
    _check_count2(particles, battrs, wattrs, spattrs)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

    if mattrs is None: mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)

    # One conversion serves both backends: attrs-module Particles cast into
    # either extension through pybind's foreign module_local loading.
    cparticles = [_to_c_particles(p) for p in particles]

    return _dispatch(
        mode,
        lambda backend, tuning: _count2(backend, cparticles, battrs, mattrs,
                                        wattrs=wattrs, sattrs=sattrs,
                                        spattrs=spattrs, tuning=tuning),
        cuda_tuning, cpu_tuning)


def _get_ells(battrs):
    if isinstance(battrs, BinAttrs):
        try:
            ells = battrs.coords('pole')
        except (ValueError, IndexError):
            ells = []
    else:
        ells = battrs
    return [int(ell) for ell in ells]


def poles_to_ells(ells1, ells2, with_prefactor: bool=True):
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
    if not with_prefactor:
        ells = [ell[1:] for ell in ells]
    return ells


def symmetrize_poles(poles, ells1, ells2, axis=-1, np=np):
    """
    Symmetrize pole coefficients following Eq. 9 of https://arxiv.org/pdf/1709.10150
    retaining only real-valued positive-m coefficients.

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

    # The jax path binds np=jnp, and jnp.take CLIPS out-of-range indices instead of raising
    npoles = poles.shape[axis]
    if keep.size and int(keep.max()) >= npoles:
        raise ValueError(f'pole axis has {npoles} entries but poles_to_ells(ells1, ells2) describes '
                         f'{len(labels)} (needing index {int(keep.max())}); ells1 = {list(ells1)}, '
                         f'ells2 = {list(ells2)} do not match the counts that were computed')

    sym = np.take(poles, keep, axis=axis)

    shape = [1] * sym.ndim
    shape[axis % sym.ndim] = factors.size
    sym = sym * factors.reshape(shape)

    return sym, out_labels


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
                nthreads: int = 1,
                backend: str = None,
                tuning: dict = None):
    """
    Perform close-triplet counts using the native cucount library.

    This is a thin frontend that prepares Python-side ``Particles`` and
    weight/selection attributes and calls the underlying
    ``cucountlib.cuda.count3close`` implementation.

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
        Deprecated: number of GPUs (within the same node) to run in parallel on.
        Use ``tuning={'nthreads': ...}`` instead.
    tuning : dict, optional
        Tuning options for the (CUDA-only) backend; accepts ``nthreads``
        (number of GPUs). Unknown keys are rejected by name.

    Returns
    -------
    dict
        Output of the native ``count3close`` call.
        Typically a dictionary such as::

            {"weight": array}
    """
    _check_count3(particles, battrs12, battrs13, battrs23, wattrs, close=True)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

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

    # Which pair the default meshes are sized for. close_pair names the search
    # strategy, and each strategy bounds a different leg: CUDA's (2, 3) walks
    # catalogue 3 from each particle 2, so there the tight window belongs to
    # the (2, 3) leg. The CPU backend has one traversal -- walk 2, then 3,
    # from each primary -- so whatever close_pair says, it needs the (1, 2)
    # and (1, 3) windows, and a mesh sized for (2, 3) makes it sweep too
    # narrowly and silently drop triplets. (1, 2) sizing is correct for every
    # strategy, since a wider window only costs candidates that the binning
    # and the selections then reject; it is merely not the tightest for CUDA.
    # 'compare' runs both backends on one mesh set, so it takes the sizing
    # that serves both.
    mesh_pair = close_pair if mode == 'cuda' else (1, 2)

    if mattrs1 is None:
        mattrs1 = MeshAttrs(
            particles[0],
            sattrs=sattrs13 if mesh_pair == (1, 3) else sattrs12,
            battrs=battrs13 if mesh_pair == (1, 3) else battrs12,
            backend=mode,
        )

    if mattrs2 is None:
        mattrs2 = MeshAttrs(
            particles[1],
            sattrs=sattrs23 if mesh_pair == (2, 3) else sattrs12,
            battrs=battrs23 if mesh_pair == (2, 3) else battrs12,
            backend=mode,
        )

    if mattrs3 is None:
        mattrs3 = MeshAttrs(
            particles[2],
            sattrs=sattrs23 if mesh_pair == (2, 3) else sattrs13,
            battrs=battrs23 if mesh_pair == (2, 3) else battrs13,
            backend=mode,
        )

    # The defaults above are already wide enough; this catches a mesh passed
    # in by hand, which the CPU traversal would otherwise walk too narrowly.
    if mode != 'cuda':
        _check_mesh_reaches(mattrs2, 'mattrs2', '(1, 2)', sattrs12, battrs12)
        _check_mesh_reaches(mattrs3, 'mattrs3', '(1, 3)', sattrs13, battrs13)

    cparticles = [_to_c_particles(p) for p in particles]
    mattrs = (mattrs1, mattrs2, mattrs3)
    battrs = (battrs12, battrs13, battrs23)
    sattrs = (sattrs12, sattrs13, sattrs23)
    vetos = (veto12, veto13, veto23)

    return _dispatch(
        mode,
        lambda backend, tuning: _count3close(backend, cparticles, mattrs, battrs,
                                             wattrs=wattrs, sattrs=sattrs, vetos=vetos,
                                             close_pair=close_pair, tuning=tuning),
        cuda_tuning, cpu_tuning)


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
           nthreads: int = 1,
           backend: str = None,
           tuning: dict = None):
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
        Deprecated: number of GPUs within the same node to run in parallel on.
        Use ``tuning={'nthreads': ...}`` instead.
    tuning : dict, optional
        Tuning options for the (CUDA-only) backend; accepts ``nthreads``
        (number of GPUs). Unknown keys are rejected by name.

    Returns
    -------
    dict
        Output of the native ``count3`` call, typically ``{"weight": array}``.
    """
    _check_count3(particles, battrs12, battrs13, None, wattrs, close=False)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

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
        mattrs1 = MeshAttrs(particles[0], sattrs=sattrs12, battrs=battrs12, backend=mode)
    if mattrs2 is None:
        mattrs2 = MeshAttrs(particles[1], sattrs=sattrs12, battrs=battrs12, backend=mode)
    if mattrs3 is None:
        mattrs3 = MeshAttrs(particles[2], sattrs=sattrs13, battrs=battrs13, backend=mode)

    cparticles = [_to_c_particles(p) for p in particles]
    mattrs = (mattrs1, mattrs2, mattrs3)

    return _dispatch(
        mode,
        lambda backend, tuning: _count3(backend, cparticles, mattrs, battrs12, battrs13,
                                        wattrs=wattrs, sattrs=(sattrs12, sattrs13),
                                        vetos=(veto12, veto13), tuning=tuning),
        cuda_tuning, cpu_tuning)


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
    Perform pair counts analytically for periodic boxes.

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



prod = functools.partial(functools.reduce, operator.mul)


def count3_analytic(battrs12: BinAttrs, battrs13: BinAttrs, mattrs: MeshAttrs=None):
    """
    Perform triplet counts analytically for periodic boxes.

    Parameters
    ----------
    battrs12, battrs13 : BinAttrs
        Binning specification (edges/shape) for the pair counts.
    mattrs : MeshAttrs or array, optional
        Mesh attributes (boxsize).

    Returns
    -------
    counts : array
        Normalized analytical triplet counts in each bin.
    """
    boxsize = getattr(mattrs, 'boxsize', mattrs) * np.ones(3, dtype=np.float64)
    dvs = []
    for battrs in [battrs12, battrs13]:
        edges = battrs.edges()
        mode = tuple(edges)
        if mode == ('s',) or mode == ('s', 'pole'):
            v = 4. / 3. * np.pi * edges['s']**3
            dv = np.diff(v, axis=-1)
        else:
            raise NotImplementedError('No analytic pair counter provided for binning {}'.format(mode))
        dv /= boxsize.prod()
        dvs.append(dv)
    dv = prod(np.meshgrid(*dvs, indexing='ij', sparse=True))
    ells = poles_to_ells(battrs12, battrs13, with_prefactor=False)
    if ells:
        factor = np.zeros(len(ells))
        ell0 = (0, 0, 0)
        if ell0 in ells:
            factor[ells.index(ell0)] = 1.
        dv = dv[..., None] * factor
    return dv