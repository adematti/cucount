"""Soft (Gaussian-softmax) two-point pair counts.

This is an *extension* to :mod:`cucount.numpy`, not a replacement: imports the
hard-binning frontend and adds one entry point beside it. The hard ``count2``
is untouched -- no option, flag or branch was added to it -- so an existing
call cannot change meaning or speed because this module exists.

Semantics follow the pyfcfc oracle (``FCFC-main/src/fcfc/2pt/soft_hist.c``):
every pair inside the separation range is spread over the separation bins with
the softmax of a Gaussian kernel centred on the bin centres,

    x_i     = (s - c_i)^2 / (2 sigma_i^2),   c_i = (edges[i] + edges[i+1]) / 2
    W_i     = exp(-x_i) / sum_j exp(-x_j)
    counts += w_pair * W_i

so a pair's weight is exactly conserved (the row sums to one), and sigma -> 0
recovers hard binning.

``sigma=None`` means "half a bin", the oracle's default, and on a non-uniform
grid it means half of *each* bin, ``sigma_i = 0.5 * (edges[i+1] - edges[i])``.
That keeps the smoothing proportional to the local resolution (a log grid gets
constant relative smoothing instead of over-smoothing its narrow bins) and makes
sigma a property of the binning rather than a free parameter. Passing a number
instead uses that absolute width in separation units for every bin.

Both uniform and arbitrary monotone edges are served: the uniform grid keeps a
multiplicative recurrence (two ``exp`` per pair), any other grid uses a binary
search for the peak plus one ``exp`` per tap.

Differences from the oracle, both because the kernel is new rather than a port:

* only the ``s`` axis is softened (the oracle now also softens ``mu``/``pi``);
* the kernel is a CUDA kernel with a shared-memory histogram tile, not the
  oracle's scalar CPU loop;
* weights outside ``soft_cutoff`` of the peak are dropped and the window is
  renormalised, as the oracle does. Its default cutoff is 1e-13, which is what
  makes the tap count (and so the cost) ~9 taps per pair at the default sigma.
"""
import numpy as np

from . import (
    BinAttrs,
    MeshAttrs,
    Particles,
    SelectionAttrs,
    SplitAttrs,
    WeightAttrs,
    _check_count2,
    _dispatch,
    _lib_and_nthreads,
    _log_timings,
    _resolve_backend,
    _resolve_tuning,
    _setup_cucount_logging,
    _to_c_particles,
)


def _bin_centres(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('soft binning needs a 1D array of at least two edges')
    return 0.5 * (edges[1:] + edges[:-1])


def default_sigma(edges):
    """Half the mean bin width, which is the oracle's default (``sigma = 0``).

    Only meaningful for a uniform grid; on a non-uniform one the kernel uses a
    per-bin sigma (see :func:`sigmas`).
    """
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * float(edges[-1] - edges[0]) / (edges.size - 1)


def is_uniform(edges):
    """True when the edges are an arithmetic progression (to 1e-9 relative)."""
    edges = np.asarray(edges, dtype=np.float64)
    ds = edges[1] - edges[0]
    want = edges[0] + np.arange(edges.size) * ds
    return bool(np.all(np.abs(edges - want) <= 1e-9 * abs(ds)))


def sigmas(edges, sigma=None):
    """Per-bin kernel widths the kernel will use: ``sigma_i``, shape ``(nbins,)``.

    ``None`` gives half of each bin's width; a number is broadcast to every bin.
    """
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    if sigma is None:
        return 0.5 * widths
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f'sigma must be positive and finite, got {sigma!r}')
    return np.full(widths.shape, sigma, dtype=np.float64)


def soft_weights(s, edges, sigma=None):
    """Reference weights, shape ``(len(s), len(edges) - 1)``.

    Full-grid stable softmax over the bin centres -- no window truncation, so
    this is the definition rather than the kernel's truncated version; the two
    agree to the cutoff tail. Follows the kernel's sigma rule: ``None`` means
    half of each bin's width, a number means one absolute width.
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    centres = _bin_centres(edges)
    sg = sigmas(edges, sigma)
    x = (s[:, None] - centres[None, :]) ** 2 / (2.0 * sg[None, :] ** 2)
    x -= x.min(axis=1, keepdims=True)
    e = np.exp(-x)
    return e / e.sum(axis=1, keepdims=True)


def count2_soft(*particles: Particles, battrs: BinAttrs, soft_sigma: float=None,
                soft_cutoff: float=1e-13, soft_taps: str='exact',
                wattrs: WeightAttrs=None,
                sattrs: SelectionAttrs=None, spattrs: SplitAttrs=None,
                mattrs: MeshAttrs=None, nthreads: int=1, backend: str=None,
                tuning: dict=None):
    """Two-point pair counts with Gaussian-softmax soft binning on ``s``.

    Parameters
    ----------
    *particles : Particles
        Exactly two catalogs, as for :func:`cucount.numpy.count2`.
    battrs : BinAttrs
        Must be the single separation axis: ``BinAttrs(s=edges)``. Any strictly
        increasing edges are served. A uniform grid (``np.arange``, a
        ``(min, max, step)`` tuple, or ``BIN_LIN``) hits the fast recurrence
        kernel; anything else uses the general path, which is exact but pays one
        ``exp`` per tap instead of two per pair.
    soft_sigma : float, optional
        Kernel width in separation units. ``None`` (or 0) means half a bin; on a
        non-uniform grid that is half of each bin, i.e. ``sigma_i`` tracks the
        local width.
    soft_cutoff : float, optional
        Relative weight below which the Gaussian tail is dropped (the window is
        renormalised, so weight stays conserved). This is the speed knob: the
        tap count grows as ``sqrt(2 ln(1 / cutoff))``. Default 1e-13 matches
        the oracle.
    soft_taps : {'exact', 'fast'}
        Only affects the general (non-uniform) path. ``'exact'`` (default) uses
        double ``exp``; ``'fast'`` uses ``__expf``, trading ~1e-6 relative
        accuracy in the weights for roughly half the transcendental cost.
    wattrs, sattrs, spattrs, mattrs, nthreads, backend, tuning
        As for :func:`cucount.numpy.count2`; the soft kernel is CUDA-only for
        now, and rejects spin, angular upweighting, split/jackknife and any
        second binning axis rather than quietly ignoring them.

    Returns
    -------
    result : dict
        ``{'weight': array(nbins)}`` plus ``result['soft']`` describing what ran
        (``sigma`` after the default, ``sigma_kind`` = 'absolute' or 'width',
        ``uniform``, ``fast_taps``, ``cutoff``, ``kmax``, ``taps`` (``-1`` when
        the window is adaptive), ``path`` = 'shared' or 'global', ``nbins``).
    """
    _setup_cucount_logging()
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    if spattrs is None: spattrs = SplitAttrs()
    wattrs.check(*particles)
    spattrs.check(*particles)
    _check_count2(particles, battrs, wattrs, spattrs)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)

    cparticles = [_to_c_particles(p) for p in particles]

    def run(which, tuning):
        if which != 'cuda':
            raise NotImplementedError(
                'soft binning is CUDA-only for now; the hard cpu backend is '
                'unchanged and count2() still serves it')
        lib, lib_nthreads = _lib_and_nthreads(which, tuning)
        result, timings = lib.count2_soft(
            *cparticles, mattrs._to_c(), battrs,
            wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
            soft_sigma=soft_sigma, soft_cutoff=float(soft_cutoff),
            soft_taps=str(soft_taps),
            nthreads=lib_nthreads, return_timings=True)
        _log_timings(which, *timings)
        return result

    return _dispatch(mode, run, cuda_tuning, cpu_tuning)


def count2_soft_vjp(*particles: Particles, battrs: BinAttrs, cotangent,
                    soft_sigma: float=None, soft_cutoff: float=1e-13,
                    soft_taps: str='exact', wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None,
                    spattrs: SplitAttrs=None, mattrs: MeshAttrs=None,
                    nthreads: int=1, backend: str=None, tuning: dict=None):
    """Vector-Jacobian product of :func:`count2_soft`.

    Given ``g = dL/dcounts`` for the soft histogram, returns the gradients of
    ``L`` with respect to the two catalogues' particle positions and individual
    weights, and with respect to ``soft_sigma``. This is what makes the soft
    counts differentiable end to end: the jax layer wraps the forward call in a
    ``custom_vjp`` and hands the incoming cotangent to this kernel, so no
    scatter is differentiated by autodiff and the whole backward pass is one
    more kernel.

    The kernel is the analytic adjoint of the forward (the same window, the same
    softmax, differentiated exactly); for the derivation and the conventions --
    window membership and the ``[smin, smax)`` range check held constant -- see
    ``cuda/include/count2_soft.h``. This is the same adjoint pyfcfc exposes as
    ``soft_hist_vjp``.

    Parameters
    ----------
    *particles : Particles
        The two catalogs, exactly as passed to :func:`count2_soft` (same
        ``battrs``, ``soft_sigma``, weighting, selections and mesh, or the
        adjoint is of a different function than the forward you ran).
    battrs, soft_sigma, soft_cutoff, soft_taps, wattrs, sattrs, spattrs, mattrs
        As for :func:`count2_soft`; they must reproduce the forward's setup.
    cotangent : array_like
        One entry per separation bin, ``dL/dcounts``.

    Returns
    -------
    result : dict
        ``grad_position1`` (n1, 3), ``grad_position2`` (n2, 3),
        ``grad_weight1`` (n1,), ``grad_weight2`` (n2,) (``None`` when the
        catalogue carries no individual weights), ``grad_sigma`` (float) and
        ``result['soft']`` as in the forward.

    Notes
    -----
    For an autocorrelation -- the same catalog passed twice -- the derivative
    with respect to that one catalog is ``grad_position1 + grad_position2``:
    each is the derivative of the ordered-pair sum with respect to one of its
    two slots. The forward enumerates both directions, so the sum is the total.
    """
    _setup_cucount_logging()
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    if spattrs is None: spattrs = SplitAttrs()
    wattrs.check(*particles)
    spattrs.check(*particles)
    _check_count2(particles, battrs, wattrs, spattrs)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)

    cparticles = [_to_c_particles(p) for p in particles]

    cotangent = np.ascontiguousarray(cotangent, dtype=np.float64)

    def run(which, tuning):
        if which != 'cuda':
            raise NotImplementedError(
                'soft binning is CUDA-only for now; the hard cpu backend is '
                'unchanged and count2() still serves it')
        lib, lib_nthreads = _lib_and_nthreads(which, tuning)
        result, timings = lib.count2_soft_vjp(
            *cparticles, mattrs._to_c(), battrs, cotangent,
            wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
            soft_sigma=soft_sigma, soft_cutoff=float(soft_cutoff),
            soft_taps=str(soft_taps),
            nthreads=lib_nthreads, return_timings=True)
        _log_timings(which, *timings)
        return result

    return _dispatch(mode, run, cuda_tuning, cpu_tuning)
