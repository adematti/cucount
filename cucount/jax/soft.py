"""Soft (Gaussian-softmax) pair counts with an analytic adjoint, for jax.

This is an extension of :mod:`cucount.jax`, not a mode of it: nothing in the hard
``count2`` path changes, and this module is free to import or not.

Semantics match :func:`cucount.numpy.soft.count2_soft` (which follows the pyfcfc
oracle): every pair inside the separation range is spread over the bins with the
softmax of a Gaussian centred on the bin centres,

    x_i = (s - c_i)^2 / (2 sigma^2),  W_i = exp(-x_i) / sum_j exp(-x_j),
    counts_i += w_pair * W_i,

so each pair's total weight is conserved and ``sigma -> 0`` recovers hard
binning. Only the separation axis is softened, and only CUDA serves it so far.

What is new here is differentiability. The counts are wrapped in
``jax.custom_vjp`` whose backward calls the CUDA adjoint kernel
(``count2_soft_vjp``, the same VJP pyfcfc exposes as ``soft_hist_vjp``): the
windowed-softmax forward is differentiated in closed form, so one backward pass
costs about one forward pass instead of carrying per-pair taps through autodiff.
Gradients are exact w.r.t. positions and individual weights. The kernel also
returns a ``sigma`` cotangent, but ``soft_sigma`` is a static argument here, so
it is discarded.
"""

import numpy as np
import jax
from jax import numpy as jnp

from . import Particles, WeightAttrs, MeshAttrs, SelectionAttrs, SplitAttrs, BinAttrs
from . import _resolve_ffi_backend, _stack_values
from ..numpy.soft import default_sigma, is_uniform

NDIM = 3


def _s_edges(battrs):
    """The single separation edge array of a 1-D ``BinAttrs``."""
    names = list(battrs.varnames)
    if names != ['s']:
        raise NotImplementedError(
            'count2_soft softens the separation axis only, so battrs must be a single '
            f's axis, e.g. BinAttrs(s=edges); got axes {names}')
    intervals = np.asarray(battrs.edges('s'), dtype=np.float64)
    edges = np.concatenate([intervals[:, 0], intervals[-1:, 1]])
    if edges.size < 2:
        raise ValueError('need at least two separation edges')
    ds = np.diff(edges)
    if not np.all(ds > 0):
        raise ValueError('separation edges must be strictly increasing')
    # Both uniform and arbitrary monotone edges are served (the kernel switches
    # from the recurrence to the general path), so there is nothing to reject.
    return edges


def _soft_spec(edges, soft_sigma=None, soft_cutoff=1e-13, soft_taps='exact'):
    """Resolve the soft request the way the kernel will (sigma, window, taps)."""
    if not 0. < soft_cutoff < 1.:
        raise ValueError(f'soft_cutoff must be in (0, 1), got {soft_cutoff!r}')
    if soft_taps not in ('exact', 'fast'):
        raise ValueError(f"soft_taps must be 'exact' or 'fast', got {soft_taps!r}")
    nbins = edges.size - 1
    ds = edges[1] - edges[0]
    uniform = is_uniform(edges)
    width_scaled = soft_sigma is None and not uniform
    sigma = (0.5 * ds) if soft_sigma is None else float(soft_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f'soft_sigma must be positive and finite, got {soft_sigma!r}')
    if uniform:
        kmax = int(np.ceil(np.sqrt(2. * np.log(1. / soft_cutoff)) * sigma / ds))
        kmax = max(1, min(kmax, nbins - 1))
    else:
        kmax = nbins - 1  # adaptive: the kernel stops on the cutoff
    return dict(sigma=sigma, sigma_units='separation',
                sigma_kind='width' if width_scaled else 'absolute',
                uniform=uniform, fast_taps=soft_taps == 'fast',
                cutoff=float(soft_cutoff), kmax=kmax,
                taps=(2 * kmax + 1) if uniform else -1, nbins=nbins)


def _vjp_bufsize(particles, mattrs, nbins=0):
    """Scratch bytes the adjoint asks XLA for.

    The adjoint allocates no histogram of its own (the cotangent is an input), but
    it keeps a slot -> input-index map per mesh so gradients reach the right
    particle after ``set_mesh`` reorders them, plus the reordered particle arrays.
    """
    meshsize = int(np.prod(np.asarray(mattrs.meshsize, dtype=np.int64)))
    n = [int(p.positions.shape[0]) for p in particles]
    return (2 * meshsize  # nparticles + cumnparticles, both meshes
            + sum(n) * (2 * (NDIM + 1) + 1)  # positions+spositions, values, index map
            + 2 * meshsize * 16  # launch bookkeeping and slack
            + 3 * nbins + 1  # centres, 1/(2 sigma_i^2) and the edges, non-uniform only
            ) * 8 + 65536


def count2_soft(*particles: Particles, battrs: BinAttrs, soft_sigma: float = None,
                soft_cutoff: float = 1e-13, soft_taps: str = 'exact',
                wattrs: WeightAttrs = None,
                sattrs: SelectionAttrs = None, spattrs: SplitAttrs = None,
                mattrs: MeshAttrs = None, backend: str = None):
    """Differentiable soft pair counts.

    Drop-in neighbour of :func:`cucount.jax.count2` for the isotropic soft case:
    same particle containers, same ``{'weight': counts}`` return, and it can be
    differentiated with ``jax.grad``/``jax.value_and_grad`` w.r.t. positions and
    individual weights. The backward pass is the CUDA adjoint kernel, so a
    gradient costs roughly one extra forward pass.

    Parameters
    ----------
    *particles : Particles
        One (autocorrelation) or two Particles. Autocorrelation is expressed
        internally as ``(p, p)``; jax sums the two cotangents for the shared
        operand, so ``jax.grad`` gives the exact total derivative.
    battrs : BinAttrs
        A single ``BinAttrs(s=edges)`` with strictly increasing edges; uniform
        grids take the fast recurrence kernel, other grids the general path.
    soft_sigma : float, optional
        Gaussian width in separation units. Default: half a bin (the pyfcfc
        default), which on a non-uniform grid means half of *each* bin, so the
        smoothing follows the local resolution.
    soft_cutoff : float, optional
        Taps whose normalized weight falls below this are dropped and the
        remaining window renormalized, so the weight sum stays conserved to
        machine precision while the tap count stays
        ``~2 sqrt(2 ln(1/cutoff)) sigma/ds + 1``. Precision/speed knob: 1e-13
        reproduces the full softmax to double precision, 1e-6 is a fast
        production choice.
    wattrs, sattrs, spattrs, mattrs, backend
        As in :func:`cucount.jax.count2`. Spin, angular upweighting,
        split/jackknife and a second binning axis are rejected: the soft kernel
        does not serve them.

    Returns
    -------
    dict
        ``{'weight': counts, 'soft': spec}``: fractional, differentiable counts
        shaped ``(nbins,)`` plus the resolved soft parameters.
    """
    mode, ffi_module, suffix = _resolve_ffi_backend(backend)
    if mode != 'cuda':
        raise NotImplementedError('soft binning is served by the CUDA backend only for now')
    if len(particles) == 1:
        particles = (particles[0], particles[0])
    if len(particles) != 2:
        raise ValueError('count2_soft takes one or two Particles')

    wattrs = WeightAttrs() if wattrs is None else wattrs
    sattrs = SelectionAttrs() if sattrs is None else sattrs
    spattrs = SplitAttrs() if spattrs is None else spattrs
    if wattrs.spin is not None:
        raise NotImplementedError('spin weights are not served by the soft kernel yet')
    if wattrs.angular is not None:
        raise NotImplementedError('angular upweighting is not served by the soft kernel yet')
    if getattr(spattrs, 'nsplits', 0):
        raise NotImplementedError('split/jackknife counts are not served by the soft kernel yet')

    edges = _s_edges(battrs)
    spec = _soft_spec(edges, soft_sigma, soft_cutoff, soft_taps)
    nbins = spec['nbins']

    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)

    # Attribute state is process-global in the FFI module (as it is for count2),
    # so the soft request is resolved once and read by both fwd and bwd handlers.
    ffi_module.set_count2_soft_attrs(mattrs._to_c(), battrs, wattrs=wattrs._to_c(),
                                     sattrs=sattrs, spattrs=spattrs,
                                     soft_sigma=0. if soft_sigma is None else float(soft_sigma),
                                     soft_cutoff=float(soft_cutoff),
                                     soft_taps=str(soft_taps))
    for icat, p in enumerate(particles):
        ffi_module.set_index_value(icat, **p.index_value._to_c())

    has_weights = [bool(p.index_value('individual_weight')) for p in particles]
    n = [int(p.positions.shape[0]) for p in particles]

    counts_type = jax.ShapeDtypeStruct((nbins,), jnp.float64)
    buffer_type = jax.ShapeDtypeStruct((_vjp_bufsize(particles, mattrs, nbins),), jnp.float64)

    def _operands(ps):
        """Positions (and weight columns, when present) for each catalogue."""
        ops = []
        for p, hw in zip(ps, has_weights):
            ops.append(p.positions)
            if hw:
                ops.append(_values(p))
        return ops

    def _values(p):
        values = _stack_values(p.values, np=jnp)
        return jnp.zeros((p.positions.shape[0], 0), jnp.float64) if values is None else values

    @jax.custom_vjp
    def _counts(*ops):
        fwd = jax.ffi.ffi_call('count2_soft' + suffix, (counts_type, buffer_type))
        return fwd(*ops)[0]

    def _counts_fwd(*ops):
        return _counts(*ops), ops

    def _counts_bwd(ops, cotangent):
        # All gradients are FFI result buffers (XLA allocates them and zero-fills
        # nothing, so the kernel zeroes them itself before accumulating).
        outputs = (
            jax.ShapeDtypeStruct((n[0] * NDIM,), jnp.float64),
            jax.ShapeDtypeStruct((n[1] * NDIM,), jnp.float64),
            jax.ShapeDtypeStruct((n[0],), jnp.float64),
            jax.ShapeDtypeStruct((n[1],), jnp.float64),
            jax.ShapeDtypeStruct((1,), jnp.float64),
            buffer_type,
        )
        vjp = jax.ffi.ffi_call('count2_soft_vjp' + suffix, outputs)
        gpos1, gpos2, gw1, gw2, _gsigma, _buf = vjp(*ops, cotangent)

        grads = []
        for ip, (p, hw, gpos, gw) in enumerate(zip(particles, has_weights, (gpos1, gpos2), (gw1, gw2))):
            grads.append(gpos.reshape(p.positions.shape))
            if hw:
                grads.append(gw.reshape((-1, 1)))
        # For one Particles the operand list holds the same arrays twice, so the
        # four cotangents accumulate onto one primal: exactly dL/dp.
        return tuple(grads)

    _counts.defvjp(_counts_fwd, _counts_bwd)

    return {'weight': _counts(*_operands(particles)), 'soft': spec}
