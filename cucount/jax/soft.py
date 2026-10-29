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

A request whose binning is ``BinAttrs(s=edges, pole=(ells, los))`` additionally
softens the mu axis, which is what makes differentiable multipoles. Each pair
then deposits ``w_pair * Ws_i * M_l(mu)`` in slot ``(i, l)``, where ``M_l`` is the
truncated softmax over a mu quadrature grid projected onto the Legendre
polynomials of the grid centres (see :func:`cucount.numpy.soft.soft_mu_weights`). ``M_0 == 1``
exactly, so the monopole a soft-mu request returns is the soft-s count, and
``sigma_mu -> 0`` lands on the hard multipoles up to the mu quadrature. mu is
still a *coordinate* (it enters the kernel as the pair's own mu), so the adjoint
carries dM_l/dmu; the mu grid itself is not differentiated.

What is new here is differentiability. The counts are wrapped in
``jax.custom_vjp`` whose backward calls the CUDA adjoint kernel
(``count2_soft_vjp``, the same VJP pyfcfc exposes as ``soft_hist_vjp``): the
windowed-softmax forward is differentiated in closed form, so one backward pass
costs about one forward pass instead of carrying per-pair taps through autodiff.
Gradients are exact w.r.t. positions and individual weights. The kernel also
returns a ``sigma`` cotangent, but ``soft_sigma`` is a static argument here, so
it is discarded.

Multi-GPU works the way the hard ``count2`` does it: inside
``with create_sharding_mesh() as mesh:`` catalogue 1 is split over the mesh axis
and catalogue 2 is replicated, each shard counts its own slice against the whole
of catalogue 2, and the partial histograms are summed with ``lax.psum``. The
adjoint follows the same split: catalogue 1's gradient stays per-shard and the
replicated catalogue's partial gradients are summed with ``psum``. The custom VJP
sits *outside* the ``shard_map`` (differentiating a globally-shaped input) rather
than inside it (differentiating a shard-local one), which is what keeps jax's
manual-axis-type bookkeeping happy: a ``custom_vjp`` inside ``shard_map`` must
return gradients whose avals carry the same ``{V:axis}`` annotations as the
shard-local primals, and FFI result buffers do not.
"""

import numpy as np
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from . import Particles, WeightAttrs, MeshAttrs, SelectionAttrs, SplitAttrs, BinAttrs
from . import _resolve_ffi_backend, _stack_values, default_sharding_mesh
from ..numpy.soft import default_sigma, is_uniform, mu_spec

NDIM = 3


MAX_POLE = 8  # the kernel's ell cap; CUCOUNT_SOFT_MAXELL is MAX_POLE + 2


def _soft_axes(battrs):
    """Resolve the softened s axis, and the pole axis when there is one.

    Returns ``(edges, ells, los)``; ``ells`` and ``los`` are ``None`` for a plain
    isotropic request. A pole axis is served as a soft mu quadrature, so it needs
    a line of sight and a strictly increasing, unique ell list within the kernel's
    range -- the same restrictions ``make_soft_bin_spec`` enforces.
    """
    names = list(battrs.varnames)
    if names not in (['s'], ['s', 'pole']):
        raise NotImplementedError(
            'count2_soft softens the s axis and, optionally, a pole axis after it, so '
            "battrs must be BinAttrs(s=edges) or BinAttrs(s=edges, pole=(ells, los)); "
            f'got axes {names}')
    intervals = np.asarray(battrs.edges('s'), dtype=np.float64)
    edges = np.concatenate([intervals[:, 0], intervals[-1:, 1]])
    if edges.size < 2:
        raise ValueError('need at least two separation edges')
    ds = np.diff(edges)
    if not np.all(ds > 0):
        raise ValueError('separation edges must be strictly increasing')
    # Both uniform and arbitrary monotone edges are served (the kernel switches
    # from the recurrence to the general path), so there is nothing to reject.
    if names == ['s']:
        return edges, None, None
    ells = np.asarray(battrs.coords('pole'), dtype=np.int64)
    if ells.size < 1 or ells.size > MAX_POLE + 1:
        raise ValueError(f'1 to {MAX_POLE + 1} multipoles are served, got {ells.size}')
    if ells.min() < 0 or ells.max() > MAX_POLE:
        raise ValueError(f'multipoles must lie in [0, {MAX_POLE}], got {ells.tolist()}')
    if np.unique(ells).size != ells.size or np.any(np.diff(ells) <= 0):
        raise ValueError(f'multipoles must be unique and increasing, got {ells.tolist()}')
    # losnames is aligned with the axes, and the s axis has none, so read the
    # name off the pole axis itself.
    los = str(battrs.losnames[names.index('pole')])
    if not los:
        raise ValueError("the pole axis needs a line of sight, e.g. pole=(ells, 'z')")
    return edges, ells, los


def _soft_spec(edges, soft_sigma=None, soft_cutoff=1e-13, soft_taps='exact',
               soft_mu_sigma=None, soft_mu_bins=0, ells=None, los=None,
               soft_mu_edges=None):
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
    meta = dict(sigma=sigma, sigma_units='separation',
                sigma_kind='width' if width_scaled else 'absolute',
                uniform=uniform, fast_taps=soft_taps == 'fast',
                cutoff=float(soft_cutoff), kmax=kmax,
                taps=(2 * kmax + 1) if uniform else -1, nbins=nbins,
                multipoles=[] if ells is None else [int(e) for e in ells],
                nell=0 if ells is None else int(ells.size),
                slots=nbins * (1 if ells is None else int(ells.size)))
    if ells is not None:
        mspec = mu_spec(soft_mu_sigma, soft_mu_bins, soft_mu_edges, soft_cutoff)
        meta.update(mu_sigma=float(mspec['sigma']), mu_bins=int(mspec['nmu']),
                    mu_kmax=int(mspec['kmax']), los=los,
                    mu_uniform=bool(mspec['uniform']),
                    mu_sigma_kind='width' if mspec['sigma'] <= 0 else 'absolute',
                    mu_taps=(2 * int(mspec['kmax']) + 1) if mspec['uniform'] else -1)
        if not mspec['uniform']:
            meta['mu_edges'] = mspec['edges']

    return meta


def _bufsize(meshsize, n, nbins=0, mu_entries=0):
    """Scratch bytes one adjoint call asks XLA for, for ``n = [n1, n2]`` inputs.

    The adjoint allocates no histogram of its own (the cotangent is an input), but
    it keeps a slot -> input-index map per mesh so gradients reach the right
    particle after ``set_mesh`` reorders them, plus the reordered particle arrays.
    ``n`` must be the particle count *this* call sees: the global count when
    running on one device, the shard-local count under ``shard_map``.
    """
    return (2 * meshsize  # nparticles + cumnparticles, both meshes
            + sum(n) * (2 * (NDIM + 1) + 1)  # positions+spositions, values, index map
            + 2 * meshsize * 16  # launch bookkeeping and slack
            + 3 * nbins + 1  # centres, 1/(2 sigma_i^2) and the edges, non-uniform only
            + mu_entries  # same three tables for a non-arithmetic mu grid
            ) * 8 + 65536


def _vjp_bufsize(particles, mattrs, nbins=0, mu_entries=0):
    """Scratch bytes for the whole (unsharded) pair of catalogues."""
    meshsize = int(np.prod(np.asarray(mattrs.meshsize, dtype=np.int64)))
    return _bufsize(meshsize, [int(p.positions.shape[0]) for p in particles], nbins,
                    mu_entries)


@default_sharding_mesh
def count2_soft(*particles: Particles, battrs: BinAttrs, soft_sigma: float = None,
                soft_cutoff: float = 1e-13, soft_taps: str = 'exact',
                soft_mu_sigma: float = None, soft_mu_bins: int = 0,
                soft_mu_edges=None,
                wattrs: WeightAttrs = None,
                sattrs: SelectionAttrs = None, spattrs: SplitAttrs = None,
                mattrs: MeshAttrs = None, backend: str = None,
                sharding_mesh: jax.sharding.Mesh = None):
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
        ``BinAttrs(s=edges)`` with strictly increasing edges (uniform grids take
        the fast recurrence kernel, other grids the general path), or
        ``BinAttrs(s=edges, pole=(ells, los))`` for soft-mu multipoles.
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
    soft_mu_sigma : float, optional
        Width of the mu kernel, in mu units, when ``battrs`` carries a pole axis.
        ``None`` (or 0) means half a mu bin. The mu axis defaults to a uniform
        grid on [-1, 1] but can be given as ``soft_mu_edges``.
    soft_mu_bins : int, optional
        Resolution of that mu quadrature; 0 derives it (``ceil(4 / sigma_mu)``,
        or 40 for the default width). More bins resolve a sharper kernel.
    soft_mu_edges : array_like, optional
        The caller's own mu cell edges instead of the internal grid; strictly
        increasing and inside ``[-1, 1]``. mu is used unfolded (a negative mu
        stays negative), so odd multipoles keep their sign. Arithmetic edges
        still take the fast recurrence path; any other grid pays one ``exp`` per
        tap and gets per-cell widths (half the cell unless ``soft_mu_sigma`` is
        given).
    wattrs, sattrs, spattrs, mattrs, backend
        As in :func:`cucount.jax.count2`. Spin, angular upweighting,
        split/jackknife and a second binning axis are rejected: the soft kernel
        does not serve them.
    sharding_mesh : jax.sharding.Mesh, optional
        Usually taken from the active context, exactly as in
        :func:`cucount.jax.count2`: inside ``with create_sharding_mesh() as mesh:``
        the counts are computed in parallel over the mesh axis with catalogue 1
        split and catalogue 2 replicated, and ``jax.grad`` works through the same
        distribution.

    Returns
    -------
    dict
        ``{'weight': counts, 'soft': spec}``: fractional, differentiable counts
        shaped ``(nbins,)``, or ``(nbins, nell)`` when a pole axis was requested,
        plus the resolved soft parameters.
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

    edges, ells, los = _soft_axes(battrs)
    spec = _soft_spec(edges, soft_sigma, soft_cutoff, soft_taps,
                      soft_mu_sigma, soft_mu_bins, ells, los, soft_mu_edges)
    nbins = spec['nbins']
    slots = spec['slots']
    shape = (nbins,) if ells is None else (nbins, int(ells.size))
    # A non-arithmetic mu grid needs the three device tables (centres, per-cell
    # inverse variances, edges) inside the scratch buffer the FFI call is given.
    mu_entries = (0 if ells is None or spec.get('mu_uniform', True)
                  else 3 * int(spec['mu_bins']) + 1)

    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)

    # Attribute state is process-global in the FFI module (as it is for count2),
    # so the soft request is resolved once and read by both fwd and bwd handlers.
    ffi_module.set_count2_soft_attrs(mattrs._to_c(), battrs, wattrs=wattrs._to_c(),
                                     sattrs=sattrs, spattrs=spattrs,
                                     soft_sigma=0. if soft_sigma is None else float(soft_sigma),
                                     soft_cutoff=float(soft_cutoff),
                                     soft_taps=str(soft_taps),
                                     soft_mu_sigma=0. if soft_mu_sigma is None else float(soft_mu_sigma),
                                     soft_mu_bins=int(soft_mu_bins),
                                     soft_mu_edges=soft_mu_edges)
    for icat, p in enumerate(particles):
        ffi_module.set_index_value(icat, **p.index_value._to_c())

    has_weights = [bool(p.index_value('individual_weight')) for p in particles]
    meshsize = int(np.prod(np.asarray(mattrs.meshsize, dtype=np.int64)))

    def _values(p):
        values = _stack_values(p.values, np=jnp)
        return jnp.zeros((p.positions.shape[0], 0), jnp.float64) if values is None else values

    def _operands(ps):
        """Positions (and weight columns, when present) for each catalogue."""
        ops = []
        for p, hw in zip(ps, has_weights):
            ops.append(p.positions)
            if hw:
                ops.append(_values(p))
        return ops

    # Flat operand layout: [pos1, (w1), pos2, (w2)]. `owners` says which catalogue
    # each operand belongs to, and `starts` where that catalogue's positions sit,
    # so a traced call can read the particle count it actually sees (global on one
    # device, shard-local under shard_map).
    owners, starts = [], []
    for icat, hw in enumerate(has_weights):
        starts.append(len(owners))
        owners.append(icat)
        if hw:
            owners.append(icat)

    def _lens(ops):
        return [int(ops[starts[icat]].shape[0]) for icat in range(2)]

    def _local_forward(*ops):
        n = _lens(ops)
        # The kernel writes one flat slot per (s bin, multipole), so the FFI
        # buffer is 1-D and the public shape is restored outside the kernel.
        counts_type = jax.ShapeDtypeStruct((slots,), jnp.float64)
        buffer_type = jax.ShapeDtypeStruct(
            (_bufsize(meshsize, n, nbins, mu_entries),), jnp.float64)
        fwd = jax.ffi.ffi_call('count2_soft' + suffix, (counts_type, buffer_type))
        return fwd(*ops)[0].reshape(shape)

    def _local_bwd(ops, cotangent):
        n = _lens(ops)
        # All gradients are FFI result buffers (XLA allocates them and zero-fills
        # nothing, so the kernel zeroes them itself before accumulating).
        outputs = (
            jax.ShapeDtypeStruct((n[0] * NDIM,), jnp.float64),
            jax.ShapeDtypeStruct((n[1] * NDIM,), jnp.float64),
            jax.ShapeDtypeStruct((n[0],), jnp.float64),
            jax.ShapeDtypeStruct((n[1],), jnp.float64),
            jax.ShapeDtypeStruct((1,), jnp.float64),
            jax.ShapeDtypeStruct((_bufsize(meshsize, n, nbins, mu_entries),),
                                 jnp.float64),
        )
        vjp = jax.ffi.ffi_call('count2_soft_vjp' + suffix, outputs)
        gpos1, gpos2, gw1, gw2, _gsigma, _buf = vjp(*ops, cotangent.reshape((slots,)))
        grads = []
        for p, hw, gpos, gw in zip(particles, has_weights, (gpos1, gpos2), (gw1, gw2)):
            grads.append(gpos.reshape((-1, NDIM)))
            if hw:
                grads.append(gw.reshape((-1, 1)))
        # For one Particles the operand list holds the same arrays twice, so the
        # four cotangents accumulate onto one primal: exactly dL/dp.
        return tuple(grads)

    # Distribution, when a non-empty mesh axis is in scope: catalogue 1 is split
    # over the axis and catalogue 2 is replicated -- the pattern the hard count2
    # uses. Every shard counts the pairs of its own slice of catalogue 1 against the
    # whole of catalogue 2, and the partial histograms are summed with lax.psum.
    axes = tuple(sharding_mesh.axis_names) if sharding_mesh is not None else ()
    distribute = bool(axes)

    def _cat_spec(icat):
        return P(axes) if (distribute and icat == 0) else P(None)

    def _in_specs():
        specs = []
        for icat, hw in enumerate(has_weights):
            specs.append(_cat_spec(icat))
            if hw:
                specs.append(_cat_spec(icat))
        return tuple(specs)

    def _forward(*ops):
        if not distribute:
            return _local_forward(*ops)
        return shard_map(lambda *o: jax.lax.psum(_local_forward(*o), axes),
                         mesh=sharding_mesh, in_specs=_in_specs(), out_specs=P())(*ops)

    def _sharded_bwd(*ops_and_cot):
        ops, cotangent = ops_and_cot[:-1], ops_and_cot[-1]
        grads = _local_bwd(ops, cotangent)
        # The split catalogue's gradient is already per-shard; the replicated one
        # needs the per-shard partial gradients summed back together.
        return tuple(g if owners[i] == 0 else jax.lax.psum(g, axes)
                     for i, g in enumerate(grads))

    @jax.custom_vjp
    def _counts(*ops):
        return _forward(*ops)

    def _counts_fwd(*ops):
        return _counts(*ops), ops

    def _counts_bwd(ops, cotangent):
        if not distribute:
            return _local_bwd(ops, cotangent)
        return shard_map(_sharded_bwd, mesh=sharding_mesh,
                         in_specs=_in_specs() + (P(None),),
                         out_specs=_in_specs())(*ops, cotangent)

    _counts.defvjp(_counts_fwd, _counts_bwd)

    return {'weight': _counts(*_operands(particles)), 'soft': spec}
