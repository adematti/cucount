import operator
import functools
from functools import partial

import numpy as np
from scipy import special

from cucount.numpy import Particles, BinAttrs, WeightAttrs, MeshAttrs, symmetrize_poles


prod = partial(functools.reduce, operator.mul)


def _use_jax(particles):
    try:
        from cucount.jax import Particles
        use_jax = isinstance(particles, Particles)
    except:
        use_jax = False
    return use_jax


def _get_coords_edges(battrs, suffix=''):
    coord_names = list(battrs.coords())
    coord_names = [name for name in coord_names if name != 'pole']
    coords, edges = battrs.coords(coord_names), battrs.edges(coord_names)
    coords = {f'{name}{suffix}': coord for name, coord in zip(coord_names, coords)}
    edges = {f'{name}{suffix}_edges': edge for name, edge in zip(coord_names, edges)}
    from cucount.numpy import _get_ells
    ells = _get_ells(battrs)
    return coords, edges, ells


def _count2_to_lsstypes(counts: np.ndarray, norm: np.ndarray, battrs=None, attrs: dict=None):
    import lsstypes as types
    coords, edges, ells = _get_coords_edges(battrs)
    norm = norm * np.ones_like(counts)
    kw = dict(**coords, **edges, coords=list(coords), attrs=attrs or {})
    if ells:
        poles = [types.Count2Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
        return types.Count2Poles(poles)
    return types.Count2(counts=counts, norm=norm, **kw)


def compute_norm2(*particles: Particles, wattrs: WeightAttrs=None, masks=None, return_sum_weights=False):
    """Compute the 2-point normalization, including autocorr self-pair removal."""
    if wattrs is None: wattrs = WeightAttrs()

    autocorr = len(particles) == 1

    weights1 = [wattrs(particle) for particle in particles]
    weights1 += [weights1[-1]] * (2 - len(weights1))

    if masks is not None:
        masks = list(masks)
        masks += [masks[-1]] * (2 - len(masks))
        weights1 = [w * m for w, m in zip(weights1, masks)]

    sum_weights1 = [w.sum() for w in weights1]
    norm = prod(sum_weights1)

    if autocorr:
        weights2 = wattrs(*(particles[:1] * 2))
        if masks is not None:
            weights2 = weights2 * masks[0] * masks[1]
        norm = norm - weights2.sum()

    if return_sum_weights:
        return norm, sum_weights1
    return norm


def count2(*particles: Particles, battrs: BinAttrs=None, wattrs: WeightAttrs=None, norm=None, **kwargs):
    """
    Perform pair counts using the native cucount library, exporting to :mod:`lsstypes` format.

    Parameters
    ----------
    *particles : Particles
        Exactly two Particles instances to correlate (positions, optional weights1/spin/bitwise).
        If from :mod:`cucount.jax`, use corresponding :func:`cucount.jax.count2`.
    battrs : BinAttrs
        Binning specification (edges/shape) for the pair counts.
    wattrs : WeightAttrs, optional
        Weight attributes (spin, angular, bitwise). If None, a default WeightAttrs()
        (no weights1) is used.
    sattrs : SelectionAttrs, optional
        Selection attributes to restrict pairs. If None, defaults to SelectionAttrs().
    spattrs : SplitAttrs, optional
        Split attributes (for jackknife). If None, defaults to SplitAttrs().
    mattrs : MeshAttrs, optional
        Mesh attributes (periodic, cellsize). If None, defaults to MeshAttrs().
    **kwargs : int, optional
        Optional arguments for count2.

    Returns
    -------
    result : dict of Count2 or Count2Jackknife
        Pair counts in :mod:`lsstypes` format.
    """
    import lsstypes as types
    if wattrs is None: wattrs = WeightAttrs()
    autocorr = len(particles) == 1
    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count2
    else:
        from cucount.numpy import count2
    raw_counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, **kwargs)
    weights2 = wattrs(*(particles[:1] * 2))
    # input_norm, just in case norm happens to be 0
    input_norm = norm

    # Preparation to remove self pairs (in a jax and numpy-friendly way)
    coord_names, zero_masks = [], []
    for name, edges in battrs.edges().items():
        if name == 'pole':
            mask = np.array([(2 * ell + 1) * special.legendre(ell)(0.) for ell in edges])
            zero_masks.append(mask)
        elif name == 'k':
            raise NotImplementedError(f'{name} not supported')
        else:
            mask = (0 >= edges[:, 0]) & (0 < edges[:, 1])
            zero_masks.append(mask)
            coord_names.append(name)

    zero = prod(np.meshgrid(*zero_masks, indexing='ij'))

    with_jackknife = bool(particles[0].index_value('split'))

    def _to_lsstypes(counts, norm, attrs=None):
        if input_norm is not None:
            norm = input_norm
        return _count2_to_lsstypes(counts=counts, norm=norm, battrs=battrs, attrs=attrs)

    result = {}
    for key, counts in raw_counts.items():
        if with_jackknife:  # With jackknife
            spattrs = kwargs['spattrs']
            ii_counts, ij_counts, ji_counts = {}, {}, {}
            for isplit in range(spattrs.nsplits):
                masks_i = [particle.get('split')[0] == isplit for particle in particles]
                masks_i += [masks_i[-1]] * (2 - len(masks_i))

                # ii counts
                norm, sum_weights1 = compute_norm2(*particles, wattrs=wattrs, masks=masks_i, return_sum_weights=True)
                _counts = counts[isplit]
                if autocorr:
                    sum_weights2 = (weights2 * masks_i[0]).sum()
                    # Correct auto-pairs
                    _counts = _counts - sum_weights2 * zero
                ii_counts[isplit] = _to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))

                # ij counts
                _counts = counts[spattrs.nsplits + isplit]
                masks_ij = (masks_i[0], ~masks_i[1])
                norm, sum_weights1 = compute_norm2(*particles, wattrs=wattrs, masks=masks_ij, return_sum_weights=True)
                ij_counts[isplit] = _to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))

                # ji counts
                _counts = counts[spattrs.nsplits * 2 + isplit]
                masks_ji = (~masks_i[0], masks_i[1])
                norm, sum_weights1 = compute_norm2(*particles, wattrs=wattrs, masks=masks_ji, return_sum_weights=True)
                ji_counts[isplit] = _to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))

            result[key] = types.Count2Jackknife(ii_counts, ij_counts, ji_counts)

        else:
            norm, sum_weights1 = compute_norm2(*particles, wattrs=wattrs, return_sum_weights=True)
            if autocorr:
                sum_weights2 = weights2.sum()
                # Correct auto-pairs
                counts = counts - sum_weights2 * zero

            result[key] = _to_lsstypes(counts=counts, norm=norm, attrs=dict(wsum=sum_weights1))

    return result


def _soft_self_pair_mask(battrs: BinAttrs, soft_sigma=None, soft_cutoff=1e-13,
                         soft_mu_sigma=None, soft_mu_bins=0, soft_mu_edges=None):
    """The profile an autocorrelation self pair deposits, shape ``counts.shape``.

    The soft counts include the diagonal pairs exactly as the kernel sees them,
    so they must be removed with the same profile. A self pair sits at ``s = 0``
    (the kernel also sets ``mu = 0`` there), so it spreads over the separation
    bins with the ordinary Gaussian softmax and over the mu quadrature with the
    soft moment profile -- like any other pair, because the kernel does not
    special-case it. The hard wrapper subtracts ``sum_weights2`` from the bin
    that contains zero; the soft wrapper subtracts ``sum_weights2 * this``.
    """
    from cucount.numpy.soft import soft_mu_weights, soft_weights
    masks = []
    for name, edges in battrs.edges().items():
        if name == 'pole':
            ells = np.asarray(edges, dtype=np.int64)
            masks.append(soft_mu_weights(np.array([0.]), ells, soft_mu_sigma,
                                         soft_mu_bins, soft_cutoff, soft_mu_edges)[0])
        elif name == 's':
            # edges() gives one (low, high) pair per bin; the kernel's window is
            # [low[0], high[-1]), so a grid that excludes zero never counted the
            # self pairs in the first place.
            edges = np.asarray(edges, dtype=np.float64)
            inside = edges[0, 0] <= 0. < edges[-1, 1]
            masks.append(soft_weights(0., np.append(edges[:, 0], edges[-1, 1]), sigma=soft_sigma)[0]
                         if inside else np.zeros(edges.shape[0]))
        else:
            raise NotImplementedError(f'soft binning does not serve the {name!r} axis')
    return prod(np.meshgrid(*masks, indexing='ij'))


def count2_soft(*particles: Particles, battrs: BinAttrs=None, wattrs: WeightAttrs=None, norm=None,
                soft_sigma: float=None, soft_cutoff: float=1e-13, soft_taps: str='exact',
                soft_mu_sigma: float=None, soft_mu_bins: int=0, soft_mu_edges=None, **kwargs):
    """
    Perform soft pair counts, exporting to :mod:`lsstypes` format.

    Drop-in neighbour of :func:`count2` for the Gaussian-softmax soft binning:
    the counts are spread over the separation bins (and, with a ``pole`` axis,
    over the mu quadrature as soft multipoles) instead of being placed in a
    single bin. Normalization, self-pair removal and the ``Count2`` /
    ``Count2Poles`` packing are the same as in :func:`count2`.

    Because the self pairs of an autocorrelation carry soft weight too, they are
    removed with the soft profile (see :func:`_soft_self_pair_mask`) rather than
    from a single bin; the rest of the normalization is unchanged.

    Parameters
    ----------
    *particles : Particles
        One (autocorrelation) or two Particles instances, exactly as in
        :func:`count2`. If from :mod:`cucount.jax`, they dispatch to
        :func:`cucount.jax.count2_soft` (and :func:`cucount.jax.count2_soft_vjp`
        remains available for gradients).
    battrs : BinAttrs
        ``BinAttrs(s=edges)`` for soft isotropic counts, or
        ``BinAttrs(s=edges, pole=(ells, los))`` for soft multipoles. Any strictly
        increasing ``s`` edges are served; arithmetic grids take the fast
        recurrence kernel.
    wattrs : WeightAttrs, optional
        Weight attributes; a default :class:`WeightAttrs` (no weights) is used
        when ``None``.
    norm : array, optional
        Override the computed normalization.
    soft_sigma : float, optional
        Gaussian width in separation units; ``None`` means half a bin (half of
        *each* bin on a non-uniform grid).
    soft_cutoff : float, optional
        Relative weight below which the Gaussian tail is dropped (the window is
        renormalized, so weight stays conserved). Speed knob: taps grow as
        ``sqrt(2 ln(1 / cutoff))``. Default ``1e-13``.
    soft_taps : {'exact', 'fast'}
        Only affects a non-uniform ``s`` grid: ``'exact'`` (default) uses double
        ``exp``, ``'fast'`` uses ``__expf`` for ~1e-6 weight accuracy.
    soft_mu_sigma : float, optional
        Width of the mu kernel when ``battrs`` has a pole axis; ``None`` means
        half a mu cell.
    soft_mu_bins : int, optional
        Resolution of the internal uniform mu quadrature; 0 derives it.
    soft_mu_edges : array_like, optional
        Caller-supplied mu cell edges (strictly increasing, inside ``[-1, 1]``,
        mu is not folded) instead of the internal uniform grid.
    **kwargs
        Selection, split, mesh and backend arguments, forwarded to the count
        function. The soft kernel is CUDA-only for now and rejects
        spin/angular/split requests rather than ignoring them.

    Returns
    -------
    result : dict of Count2 or Count2Poles
        Pair counts in :mod:`lsstypes` format, with the soft configuration in
        ``attrs`` (``soft_sigma``, ``soft_cutoff``, ``soft_multipoles``, ...).
    """
    if wattrs is None: wattrs = WeightAttrs()
    autocorr = len(particles) == 1
    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count2_soft
    else:
        from cucount.numpy import count2_soft
    # The soft kernel rejects split/jackknife, so the Count2Jackknife branch of
    # count2 has no counterpart here; raise instead of silently dropping it.
    if getattr(kwargs.get('spattrs'), 'nsplits', 0):
        raise NotImplementedError('split/jackknife counts are not served by the soft kernel yet')

    raw_counts = count2_soft(*(particles * 2 if autocorr else particles),
                             battrs=battrs, wattrs=wattrs,
                             soft_sigma=soft_sigma, soft_cutoff=soft_cutoff,
                             soft_taps=soft_taps, soft_mu_sigma=soft_mu_sigma,
                             soft_mu_bins=soft_mu_bins, soft_mu_edges=soft_mu_edges,
                             **kwargs)
    meta = dict(raw_counts.get('soft') or {})

    norm_value, sum_weights1 = compute_norm2(*particles, wattrs=wattrs, return_sum_weights=True)
    if norm is not None:
        norm_value = norm

    attrs = dict(wsum=sum_weights1)
    # Keep the soft configuration with the counts: without it a saved
    # correlation cannot be told apart from a hard-binned one.
    for key in ('sigma', 'sigma_kind', 'cutoff', 'kmax', 'multipoles',
                'mu_sigma', 'mu_bins', 'mu_kmax', 'mu_uniform', 'mu_sigma_kind', 'los'):
        if key in meta:
            attrs[f'soft_{key}'] = meta[key]

    zero = None
    if autocorr:
        zero = _soft_self_pair_mask(battrs, soft_sigma=soft_sigma, soft_cutoff=soft_cutoff,
                                    soft_mu_sigma=soft_mu_sigma, soft_mu_bins=soft_mu_bins,
                                    soft_mu_edges=soft_mu_edges)
    result = {}
    for key, counts in raw_counts.items():
        if key == 'soft':
            continue
        if autocorr:
            counts = counts - wattrs(*(particles[:1] * 2)).sum() * zero
        result[key] = _count2_to_lsstypes(counts=counts, norm=norm_value, battrs=battrs, attrs=attrs)
    return result


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
    counts : Count2
        Pair counts in :mod:`lsstypes` format.
    """
    from cucount.numpy import count2_analytic

    return _count2_to_lsstypes(count2_analytic(battrs=battrs, mattrs=mattrs), norm=1., battrs=battrs, attrs={})



def _count3_to_lsstypes(counts: np.ndarray, norm: np.ndarray, battrs12=None, battrs13=None, battrs23=None, attrs: dict=None, symmetrize_poles=symmetrize_poles):
    import lsstypes as types
    coords12, edges12, ells12 = _get_coords_edges(battrs12, suffix='1')
    coords13, edges13, ells13 = _get_coords_edges(battrs13, suffix='2')
    coords = coords12 | coords13
    edges = edges12 | edges13
    if battrs23 is not None:
        coords23, edges23, ells23 = _get_coords_edges(battrs23, suffix='3')
        coords = coords | coords23
        edges = edges | edges23
    names = list(coords)
    kw = dict(**coords, **edges, coords=names, attrs=attrs)
    if ells12 and ells13:
        counts, ells = symmetrize_poles(counts, ells12, ells13)
        # broadcast the (scalar) norm only AFTER symmetrisation: symmetrize_poles replaces the pole axis
        norm = norm * np.ones_like(counts)
        poles = [types.Count3Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
        return types.Count3Poles(poles)
    norm = norm * np.ones_like(counts)
    return types.Count3(counts=counts, norm=norm, **kw)


def compute_norm3(*particles: Particles, wattrs: WeightAttrs=None, return_sum_weights=False):
    """Compute the 3-point normalization."""
    if wattrs is None: wattrs = WeightAttrs()

    weights1 = [wattrs(particle) for particle in particles]
    weights1 += [weights1[-1]] * (3 - len(weights1))
    sum_weights1 = [w.sum() for w in weights1]

    weights2 = [wattrs(particle, particle) for particle in particles]
    sum_weights2 = [w.sum() for w in weights2]

    if len(particles) == 1:
        weights3 = wattrs(*(particles[:1] * 3))
        sum_weights3 = weights3.sum()
        norm = sum_weights1[0]**3 - 3 * sum_weights1[0] * sum_weights2[0] + 2 * sum_weights3
    elif len(particles) == 2:
        # because padding gives (0, 1, 1)
        norm = sum_weights1[0] * (sum_weights1[1]**2 - sum_weights2[1])
    else:
        norm = prod(sum_weights1)

    if return_sum_weights:
        return norm, sum_weights1
    return norm


def count3close(*particles: Particles,
                battrs12: BinAttrs,
                battrs13: BinAttrs,
                battrs23: BinAttrs=None,
                wattrs: WeightAttrs=None,
                norm=None,
                **kwargs):
    """
    Perform close-triplet counts using the native cucount library, exporting to :mod:`lsstypes` format.

    Parameters
    ----------
    *particles : Particles
        Exactly three Particles instances to correlate.
        If from :mod:`cucount.jax`, use corresponding :func:`cucount.jax.count3close`.
    battrs12 : BinAttrs
        Binning specification for pair (1, 2).
    battrs13 : BinAttrs
        Binning specification for pair (1, 3).
    battrs23 : BinAttrs, optional
        Binning specification for pair (2, 3).
    wattrs : WeightAttrs, optional
        Weight attributes. If ``None``, a default :class:`WeightAttrs` is used.
    **kwargs
        Optional arguments forwarded to :func:`cucount.numpy.count3close` or :func:`cucount.jax.count3close`.

    Returns
    -------
    result : dict of Count3
        Three-point counts in :mod:`lsstypes` format.
    """
    input_norm = norm

    if wattrs is None:
        wattrs = WeightAttrs()

    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count3close, symmetrize_poles
    else:
        from cucount.numpy import count3close, symmetrize_poles

    raw_counts = count3close(*(particles + (particles[-1],) * (3 - len(particles))),
                            battrs12=battrs12, battrs23=battrs23, battrs13=battrs13,
                            wattrs=wattrs, **kwargs)

    norm, sum_weights1 = compute_norm3(*particles, wattrs=wattrs, return_sum_weights=True)
    if input_norm is not None:
        norm = input_norm

    result = {}
    for key, counts in raw_counts.items():
        result[key] = _count3_to_lsstypes(counts=counts, norm=norm, battrs12=battrs12, battrs13=battrs13, battrs23=battrs23, attrs=dict(wsum=sum_weights1), symmetrize_poles=symmetrize_poles)

    return result


def count3(*particles: Particles,
           battrs12: BinAttrs,
           battrs13: BinAttrs,
           wattrs: WeightAttrs=None,
           norm=None,
           **kwargs):
    """
    Perform factorized triplet counts using the native cucount library,
    exporting to :mod:`lsstypes` format.

    For each primary particle in catalog 1, particles in catalog 2 are
    binned as a function of the (1, 2) separation and particles in catalog 3
    are binned as a function of the (1, 3) separation. The accumulated
    contribution is

    .. math::

        w_1 \\, w_2(r_{12}) \\, w_3(r_{13})

    There is no binning or selection involving the (2, 3) separation.

    Parameters
    ----------
    *particles : Particles
        Exactly three ``Particles`` instances to correlate.
        If from :mod:`cucount.jax`, use corresponding
        :func:`cucount.jax.count3`.
    battrs12 : BinAttrs
        Binning specification for pair (1, 2).
    battrs13 : BinAttrs
        Binning specification for pair (1, 3).
    wattrs : WeightAttrs, optional
        Weight attributes. If ``None``, a default :class:`WeightAttrs`
        is used.
    **kwargs
        Optional arguments forwarded to :func:`cucount.numpy.count3`
        or :func:`cucount.jax.count3`.

    Returns
    -------
    result : dict of Count3
        Three-point counts in :mod:`lsstypes` format.
    """
    input_norm = norm

    if wattrs is None:
        wattrs = WeightAttrs()

    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count3, symmetrize_poles
    else:
        from cucount.numpy import count3, symmetrize_poles

    raw_counts = count3(
        *(particles + (particles[-1],) * (3 - len(particles))),
        battrs12=battrs12, battrs13=battrs13, wattrs=wattrs, **kwargs)

    norm, sum_weights1 = compute_norm3(*particles, wattrs=wattrs, return_sum_weights=True)
    if input_norm is not None:
        norm = input_norm

    result = {}
    for key, counts in raw_counts.items():
        result[key] = _count3_to_lsstypes(counts=counts, norm=norm, battrs12=battrs12, battrs13=battrs13, attrs=dict(wsum=sum_weights1), symmetrize_poles=symmetrize_poles)

    return result


def count3_analytic(battrs12: BinAttrs, battrs13: BinAttrs, mattrs: MeshAttrs=None):
    """
    Perform triplet counts analytically for periodic boxes.

    Parameters
    ----------
    battrs12 : BinAttrs
        Binning specification for pair (1, 2).
    battrs13 : BinAttrs
        Binning specification for pair (1, 3).
    mattrs : MeshAttrs or array, optional
        Mesh attributes (boxsize).

    Returns
    -------
    counts : Count3
        Three-point counts in :mod:`lsstypes` format.
    """
    from cucount.numpy import count3_analytic

    return _count3_to_lsstypes(count3_analytic(battrs12=battrs12, battrs13=battrs13, mattrs=mattrs), norm=1., battrs12=battrs12, battrs13=battrs13, attrs={})
