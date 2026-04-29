import operator
import functools
from functools import partial

import numpy as np
from scipy import special

from cucount.numpy import Particles, BinAttrs, WeightAttrs, MeshAttrs, poles_to_ells


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


def count2(*particles: Particles, battrs: BinAttrs=None, wattrs: WeightAttrs=None, **kwargs):
    """
    Perform two-point pair counts using the native cucount library, exporting to :mod:`lsstypes` format.

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
        Two-point counts in :mod:`lsstypes` format.
    """
    import lsstypes as types

    def count2_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords, edges, ells = _get_coords_edges(battrs)
        norm = norm * np.ones_like(counts)
        kw = dict(**coords, **edges, coords=list(coords), attrs=attrs)
        if ells:
            poles = [types.Count2Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
            return types.Count2Poles(poles)
        return types.Count2(counts=counts, norm=norm, **kw)

    if wattrs is None: wattrs = WeightAttrs()
    autocorr = len(particles) == 1
    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count2
    else:
        from cucount.numpy import count2
    raw_counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, **kwargs)
    weights2 = wattrs(*(particles[:1] * 2))
    weights1 = [wattrs(particle) for particle in particles]
    weights1 += [weights1[-1]] * (2 - len(weights1))

    # Preparation to remove self pairs (in a jax and numpy-friendly way)
    coord_names, zero_masks = [], []
    for name, edges in battrs.edges():
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

    result = {}
    for key, counts in raw_counts.items():
        if with_jackknife:  # With jackknife
            spattrs = kwargs['spattrs']
            ii_counts, ij_counts, ji_counts = {}, {}, {}
            for isplit in range(spattrs.nsplits):
                masks_i = [particle.get('split')[0] == isplit for particle in particles]
                masks_i += [masks_i[-1]] * (2 - len(masks_i))
                # ii counts
                _weights1 = [weights1[i] * masks_i[i] for i in range(len(weights1))]
                sum_weights1 = [w.sum() for w in _weights1]
                norm = prod(sum_weights1)
                _counts = counts[isplit]
                if autocorr:
                    sum_weights2 = (weights2 * masks_i[0]).sum()
                    norm = norm - sum_weights2
                    # Correct auto-pairs
                    _counts = _counts - sum_weights2 * zero
                ii_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))
                # ij counts
                _counts = counts[spattrs.nsplits + isplit]
                _weights1 = (weights1[0] * masks_i[0], weights1[1] * (~masks_i[1]))
                sum_weights1 = [w.sum() for w in _weights1]
                norm = prod(sum_weights1)
                ij_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))
                # ji counts
                _counts = counts[spattrs.nsplits * 2 + isplit]
                _weights1 = (weights1[0] * (~masks_i[0]), weights1[1] * masks_i[1])
                sum_weights1 = [w.sum() for w in _weights1]
                norm = prod(sum_weights1)
                ji_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=sum_weights1))
            result[key] = types.Count2Jackknife(ii_counts, ij_counts, ji_counts)

        else:
            sum_weights1 = [w.sum() for w in weights1]
            norm = prod(sum_weights1)
            if autocorr:
                sum_weights2 = weights2.sum()
                norm = norm - sum_weights2
                # Correct auto-pairs
                counts = counts - sum_weights2 * zero
            result[key] = count2_to_lsstypes(counts=counts, norm=norm, attrs=dict(wsum=sum_weights1))

    return result


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
    import lsstypes as types
    from cucount.numpy import count2_analytic

    def count2_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords, edges, ells = _get_coords_edges(battrs)
        norm = norm * np.ones_like(counts)
        kw = dict(**coords, **edges, coords=list(coords), attrs=attrs)
        if ells:
            poles = [types.Count2Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
            return types.Count2Poles(poles)
        return types.Count2(counts=counts, norm=norm, **kw)

    return count2_to_lsstypes(count2_analytic(battrs=battrs, mattrs=mattrs), norm=1., attrs={})



def count3close(*particles: Particles,
                battrs12: BinAttrs,
                battrs23: BinAttrs,
                battrs13: BinAttrs=None,
                wattrs: WeightAttrs = None,
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
    import lsstypes as types

    def count3_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords12, edges12, ells12 = _get_coords_edges(battrs12, suffix='1')
        coords13, edges13, ells13 = _get_coords_edges(battrs12, suffix='2')
        coords = coords12 | coords13
        edges = edges12 | edges13
        if battrs23 is not None:
            coords23, edges23, ells23 = _get_coords_edges(battrs23, suffix='3')
            coords = coords | coords23
            edges = edges | edges23
        names = list(coords)
        norm = norm * np.ones_like(counts)
        kw = dict(**coords, **edges, coords=names, attrs=attrs)
        if ells12 and ells13:
            ells = poles_to_ells(ells12, ells13)
            poles = [types.Count3Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
            return types.Count3Poles(poles)
        return types.Count3(counts=counts, norm=norm, **kw)

    if wattrs is None:
        wattrs = WeightAttrs()

    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count3close
    else:
        from cucount.numpy import count3close

    raw_counts = count3close(
        *(particles + (particles[-1],) * (3 - len(particles))),
        battrs12=battrs12,
        battrs23=battrs23,
        battrs13=battrs13,
        wattrs=wattrs,
        **kwargs,
    )

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

    result = {}
    for key, counts in raw_counts.items():
        result[key] = count3_to_lsstypes(counts=counts, norm=norm, attrs=dict(wsum=sum_weights1))

    return result


def count3(*particles: Particles,
           battrs12: BinAttrs,
           battrs13: BinAttrs,
           wattrs: WeightAttrs = None,
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
    import lsstypes as types

    def count3_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords12, edges12, ells12 = _get_coords_edges(battrs12, suffix='1')
        coords13, edges13, ells13 = _get_coords_edges(battrs12, suffix='2')
        coords = coords12 | coords13
        edges = edges12 | edges13
        names = list(coords)
        norm = norm * np.ones_like(counts)
        kw = dict(**coords, **edges, coords=names, attrs=attrs)
        if ells12 and ells13:
            ells = poles_to_ells(ells12, ells13)
            poles = [types.Count3Pole(counts=counts[..., ill], norm=norm[..., ill], **kw, ell=ell) for ill, ell in enumerate(ells)]
            return types.Count3Poles(poles)
        return types.Count3(counts=counts, norm=norm, **kw)

    if wattrs is None:
        wattrs = WeightAttrs()

    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count3
    else:
        from cucount.numpy import count3

    raw_counts = count3(
        *(particles + (particles[-1],) * (3 - len(particles))),
        battrs12=battrs12,
        battrs13=battrs13,
        wattrs=wattrs,
        **kwargs,
    )

    weights1 = [wattrs(particle) for particle in particles]
    weights1 += [weights1[-1]] * (3 - len(weights1))
    sum_weights1 = [w.sum() for w in weights1]

    weights2 = [wattrs(particle, particle) for particle in particles]
    sum_weights2 = [w.sum() for w in weights2]

    if len(particles) == 1:
        weights3 = wattrs(*(particles[:1] * 3))
        sum_weights3 = weights3.sum()
        norm = (sum_weights1[0] ** 3 - 3 * sum_weights1[0] * sum_weights2[0] + 2 * sum_weights3)

    elif len(particles) == 2:
        # padding gives (0, 1, 1)
        norm = sum_weights1[0] * (sum_weights1[1] ** 2 - sum_weights2[1])
    else:
        norm = prod(sum_weights1)

    result = {}
    for key, counts in raw_counts.items():
        result[key] = count3_to_lsstypes(
            counts=counts,
            norm=norm,
            attrs=dict(wsum=sum_weights1),
        )

    return result