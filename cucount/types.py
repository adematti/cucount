import operator
import functools
from functools import partial
import numpy as np

from cucount.numpy import Particles, BinAttrs, WeightAttrs, MeshAttrs


prod = partial(functools.reduce, operator.mul)


def _use_jax(particles):
    try:
        from cucount.jax import Particles
        use_jax = isinstance(particles, Particles)
    except:
        use_jax = False
    return use_jax


def count2(*particles: Particles, battrs: BinAttrs=None, wattrs: WeightAttrs=None, **kwargs):
    """
    Perform two-point pair counts using the native cucount library, exporting to :mod:`lsstypes` format.

    Parameters
    ----------
    *particles : Particles
        Exactly two Particles instances to correlate (positions, optional weights/spin/bitwise).
        If from :mod:`cucount.jax`, use corresponding :func:`cucount.jax.count2`.
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
    **kwargs : int, optional
        Optional arguments for count2.

    Returns
    -------
    result : dict of Count2 or Count2Jackknife
        Two-point counts in :mod:`lsstypes` format.
    """
    import lsstypes as types

    def count2_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return types.Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

    if wattrs is None: wattrs = WeightAttrs()
    autocorr = len(particles) == 1
    use_jax = _use_jax(particles[0])
    if use_jax:
        from cucount.jax import count2
    else:
        from cucount.numpy import count2
    raw_counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, **kwargs)
    auto_weights = wattrs(*(particles[:1] * 2))
    cross_weights = [wattrs(particle) for particle in particles]
    cross_weights += [cross_weights[-1]] * (2 - len(cross_weights))

    # Preparation to remove self pairs (in a jax and numpy-friendly way)
    zero_masks = tuple((0 >= edges[:, 0]) & (0 < edges[:, 1]) for edges in battrs.edges().values())
    zero = np.zeros(tuple(mask.size for mask in zero_masks))
    zero[np.ix_(*zero_masks)] = 1.

    result = {}
    for key, counts in raw_counts.items():
        if particles[0].index_value('split'):  # With jackknife
            spattrs = kwargs['spattrs']
            ii_counts, ij_counts, ji_counts = {}, {}, {}
            for isplit in range(spattrs.nsplits):
                masks_i = [particle.get('split')[0] == isplit for particle in particles]
                masks_i += [masks_i[-1]] * (2 - len(masks_i))
                # ii counts
                _counts = counts[isplit]
                _cross_weights = [cross_weights[i] * masks_i[i] for i in range(len(cross_weights))]
                wsum = [w.sum() for w in _cross_weights]
                norm = prod(wsum)
                if autocorr:
                    auto_sum = (auto_weights * masks_i[0]).sum()
                    norm = norm - auto_sum
                    # Correct auto-pairs
                    _counts = _counts - auto_sum * zero
                ii_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=wsum))
                # ij counts
                _counts = counts[spattrs.nsplits + isplit]
                _cross_weights = (cross_weights[0] * masks_i[0], cross_weights[1] * (~masks_i[1]))
                wsum = [w.sum() for w in _cross_weights]
                norm = prod(wsum)
                ij_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=wsum))
                # ji counts
                _counts = counts[spattrs.nsplits * 2 + isplit]
                _cross_weights = (cross_weights[0] * (~masks_i[0]), cross_weights[1] * masks_i[1])
                wsum = [w.sum() for w in _cross_weights]
                norm = prod(wsum)
                ji_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=wsum))
            result[key] = types.Count2Jackknife(ii_counts, ij_counts, ji_counts)

        else:
            wsum = [w.sum() for w in cross_weights]
            norm = prod(wsum)
            if autocorr:
                auto_sum = auto_weights.sum()
                norm = norm - auto_sum
                # Correct auto-pairs
                counts = counts - auto_sum * zero
            result[key] = count2_to_lsstypes(counts=counts, norm=norm, attrs=dict(wsum=wsum))

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
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return types.Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

    return count2_to_lsstypes(count2_analytic(battrs=battrs, mattrs=mattrs), norm=1., attrs={})