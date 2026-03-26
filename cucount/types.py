import operator
import functools
from functools import partial
import numpy as np

from cucount.numpy import Particles, BinAttrs, WeightAttrs


prod = partial(functools.reduce, operator.mul)


def _use_jax(particles):
    try:
        from cucount.jax import Particles
        use_jax = isinstance(particles, Particles)
    except:
        use_jax = False
    return use_jax


def count2(*particles: Particles, battrs: BinAttrs=None, wattrs: WeightAttrs=None, **kwargs):
    import lsstypes as types

    def count2_to_lsstypes(counts: np.ndarray, norm: np.ndarray, attrs: dict):
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return types.Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

    if wattrs is None: wattrs = WeightAttrs()
    autocorr = len(particles) == 1
    use_jax = _use_jax(particles)
    if use_jax:
        from cucount.jax import count2
    else:
        from cucount.numpy import count2
    counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, **kwargs)['weight']
    auto_weights = wattrs(*(particles[:1] * 2))
    cross_weights = [wattrs(particle) for particle in particles]
    zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())

    if particles[0].index_value.get('split'):  # With jackknife
        spattrs = kwargs['spattrs']
        ii_counts, ij_counts, ji_counts = {}, {}, {}
        for isplit in range(spattrs.nsplits):
            masks_i = [particle.get('split') == isplit for particle in particles]
            # ii counts
            _counts = counts[isplit]
            _cross_weights = [cross_weights[i] * masks_i[i] for i in range(len(cross_weights))]
            wsum = [w.sum() for w in _cross_weights]
            norm = prod(wsum)
            if autocorr:
                auto_sum = (auto_weights * masks_i[0]).sum()
                norm = norm - auto_sum
                # Correct auto-pairs
                _counts = _counts.at[zero_index].add(-auto_sum)
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
            ij_counts[isplit] = count2_to_lsstypes(counts=_counts, norm=norm, attrs=dict(wsum=wsum))
        return types.Count2Jackknife(ii_counts, ij_counts, ji_counts)

    else:

        wsum = [w.sum() for w in cross_weights]
        norm = prod(wsum)
        if autocorr:
            auto_sum = auto_weights.sum()
            norm = norm - auto_sum
            # Correct auto-pairs
            counts = counts.at[zero_index].add(-auto_sum)
        return count2_to_lsstypes(counts=counts, norm=norm, attrs=dict(wsum=wsum))