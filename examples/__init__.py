"""
Top-level cucount Python API.

This exposes a simplified helper to compute 2-point correlations with arbitrary spin
on top of the existing low-level count2 interface.

Example
-------
    import cucount
    from cucount.numpy import Particles
    from cucount.examples import calculate_correlation

    # Galaxy-galaxy clustering
    xi = calculate_correlation(
        lenses, lenses, spin_one=0, spin_two=0,
        randoms_particles_one=randoms, randoms_particles_two=randoms
    )

    # Galaxy-shear correlation
    xi_plus, xi_cross = calculate_correlation(
        lenses, sources, spin_one=0, spin_two=2,
        randoms_particles_one=randoms
    )

    # Cosmic shear
    xi_pp, xi_xp, xi_xx = cucount.calculate_correlation(
        sources, sources, spin_one=2, spin_two=2
    )
"""

from .correlations import calculate_correlation

__all__ = [
    "calculate_correlation"
]