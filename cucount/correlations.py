"""
Simplified correlation helpers built on top of cucount.numpy.count2.

Public API
----------
- calculate_correlation(particles_one, particles_two, spin_one, spin_two,
                       randoms_particles_one=None, randoms_particles_two=None,
                       battrs=None, theta_edges=None, min_theta=0.01, max_theta=1.0, nbins=20)

This computes the correlation for arbitrary spin combinations. It normalizes spin
statistics by the pair counts (spin=0) so the return values are per-pair
averages in each angular bin. For spin=(0,0), we use the Landy–Szalay formula.
"""

import numpy as np
from cucount.numpy import count2, BinAttrs


def calculate_correlation(particles_one, particles_two=None, spin_one=0, spin_two=0,
                          randoms_particles_one=None, randoms_particles_two=None,
                          battrs=None, theta_edges=None, min_theta=0.01, max_theta=1.0, nbins=20):
    """
    Calculate the correlation function between two particle catalogs with arbitrary spin.

    Parameters
    ----------
    particles_one : cucount.numpy.Particles
        The first particle catalog.
    particles_two : cucount.numpy.Particles, optional
        The second particle catalog. If None, computes auto-correlation.
    spin_one : int, optional
        Spin of the first catalog (0 for scalar, 2 for shear, etc.). Defaults to 0.
    spin_two : int, optional
        Spin of the second catalog (0 for scalar, 2 for shear, etc.). Defaults to 0.
    randoms_particles_one : cucount.numpy.Particles, optional
        Random catalog for first particles (needed for spin=(0,0) and spin=(0,s))
    randoms_particles_two : cucount.numpy.Particles, optional
        Random catalog for second particles (needed for spin=(0,0) cross-correlation)
    battrs : BinAttrs, optional
        Binning attributes object. If None, will be generated from theta_edges or defaults.
    theta_edges : array-like, optional
        Edges of the angular bins in degrees. If None, will be generated using
        min_theta, max_theta, and nbins.
    min_theta : float, optional
        Minimum theta value in degrees if theta_edges is None. Defaults to 0.01.
    max_theta : float, optional
        Maximum theta value in degrees if theta_edges is None. Defaults to 1.0.
    nbins : int, optional
        Number of bins if theta_edges is None. Defaults to 20.

    Returns
    -------
    corr : array or tuple of arrays
        Correlation function(s). Format depends on spins:
        - spin=(0,0): single array (ξ)
        - spin=(0,s) or (s,0): tuple (ξ_plus, ξ_cross)
        - spin=(s1,s2): tuple (ξ_++, ξ_×+, ξ_××)

    Examples
    --------
    >>> import numpy as np
    >>> from cucount.numpy import Particles, BinAttrs
    >>> from cucount.correlations import calculate_correlation

    >>> # Galaxy-galaxy clustering
    >>> xi = calculate_correlation(
    ...     lenses, lenses, spin_one=0, spin_two=0,
    ...     randoms_particles_one=randoms, randoms_particles_two=randoms
    ... )

    >>> # Galaxy-shear correlation
    >>> xi_plus, xi_cross = calculate_correlation(
    ...     lenses, sources, spin_one=0, spin_two=2,
    ...     randoms_particles_one=randoms
    ... )

    >>> # Cosmic shear
    >>> xi_pp, xi_xp, xi_xx = calculate_correlation(
    ...     sources, sources, spin_one=2, spin_two=2
    ... )
    """

    # Input validation
    if (spin_one != 0 or spin_two != 0) and particles_two is None:
        raise ValueError("particles_two is required when spin_one != 0 or spin_two != 0")

    # Create BinAttrs if not provided
    if battrs is None:
        if theta_edges is not None:
            # Convert degrees to radians
            theta_edges_rad = np.asarray(theta_edges) * (np.pi / 180.0)
        else:
            # Generate log-spaced bins
            theta_edges_rad = np.logspace(
                np.log10(min_theta * np.pi / 180.0),
                np.log10(max_theta * np.pi / 180.0),
                nbins + 1
            )
        battrs = BinAttrs(theta=theta_edges_rad)

    # Auto-correlation handling
    if particles_two is None:
        particles_two = particles_one
        is_auto = True
    else:
        is_auto = False

    # Compute correlations based on spin combination
    if spin_one == 0 and spin_two == 0:
        # Count-count correlation
        return _compute_count_count_correlation(
            particles_one, particles_two, is_auto,
            randoms_particles_one, randoms_particles_two,
            battrs
        )

    elif spin_one == 0 and spin_two != 0:
        # Scalar-spin correlation (galaxy-shear)
        return _compute_scalar_spin_correlation(
            particles_one, particles_two,
            randoms_particles_one,
            spin_two, battrs
        )

    elif spin_one != 0 and spin_two == 0:
        # Spin-scalar correlation (reversed galaxy-shear)
        return _compute_scalar_spin_correlation(
            particles_two, particles_one,
            randoms_particles_two,
            spin_one, battrs
        )

    elif spin_one != 0 and spin_two != 0:
        # Spin-spin correlation (shear-shear)
        return _compute_spin_spin_correlation(
            particles_one, particles_two,
            spin_one, spin_two, battrs
        )

    else:
        raise ValueError(f"Invalid spin combination: spin_one={spin_one}, spin_two={spin_two}")


def _compute_count_count_correlation(particles_one, particles_two, is_auto,
                                     randoms_one, randoms_two, battrs):
    """
    Compute count-count correlation using Landy-Szalay estimator.

    Auto-correlation: ξ = (DD - 2DR + RR) / RR
    Cross-correlation: ξ = (D1D2 - D1R2 - D2R1 + R1R2) / R1R2
    """
    if randoms_one is None:
        raise ValueError("Random catalog is required for count-count correlation")

    if is_auto:
        # Auto-correlation: Landy-Szalay
        DD = count2(particles_one, particles_one, battrs=battrs, spin1=0, spin2=0)
        DR = count2(particles_one, randoms_one, battrs=battrs, spin1=0, spin2=0)
        RR = count2(randoms_one, randoms_one, battrs=battrs, spin1=0, spin2=0)

        # Normalization factors (account for pairs that can't pair with themselves)
        DD_norm = np.sum(particles_one.weights)**2 - np.sum(particles_one.weights**2)
        DR_norm = np.sum(particles_one.weights) * np.sum(randoms_one.weights)
        RR_norm = np.sum(randoms_one.weights)**2 - np.sum(randoms_one.weights**2)

        # Landy-Szalay estimator
        numerator = (DD / DD_norm) - 2 * (DR / DR_norm) + (RR / RR_norm)
        denominator = RR / RR_norm

    else:
        # Cross-correlation
        if randoms_two is None:
            raise ValueError("randoms_particles_two is required for cross-correlation")

        D1D2 = count2(particles_one, particles_two, battrs=battrs, spin1=0, spin2=0)
        D1R2 = count2(particles_one, randoms_two, battrs=battrs, spin1=0, spin2=0)
        D2R1 = count2(particles_two, randoms_one, battrs=battrs, spin1=0, spin2=0)
        R1R2 = count2(randoms_one, randoms_two, battrs=battrs, spin1=0, spin2=0)

        # Normalization factors
        D1D2_norm = np.sum(particles_one.weights) * np.sum(particles_two.weights)
        D1R2_norm = np.sum(particles_one.weights) * np.sum(randoms_two.weights)
        D2R1_norm = np.sum(particles_two.weights) * np.sum(randoms_one.weights)
        R1R2_norm = np.sum(randoms_one.weights) * np.sum(randoms_two.weights)

        numerator = (D1D2 / D1D2_norm) - (D1R2 / D1R2_norm) - (D2R1 / D2R1_norm) + (R1R2 / R1R2_norm)
        denominator = R1R2 / R1R2_norm

    return numerator / denominator


def _compute_scalar_spin_correlation(scalar_particles, spin_particles,
                                     randoms, spin, battrs):
    """
    Compute scalar-spin correlation (galaxy-shear).

    Returns: (ξ_plus, ξ_cross)
    """
    if randoms is None:
        # Without randoms, just compute raw spin correlation
        PS_result = count2(scalar_particles, spin_particles, battrs=battrs, spin1=0, spin2=spin)
        PS_count = count2(scalar_particles, spin_particles, battrs=battrs, spin1=0, spin2=0)

        xi_plus = PS_result["plus"] / PS_count
        xi_cross = PS_result["cross"] / PS_count
    else:
        # With randoms: subtract random-source correlation
        PS_result = count2(scalar_particles, spin_particles, battrs=battrs, spin1=0, spin2=spin)
        PS_count = count2(scalar_particles, spin_particles, battrs=battrs, spin1=0, spin2=0)

        RS_result = count2(randoms, spin_particles, battrs=battrs, spin1=0, spin2=spin)
        RS_count = count2(randoms, spin_particles, battrs=battrs, spin1=0, spin2=0)

        xi_plus = (PS_result["plus"] / PS_count) - (RS_result["plus"] / RS_count)
        xi_cross = (PS_result["cross"] / PS_count) - (RS_result["cross"] / RS_count)

    return xi_plus, xi_cross


def _compute_spin_spin_correlation(particles_one, particles_two, spin_one, spin_two, battrs):
    """
    Compute spin-spin correlation (shear-shear).

    Returns: (ξ_++, ξ_×+, ξ_××)
    """
    SS_result = count2(particles_one, particles_two, battrs=battrs, spin1=spin_one, spin2=spin_two)
    SS_count = count2(particles_one, particles_two, battrs=battrs, spin1=0, spin2=0)

    xi_plus_plus = SS_result["plus_plus"] / SS_count
    xi_cross_plus = SS_result["cross_plus"] / SS_count
    xi_cross_cross = SS_result["cross_cross"] / SS_count

    return xi_plus_plus, xi_cross_plus, xi_cross_cross
