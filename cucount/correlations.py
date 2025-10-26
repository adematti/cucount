"""
Correlation helpers built on top of cucount.numpy.count2.

Public API
----------
- calculate_correlation(tracer_one, tracer_two, spin_tracer_one, spin_tracer_two,
                       battrs=None, theta_edges=None, min_theta=0.01, max_theta=1.0, nbins=20,
                       errors=None, n_jackknife_regions=16, seed=1234 , return_regions = False )

This computes the correlation for arbitrary spin combinations and, if requested,
jackknife errors using a lightweight region assignment (following the approach here:
 https://github.com/theonefromnowhere/JK_pycorr/blob/main/CF_JK_ST_conf.py). It normalizes spin
statistics by the pair counts (spin-i/0) so the return values are per-pair
averages in each angular bin. For spin=(0,0), the raw per-bin pair counts we use 
the Landy–Szalay formula.

"""

import numpy as np
from cucount.numpy import count2
from . import __jackknife__

def calculate_correlation(particles_one, particles_two, spin_one, spin_two, randoms_particles_one=None, randoms_particles_two=None,
                          battrs=None, theta_edges=None, min_theta=0.01, max_theta=1.0, nbins=20,
                          errors=None, n_jackknife_regions=16, seed=1234, return_regions=False):
    """
    Calculate the correlation function between two particle catalogs with arbitrary spin.

    Parameters
    ----------
    particles_one : cucount.numpy.Particles
        The first particle catalog.
    particles_two : cucount.numpy.Particles
        The second particle catalog.
    spin_one : int
        Spin of the first catalog (0 for scalar, 2 for shear, etc.).
    spin_two : int
        Spin of the second catalog (0 for scalar, 2 for shear, etc.).
    randoms_particles_one : cucount.numpy.Particles, optional
        Random catalog for first particles (needed for spin=(0,0) and spin=(0,s) with jackknife)
    randoms_particles_two : cucount.numpy.Particles, optional
        Random catalog for second particles (needed for spin=(0,0) with jackknife)
    battrs : BinAttrs, optional
        Binning attributes object. Defaults to None.
    theta_edges : array-like, optional
        Edges of the angular bins. If None, will be generated using min_theta, max_theta, and nbins. Defaults to None.
    min_theta : float, optional
        Minimum theta value for binning if theta_edges is None. Defaults to 0.01.
    max_theta : float, optional
        Maximum theta value for binning if theta_edges is None. Defaults to 1.0
    nbins : int, optional
        Number of bins if theta_edges is None. Defaults to 20.
    errors : str, optional
        Type of error estimation to use ('jackknife' or None). Defaults to None.
    n_jackknife_regions : int, optional
        Number of jackknife regions to use if errors is 'jackknife'. Defaults to 16.
    seed : int, optional
        Random seed for jackknife region assignment. Defaults to 1234.
    return_regions : bool, optional
        Whether to return the correlation values for each jackknife region. This is useful for calculating derived
        quantities. Defaults to False.

    Returns
    -------
    corr : array or tuple of arrays
        Correlation function(s). Format depends on spins:
        - spin=(0,0): single array
        - spin=(0,s): tuple (plus, cross)
        - spin=(s1,s2): tuple (plus_plus, cross_plus, cross_cross)
    err : array or tuple of arrays or None
        Errors (if errors='jackknife'), same format as corr. None if errors=None.
    regions : array or tuple of arrays, optional
        Per-region jackknife samples (only if return_regions=True)
    """

    # Input validation
    if (spin_one != 0 or spin_two != 0) and particles_two is None:
        raise ValueError("particles_two is required when spin_one != 0 or spin_two != 0")

    # Compute total number of bins for jackknife
    if battrs is not None:
        # Assume battrs defines the binning
        # For now, we'll compute n_bins from battrs.size attribute if available
        # This will be passed to jackknife functions
        if hasattr(battrs, 'size'):
            n_bins = battrs.size
        else:
            # Fallback: estimate from theta_edges or defaults
            if theta_edges is not None:
                n_bins = len(theta_edges) - 1
            else:
                n_bins = nbins

    # Just calculate the non-jackknife correlation
    if errors != 'jackknife':
        if spin_one == 0 and spin_two == 0:
            # Count-count correlation: use randoms if provided, otherwise simple counts
            if randoms_particles_one is not None:
                # Determine if auto or cross correlation
                is_cross = (particles_two is not None and randoms_particles_two is not None)

                if is_cross:
                    # Cross-correlation
                    corr = __compute_count_count_correlation__(
                        particles_one, particles_two,
                        randoms_particles_one, randoms_particles_two,
                        particles_one.weights, particles_two.weights,
                        randoms_particles_one.weights, randoms_particles_two.weights,
                        battrs,
                    )
                else:
                    # Auto-correlation
                    corr = __compute_count_count_correlation__(
                        particles_one, None,
                        randoms_particles_one, None,
                        particles_one.weights, None,
                        randoms_particles_one.weights, None,
                        battrs,
                    )
            else:
                # Simple pair counts without randoms
                corr = count2(particles_one, particles_two, battrs=battrs, spin1=0, spin2=0)
            return corr, None

        elif spin_one == 0 and spin_two != 0:
            # Scalar-spin correlation
            if randoms_particles_one is not None:
                xi_g_plus, xi_g_cross = __compute_count_spin_correlation__(
                    particles_one,
                    particles_two,
                    randoms_particles_one,
                    spin_two,
                    battrs,
                )
            else:
                # Without randoms, just compute raw spin correlation
                result = count2(particles_one, particles_two, battrs=battrs, spin1=0, spin2=spin_two)
                counts = count2(particles_one, particles_two, battrs=battrs, spin1=0, spin2=0)
                xi_g_plus = result["plus"] / counts
                xi_g_cross = result["cross"] / counts
            corr = (xi_g_plus, xi_g_cross)
            return corr, None

        elif spin_one != 0 and spin_two == 0:
            # Spin-scalar correlation (reverse order)
            if randoms_particles_two is not None:
                xi_g_plus, xi_g_cross = __compute_count_spin_correlation__(
                    particles_two,
                    particles_one,
                    randoms_particles_two,
                    spin_one,
                    battrs,
                )
            else:
                # Without randoms, just compute raw spin correlation
                result = count2(particles_two, particles_one, battrs=battrs, spin1=0, spin2=spin_one)
                counts = count2(particles_two, particles_one, battrs=battrs, spin1=0, spin2=0)
                xi_g_plus = result["plus"] / counts
                xi_g_cross = result["cross"] / counts
            corr = (xi_g_plus, xi_g_cross)
            return corr, None

        elif spin_one != 0 and spin_two != 0:
            # Spin-spin correlation
            xi_pp, xi_xp, xi_xx = __compute_spin_spin_correlation__(
                particles_one,
                spin_one,
                particles_two,
                spin_two,
                battrs,
            )
            corr = (xi_pp, xi_xp, xi_xx)
            return corr, None

        else:
            raise ValueError("Invalid spin combination.")

    elif errors == 'jackknife':
        # Generate jackknife region assignments based on sky position
        # All catalogs use the same spatial regions
        assignments_one = __jackknife__.assign_jackknife_regions(particles_one, n_jackknife_regions, seed)

        # If particles_two exists, it should use the same sky-based regions as particles_one
        # (same k-means clustering result since based on same sky positions)
        if particles_two is not None:
            assignments_two = __jackknife__.assign_jackknife_regions(particles_two, n_jackknife_regions, seed)
        else:
            assignments_two = None  # Not needed if particles_two doesn't exist

        # Route based on spin combination
        if spin_one == 0 and spin_two == 0:
            # Count-count correlation: need randoms
            if randoms_particles_one is None:
                raise ValueError("Jackknife for spin=(0,0) requires randoms_particles_one")

            # Determine auto vs cross correlation
            is_cross = (randoms_particles_two is not None)

            assignments_randoms_one = __jackknife__.assign_jackknife_regions(
                randoms_particles_one, n_jackknife_regions, seed
            )

            if is_cross:
                # Cross-correlation jackknife (fast with hole subtraction)
                assignments_randoms_two = __jackknife__.assign_jackknife_regions(
                    randoms_particles_two, n_jackknife_regions, seed
                )

                xi_full, jk_samples = __jackknife__.run_fast_cross_count_jackknife(
                    particles_one, particles_two,
                    randoms_particles_one, randoms_particles_two,
                    assignments_one, assignments_two,
                    assignments_randoms_one, assignments_randoms_two,
                    battrs, n_bins, n_jackknife_regions
                )
            else:
                # Auto-correlation jackknife (fast with hole subtraction)
                # Use particles_one for both, and same assignments
                xi_full, jk_samples = __jackknife__.run_fast_count_count_jackknife(
                    particles_one, particles_one,
                    randoms_particles_one, randoms_particles_one,
                    assignments_one, assignments_one,  # Same assignments for auto-correlation
                    assignments_randoms_one, assignments_randoms_one,  # Same assignments for auto-correlation
                    battrs, n_bins, n_jackknife_regions
                )

            # Compute covariance and errors
            cov = __jackknife__.compute_jackknife_covariance(jk_samples)
            err = __jackknife__.extract_errors(cov)

            if return_regions:
                return xi_full, err, jk_samples
            else:
                return xi_full, err

        elif spin_one == 0 and spin_two != 0:
            # Scalar-spin correlation: need randoms
            if randoms_particles_one is None:
                raise ValueError("Jackknife for spin=(0,s) requires randoms_particles_one")

            assignments_randoms = __jackknife__.assign_jackknife_regions(randoms_particles_one, n_jackknife_regions, seed)

            xi_plus_full, xi_cross_full, jk_plus, jk_cross = __jackknife__.run_fast_scalar_spin_jackknife(
                particles_one, particles_two, randoms_particles_one,
                assignments_one, assignments_two, assignments_randoms,
                spin_two, battrs, n_bins, n_jackknife_regions
            )

            # Compute covariance and errors for each component
            cov_plus = __jackknife__.compute_jackknife_covariance(jk_plus)
            cov_cross = __jackknife__.compute_jackknife_covariance(jk_cross)
            err_plus = __jackknife__.extract_errors(cov_plus)
            err_cross = __jackknife__.extract_errors(cov_cross)

            corr = (xi_plus_full, xi_cross_full)
            err = (err_plus, err_cross)

            if return_regions:
                regions = (jk_plus, jk_cross)
                return corr, err, regions
            else:
                return corr, err

        elif spin_one != 0 and spin_two == 0:
            # Spin-scalar correlation (reverse order): need randoms
            if randoms_particles_two is None:
                raise ValueError("Jackknife for spin=(s,0) requires randoms_particles_two")

            assignments_randoms = __jackknife__.assign_jackknife_regions(randoms_particles_two, n_jackknife_regions, seed)

            xi_plus_full, xi_cross_full, jk_plus, jk_cross = __jackknife__.run_fast_scalar_spin_jackknife(
                particles_two, particles_one, randoms_particles_two,
                assignments_two, assignments_one, assignments_randoms,
                spin_one, battrs, n_bins, n_jackknife_regions
            )

            # Compute covariance and errors for each component
            cov_plus = __jackknife__.compute_jackknife_covariance(jk_plus)
            cov_cross = __jackknife__.compute_jackknife_covariance(jk_cross)
            err_plus = __jackknife__.extract_errors(cov_plus)
            err_cross = __jackknife__.extract_errors(cov_cross)

            corr = (xi_plus_full, xi_cross_full)
            err = (err_plus, err_cross)

            if return_regions:
                regions = (jk_plus, jk_cross)
                return corr, err, regions
            else:
                return corr, err

        elif spin_one != 0 and spin_two != 0:
            # Spin-spin correlation: no randoms needed
            xi_pp_full, xi_xp_full, xi_xx_full, jk_pp, jk_xp, jk_xx = __jackknife__.run_fast_spin_spin_jackknife(
                particles_one, assignments_one,
                spin_one, spin_two,
                battrs, n_bins, n_jackknife_regions
            )

            # Compute covariance and errors for each component
            cov_pp = __jackknife__.compute_jackknife_covariance(jk_pp)
            cov_xp = __jackknife__.compute_jackknife_covariance(jk_xp)
            cov_xx = __jackknife__.compute_jackknife_covariance(jk_xx)
            err_pp = __jackknife__.extract_errors(cov_pp)
            err_xp = __jackknife__.extract_errors(cov_xp)
            err_xx = __jackknife__.extract_errors(cov_xx)

            corr = (xi_pp_full, xi_xp_full, xi_xx_full)
            err = (err_pp, err_xp, err_xx)

            if return_regions:
                regions = (jk_pp, jk_xp, jk_xx)
                return corr, err, regions
            else:
                return corr, err
        else:
            raise ValueError("Invalid spin combination.")

def __compute_count_count_correlation__(
    pos_particles_one,
    pos_particles_two,   # None for auto-correlation
    ran_particles_one,
    ran_particles_two,   # None for auto-correlation
    pos_weights_one: np.ndarray,
    pos_weights_two,     # None for auto-correlation
    ran_weights_one: np.ndarray,
    ran_weights_two,     # None for auto-correlation
    battrs,
) -> np.ndarray:
    """
    Compute count-count correlation.

    Auto-correlation (if pos_particles_two or ran_particles_two is None):
        (DD - 2DR + RR) / RR  (Landy-Szalay)

    Cross-correlation (if both provided and different):
        (D1D2 - D1R2 - D2R1 + R1R2) / R1R2
    """
    # Detect auto vs cross correlation
    is_auto = (pos_particles_two is None or ran_particles_two is None)

    if is_auto:
        # Auto-correlation: Landy-Szalay (saves computing D1R2 vs D2R1)
        DD = count2(pos_particles_one, pos_particles_one, battrs=battrs, spin1=0, spin2=0)
        DR = count2(pos_particles_one, ran_particles_one, battrs=battrs, spin1=0, spin2=0)
        RR = count2(ran_particles_one, ran_particles_one, battrs=battrs, spin1=0, spin2=0)

        DD_norm = np.sum(pos_weights_one) ** 2 - np.sum(pos_weights_one**2)
        DR_norm = np.sum(pos_weights_one) * np.sum(ran_weights_one)
        RR_norm = np.sum(ran_weights_one) ** 2 - np.sum(ran_weights_one**2)

        numerator = (DD / DD_norm) - 2 * (DR / DR_norm) + (RR / RR_norm)
        denominator = RR / RR_norm

    else:
        # Cross-correlation: full 4-term estimator
        D1D2 = count2(pos_particles_one, pos_particles_two, battrs=battrs, spin1=0, spin2=0)
        D1R2 = count2(pos_particles_one, ran_particles_two, battrs=battrs, spin1=0, spin2=0)
        D2R1 = count2(pos_particles_two, ran_particles_one, battrs=battrs, spin1=0, spin2=0)
        R1R2 = count2(ran_particles_one, ran_particles_two, battrs=battrs, spin1=0, spin2=0)

        # Normalization: account for particles that can't pair with themselves
        D1D2_norm = np.sum(pos_weights_one) * np.sum(pos_weights_two) - np.sum(pos_weights_one * pos_weights_two)
        D1R2_norm = np.sum(pos_weights_one) * np.sum(ran_weights_two)
        D2R1_norm = np.sum(pos_weights_two) * np.sum(ran_weights_one)
        R1R2_norm = np.sum(ran_weights_one) * np.sum(ran_weights_two) - np.sum(ran_weights_one * ran_weights_two)

        numerator = (D1D2 / D1D2_norm) - (D1R2 / D1R2_norm) - (D2R1 / D2R1_norm) + (R1R2 / R1R2_norm)
        denominator = R1R2 / R1R2_norm

    return numerator / denominator


def __compute_count_spin_correlation__(
    pos_particles,
    spin_particles,
    ran_particles,
    spin_two,
    battrs,
):
    """Compute count-spin correlation (galaxy-shear)."""
    PS_result = count2(pos_particles, spin_particles, battrs=battrs, spin1=0, spin2=spin_two)
    PS_count = count2(pos_particles, spin_particles, battrs=battrs, spin1=0, spin2=0)
    RS_result = count2(ran_particles, spin_particles, battrs=battrs, spin1=0, spin2=spin_two)
    RS_count = count2(ran_particles, spin_particles, battrs=battrs, spin1=0, spin2=0)

    xi_g_plus = (PS_result["plus"] / PS_count) - (RS_result["plus"] / RS_count)
    xi_g_cross = (PS_result["cross"] / PS_count) - (RS_result["cross"] / RS_count)
    return xi_g_plus, xi_g_cross


def __compute_spin_spin_correlation__(
    spin_particles_one,
    spin_one,
    spin_particles_two,
    spin_two,
    battrs,
):
    """Compute spin-spin correlation (shear-shear)."""
    SS_result = count2(spin_particles_one, spin_particles_two, battrs=battrs, spin1=spin_one, spin2=spin_two)
    SS_counts = count2(spin_particles_one, spin_particles_two, battrs=battrs, spin1=0, spin2=0)
    xi_pp = SS_result["plus_plus"] / SS_counts
    xi_xp = SS_result["cross_plus"] / SS_counts
    xi_xx = SS_result["cross_cross"] / SS_counts
    return xi_pp, xi_xp, xi_xx
