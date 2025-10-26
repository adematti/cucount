"""
Fast jackknife estimators using analytic hole subtraction.
Based off of: https://github.com/theonefromnowhere/JK_pycorr/blob/main/CF_JK_ST_conf.py
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
from cucount.numpy import count2, Particles


# ============================================================================
# Utility Functions
# ============================================================================

def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Safely divide two arrays, returning zero where denominator is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def assign_jackknife_regions(particles: Particles, n_regions: int, seed: int = 1234) -> np.ndarray:
    """
    Assign particles to jackknife regions using k-means clustering on sky coordinates.

    Parameters
    ----------
    particles : Particles
        cucount Particles object
    n_regions : int
        Number of jackknife regions
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of region assignments (integers 0 to n_regions-1)
    """
    from sklearn.cluster import KMeans

    # Extract sky coordinates (RA, Dec)
    # particles.sky_coords is shape (n_particles, 2) with [RA, Dec] in radians
    if hasattr(particles, 'sky_coords') and particles.sky_coords is not None:
        coords = particles.sky_coords
    else:
        # Fallback: compute from positions
        # This assumes positions are in cartesian (x, y, z)
        pos = particles.positions
        ra = np.arctan2(pos[:, 1], pos[:, 0])
        dec = np.arcsin(pos[:, 2] / np.linalg.norm(pos, axis=1))
        coords = np.column_stack([ra, dec])

    # Run k-means
    kmeans = KMeans(n_clusters=n_regions, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(coords)

    return assignments


def subset_particles(particles: Particles, mask: np.ndarray) -> Particles:
    """
    Create a new Particles object containing only entries where mask is True.

    Parameters
    ----------
    particles : Particles
        cucount Particles object
    mask : np.ndarray
        Boolean mask array

    Returns
    -------
    Particles
        New Particles object with subset of data
    """
    # Extract attributes
    positions = particles.positions[mask]
    weights = particles.weights[mask]

    # Handle optional attributes
    sky_coords = None
    if hasattr(particles, 'sky_coords') and particles.sky_coords is not None:
        sky_coords = particles.sky_coords[mask]

    spin_values = None
    if hasattr(particles, 'spin_values') and particles.spin_values is not None:
        spin_values = particles.spin_values[mask]

    # Create new Particles object
    if spin_values is not None and sky_coords is not None:
        return Particles(positions, weights, sky_coords, spin_values)
    elif sky_coords is not None:
        return Particles(positions, weights, sky_coords)
    else:
        return Particles(positions, weights)


def compute_jackknife_covariance(jk_samples: np.ndarray) -> np.ndarray:
    """
    Compute jackknife covariance matrix.

    Parameters
    ----------
    jk_samples : np.ndarray
        Jackknife samples, shape (n_regions, n_bins)

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (n_bins, n_bins)
    """
    n_regions = jk_samples.shape[0]
    mean = np.mean(jk_samples, axis=0)
    diff = jk_samples - mean
    cov = (n_regions - 1) / n_regions * (diff.T @ diff)
    return cov


def extract_errors(cov: np.ndarray) -> np.ndarray:
    """
    Extract standard errors from covariance matrix diagonal.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix

    Returns
    -------
    np.ndarray
        Standard errors (square root of diagonal)
    """
    return np.sqrt(np.diag(cov))


# ============================================================================
# Pair Counting Functions
# ============================================================================

def _count_scalar_pairs(
    particles_a: Particles,
    particles_b: Particles,
    bin_attrs,
    n_bins: int,
) -> np.ndarray:
    """
    Count scalar-scalar pairs (spin1=0, spin2=0).

    Parameters
    ----------
    particles_a : Particles
        First particle set
    particles_b : Particles
        Second particle set
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins

    Returns
    -------
    np.ndarray
        Flattened array of pair counts
    """
    if len(particles_a.positions) == 0 or len(particles_b.positions) == 0:
        return np.zeros(n_bins, dtype=float)

    counts = count2(
        particles_a,
        particles_b,
        battrs=bin_attrs,
        spin1=0,
        spin2=0,
    )
    return np.asarray(counts, dtype=float)


def _count_scalar_spin_pairs(
    particles_a: Particles,
    particles_b: Particles,
    spin_b: int,
    bin_attrs,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count scalar-spin pairs (spin1=0, spin2=spin_b).

    Parameters
    ----------
    particles_a : Particles
        First particle set (scalar)
    particles_b : Particles
        Second particle set (with spin)
    spin_b : int
        Spin value of second particle set
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (plus_component, cross_component) arrays
    """
    if len(particles_a.positions) == 0 or len(particles_b.positions) == 0:
        return np.zeros(n_bins, dtype=float), np.zeros(n_bins, dtype=float)

    counts = count2(
        particles_a,
        particles_b,
        battrs=bin_attrs,
        spin1=0,
        spin2=spin_b,
    )
    plus = np.asarray(counts["plus"], dtype=float)
    cross = np.asarray(counts["cross"], dtype=float)
    return plus, cross


def _count_spin_spin_pairs(
    particles_a: Particles,
    particles_b: Particles,
    spin_a: int,
    spin_b: int,
    bin_attrs,
    n_bins: int,
) -> Dict[str, np.ndarray]:
    """
    Count spin-spin pairs (spin1=spin_a, spin2=spin_b).

    Parameters
    ----------
    particles_a : Particles
        First particle set (with spin)
    particles_b : Particles
        Second particle set (with spin)
    spin_a : int
        Spin value of first particle set
    spin_b : int
        Spin value of second particle set
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys "plus_plus", "cross_plus", "cross_cross"
    """
    if len(particles_a.positions) == 0 or len(particles_b.positions) == 0:
        return {
            "plus_plus": np.zeros(n_bins, dtype=float),
            "cross_plus": np.zeros(n_bins, dtype=float),
            "cross_cross": np.zeros(n_bins, dtype=float),
        }

    counts = count2(
        particles_a,
        particles_b,
        battrs=bin_attrs,
        spin1=spin_a,
        spin2=spin_b,
    )
    return {
        "plus_plus": np.asarray(counts["plus_plus"], dtype=float),
        "cross_plus": np.asarray(counts["cross_plus"], dtype=float),
        "cross_cross": np.asarray(counts["cross_cross"], dtype=float),
    }


# ============================================================================
# Fast Jackknife Estimators
# ============================================================================

def run_fast_count_count_jackknife(
    particles_one: Particles,
    particles_two: Particles,
    randoms_one: Particles,
    randoms_two: Particles,
    assignments_one: np.ndarray,
    assignments_two: np.ndarray,
    assignments_randoms_one: np.ndarray,
    assignments_randoms_two: np.ndarray,
    bin_attrs,
    n_bins: int,
    n_regions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast jackknife for count-count correlations using Landy-Szalay estimator.

    Adapted from _fast_gg in fast_jackknife.py.

    Parameters
    ----------
    particles_one : Particles
        First data catalog
    particles_two : Particles
        Second data catalog
    randoms_one : Particles
        First random catalog
    randoms_two : Particles
        Second random catalog
    assignments_one : np.ndarray
        Region assignments for particles_one
    assignments_two : np.ndarray
        Region assignments for particles_two
    assignments_randoms_one : np.ndarray
        Region assignments for randoms_one
    assignments_randoms_two : np.ndarray
        Region assignments for randoms_two
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins
    n_regions : int
        Number of jackknife regions

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (xi_full, jk_samples) where both are 1D arrays
    """
    # Initialize arrays for hole/keep decomposition
    DhDn = np.zeros((n_regions, n_bins), dtype=float)
    DhDh = np.zeros((n_regions, n_bins), dtype=float)
    RhRn = np.zeros((n_regions, n_bins), dtype=float)
    RhRh = np.zeros((n_regions, n_bins), dtype=float)
    DnRh = np.zeros((n_regions, n_bins), dtype=float)
    DhRn = np.zeros((n_regions, n_bins), dtype=float)
    DhRh_cross = np.zeros((n_regions, n_bins), dtype=float)

    dw_total = np.sum(particles_one.weights)
    rw_total = np.sum(randoms_one.weights)
    dw_holes = np.zeros(n_regions)
    rw_holes = np.zeros(n_regions)

    # Loop over regions to compute hole/keep contributions
    for region in range(n_regions):
        pos_mask_hole = assignments_one == region
        rand_mask_hole = assignments_randoms_one == region

        pos_hole = subset_particles(particles_one, pos_mask_hole)
        pos_keep = subset_particles(particles_one, ~pos_mask_hole)
        rand_hole = subset_particles(randoms_one, rand_mask_hole)
        rand_keep = subset_particles(randoms_one, ~rand_mask_hole)

        dw_holes[region] = np.sum(pos_hole.weights)
        rw_holes[region] = np.sum(rand_hole.weights)

        DhDn[region] = _count_scalar_pairs(pos_hole, pos_keep, bin_attrs, n_bins)
        DhDh[region] = _count_scalar_pairs(pos_hole, pos_hole, bin_attrs, n_bins)
        RhRn[region] = _count_scalar_pairs(rand_hole, rand_keep, bin_attrs, n_bins)
        RhRh[region] = _count_scalar_pairs(rand_hole, rand_hole, bin_attrs, n_bins)
        DnRh[region] = _count_scalar_pairs(pos_keep, rand_hole, bin_attrs, n_bins)
        DhRn[region] = _count_scalar_pairs(pos_hole, rand_keep, bin_attrs, n_bins)
        DhRh_cross[region] = _count_scalar_pairs(pos_hole, rand_hole, bin_attrs, n_bins)

    # Compute total counts
    DD_total = np.sum(DhDh + DhDn, axis=0)
    RR_total = np.sum(RhRh + RhRn, axis=0)
    DR_total = np.sum(DnRh + DhRh_cross, axis=0)

    # Compute full correlation
    DD = DD_total / (dw_total * (dw_total + 1))
    DR = DR_total / (dw_total * rw_total)
    RR = RR_total / (rw_total * (rw_total + 1))
    xi_full = _safe_divide((DD - 2 * DR + RR), RR)

    # Compute jackknife samples with alpha correction
    alpha = n_regions / (2 + np.sqrt(2) * (n_regions - 1)) / 4
    jackknife_samples = np.zeros((n_regions, n_bins), dtype=float)

    for region in range(n_regions):
        dw_hole = dw_holes[region]
        rw_hole = rw_holes[region]
        dw_keep = dw_total - dw_hole
        rw_keep = rw_total - rw_hole

        norm_DD = dw_keep * (dw_keep + 1) + 2 * alpha * (dw_hole * dw_keep)
        norm_DR = (rw_keep * dw_keep) + alpha * (
            dw_hole * rw_keep + rw_hole * dw_keep
        )
        norm_RR = rw_keep * (rw_keep + 1) + 2 * alpha * (rw_hole * rw_keep)

        DD_jk = (DD_total - 2 * (1 - alpha) * DhDn[region] - DhDh[region]) / max(norm_DD, 1e-12)
        DR_jk = (
            DR_total
            - (1 - alpha) * (DnRh[region] + DhRn[region])
            - DhRh_cross[region]
        ) / max(norm_DR, 1e-12)
        RR_jk = (
            RR_total - 2 * (1 - alpha) * RhRn[region] - RhRh[region]
        ) / max(norm_RR, 1e-12)

        jk = (DD_jk - 2 * DR_jk + RR_jk)
        jackknife_samples[region] = _safe_divide(jk, RR_jk)

    return xi_full, jackknife_samples


def run_fast_scalar_spin_jackknife(
    particles_positions: Particles,
    particles_shapes: Particles,
    randoms: Particles,
    assignments_pos: np.ndarray,
    assignments_shapes: np.ndarray,
    assignments_randoms: np.ndarray,
    spin_shapes: int,
    bin_attrs,
    n_bins: int,
    n_regions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast jackknife for scalar-spin correlations (galaxy-shear).

    Adapted from _fast_gs in fast_jackknife.py.

    Parameters
    ----------
    particles_positions : Particles
        Position catalog (lenses)
    particles_shapes : Particles
        Shape catalog (sources)
    randoms : Particles
        Random catalog
    assignments_pos : np.ndarray
        Region assignments for positions
    assignments_shapes : np.ndarray
        Region assignments for shapes
    assignments_randoms : np.ndarray
        Region assignments for randoms
    spin_shapes : int
        Spin value of shape catalog
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins
    n_regions : int
        Number of jackknife regions

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (xi_plus_full, xi_cross_full, jk_plus, jk_cross) - all 1D arrays
    """
    # Initialize arrays for hole/keep decomposition
    AhBo_plus = np.zeros((n_regions, n_bins))
    AhBo_cross = np.zeros((n_regions, n_bins))
    AoBh_plus = np.zeros((n_regions, n_bins))
    AoBh_cross = np.zeros((n_regions, n_bins))
    AhBh_plus = np.zeros((n_regions, n_bins))
    AhBh_cross = np.zeros((n_regions, n_bins))

    AhBo_count = np.zeros((n_regions, n_bins))
    AoBh_count = np.zeros((n_regions, n_bins))
    AhBh_count = np.zeros((n_regions, n_bins))

    RAhBo_plus = np.zeros((n_regions, n_bins))
    RAhBo_cross = np.zeros((n_regions, n_bins))
    RAoBh_plus = np.zeros((n_regions, n_bins))
    RAoBh_cross = np.zeros((n_regions, n_bins))
    RAhBh_plus = np.zeros((n_regions, n_bins))
    RAhBh_cross = np.zeros((n_regions, n_bins))

    RAhBo_count = np.zeros((n_regions, n_bins))
    RAoBh_count = np.zeros((n_regions, n_bins))
    RAhBh_count = np.zeros((n_regions, n_bins))

    # Loop over regions
    for region in range(n_regions):
        pos_hole = subset_particles(particles_positions, assignments_pos == region)
        pos_keep = subset_particles(particles_positions, assignments_pos != region)
        shape_hole = subset_particles(particles_shapes, assignments_shapes == region)
        shape_keep = subset_particles(particles_shapes, assignments_shapes != region)
        rand_hole = subset_particles(randoms, assignments_randoms == region)
        rand_keep = subset_particles(randoms, assignments_randoms != region)

        plus, cross = _count_scalar_spin_pairs(pos_hole, shape_keep, spin_shapes, bin_attrs, n_bins)
        AhBo_plus[region] = plus
        AhBo_cross[region] = cross
        AhBo_count[region] = _count_scalar_pairs(pos_hole, shape_keep, bin_attrs, n_bins)

        plus, cross = _count_scalar_spin_pairs(pos_keep, shape_hole, spin_shapes, bin_attrs, n_bins)
        AoBh_plus[region] = plus
        AoBh_cross[region] = cross
        AoBh_count[region] = _count_scalar_pairs(pos_keep, shape_hole, bin_attrs, n_bins)

        plus, cross = _count_scalar_spin_pairs(pos_hole, shape_hole, spin_shapes, bin_attrs, n_bins)
        AhBh_plus[region] = plus
        AhBh_cross[region] = cross
        AhBh_count[region] = _count_scalar_pairs(pos_hole, shape_hole, bin_attrs, n_bins)

        plus, cross = _count_scalar_spin_pairs(rand_hole, shape_keep, spin_shapes, bin_attrs, n_bins)
        RAhBo_plus[region] = plus
        RAhBo_cross[region] = cross
        RAhBo_count[region] = _count_scalar_pairs(rand_hole, shape_keep, bin_attrs, n_bins)

        plus, cross = _count_scalar_spin_pairs(rand_keep, shape_hole, spin_shapes, bin_attrs, n_bins)
        RAoBh_plus[region] = plus
        RAoBh_cross[region] = cross
        RAoBh_count[region] = _count_scalar_pairs(rand_keep, shape_hole, bin_attrs, n_bins)

        plus, cross = _count_scalar_spin_pairs(rand_hole, shape_hole, spin_shapes, bin_attrs, n_bins)
        RAhBh_plus[region] = plus
        RAhBh_cross[region] = cross
        RAhBh_count[region] = _count_scalar_pairs(rand_hole, shape_hole, bin_attrs, n_bins)

    # Compute totals
    PS_plus_total, PS_cross_total = _count_scalar_spin_pairs(
        particles_positions, particles_shapes, spin_shapes, bin_attrs, n_bins
    )
    PS_count_total = _count_scalar_pairs(particles_positions, particles_shapes, bin_attrs, n_bins)
    RS_plus_total, RS_cross_total = _count_scalar_spin_pairs(
        randoms, particles_shapes, spin_shapes, bin_attrs, n_bins
    )
    RS_count_total = _count_scalar_pairs(randoms, particles_shapes, bin_attrs, n_bins)

    # Compute full correlations
    xi_plus_full = _safe_divide(PS_plus_total, PS_count_total) - _safe_divide(RS_plus_total, RS_count_total)
    xi_cross_full = _safe_divide(PS_cross_total, PS_count_total) - _safe_divide(RS_cross_total, RS_count_total)

    # Compute jackknife samples
    jk_plus = np.zeros((n_regions, n_bins))
    jk_cross = np.zeros((n_regions, n_bins))

    for region in range(n_regions):
        ps_count_jk = PS_count_total - AhBo_count[region] - AoBh_count[region] - AhBh_count[region]
        rs_count_jk = RS_count_total - RAhBo_count[region] - RAoBh_count[region] - RAhBh_count[region]

        ps_plus_jk = _safe_divide(
            PS_plus_total - AhBo_plus[region] - AoBh_plus[region] - AhBh_plus[region],
            ps_count_jk,
        )
        ps_cross_jk = _safe_divide(
            PS_cross_total - AhBo_cross[region] - AoBh_cross[region] - AhBh_cross[region],
            ps_count_jk,
        )
        rs_plus_jk = _safe_divide(
            RS_plus_total - RAhBo_plus[region] - RAoBh_plus[region] - RAhBh_plus[region],
            rs_count_jk,
        )
        rs_cross_jk = _safe_divide(
            RS_cross_total - RAhBo_cross[region] - RAoBh_cross[region] - RAhBh_cross[region],
            rs_count_jk,
        )
        jk_plus[region] = ps_plus_jk - rs_plus_jk
        jk_cross[region] = ps_cross_jk - rs_cross_jk

    return xi_plus_full, xi_cross_full, jk_plus, jk_cross


def run_fast_spin_spin_jackknife(
    particles_spins: Particles,
    assignments: np.ndarray,
    spin_one: int,
    spin_two: int,
    bin_attrs,
    n_bins: int,
    n_regions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast jackknife for spin-spin correlations (shear-shear).

    Adapted from _fast_ss in fast_jackknife.py.

    Parameters
    ----------
    particles_spins : Particles
        Spin catalog
    assignments : np.ndarray
        Region assignments for spins
    spin_one : int
        Spin value of first field
    spin_two : int
        Spin value of second field
    bin_attrs : BinAttrs
        Binning attributes
    n_bins : int
        Total number of bins
    n_regions : int
        Number of jackknife regions

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (xi_pp_full, xi_xp_full, xi_xx_full, jk_pp, jk_xp, jk_xx) - all 1D or 2D arrays
    """
    # Initialize arrays for hole/keep decomposition
    AhBo = {"plus_plus": np.zeros((n_regions, n_bins)),
            "cross_plus": np.zeros((n_regions, n_bins)),
            "cross_cross": np.zeros((n_regions, n_bins))}
    AoBh = {"plus_plus": np.zeros((n_regions, n_bins)),
            "cross_plus": np.zeros((n_regions, n_bins)),
            "cross_cross": np.zeros((n_regions, n_bins))}
    AhBh = {"plus_plus": np.zeros((n_regions, n_bins)),
            "cross_plus": np.zeros((n_regions, n_bins)),
            "cross_cross": np.zeros((n_regions, n_bins))}

    AhBo_count = np.zeros((n_regions, n_bins))
    AoBh_count = np.zeros((n_regions, n_bins))
    AhBh_count = np.zeros((n_regions, n_bins))

    # Loop over regions
    for region in range(n_regions):
        shape_hole = subset_particles(particles_spins, assignments == region)
        shape_keep = subset_particles(particles_spins, assignments != region)

        counts = _count_spin_spin_pairs(shape_hole, shape_keep, spin_one, spin_two, bin_attrs, n_bins)
        for key in AhBo:
            AhBo[key][region] = counts[key]
        AhBo_count[region] = _count_scalar_pairs(shape_hole, shape_keep, bin_attrs, n_bins)

        counts = _count_spin_spin_pairs(shape_keep, shape_hole, spin_one, spin_two, bin_attrs, n_bins)
        for key in AoBh:
            AoBh[key][region] = counts[key]
        AoBh_count[region] = _count_scalar_pairs(shape_keep, shape_hole, bin_attrs, n_bins)

        counts = _count_spin_spin_pairs(shape_hole, shape_hole, spin_one, spin_two, bin_attrs, n_bins)
        for key in AhBh:
            AhBh[key][region] = counts[key]
        AhBh_count[region] = _count_scalar_pairs(shape_hole, shape_hole, bin_attrs, n_bins)

    # Compute totals
    totals = _count_spin_spin_pairs(particles_spins, particles_spins, spin_one, spin_two, bin_attrs, n_bins)
    counts_total = _count_scalar_pairs(particles_spins, particles_spins, bin_attrs, n_bins)

    # Compute full correlations
    xi_pp_full = _safe_divide(totals["plus_plus"], counts_total)
    xi_xp_full = _safe_divide(totals["cross_plus"], counts_total)
    xi_xx_full = _safe_divide(totals["cross_cross"], counts_total)

    # Compute jackknife samples
    jk_pp = np.zeros((n_regions, n_bins))
    jk_xp = np.zeros((n_regions, n_bins))
    jk_xx = np.zeros((n_regions, n_bins))

    for region in range(n_regions):
        denom = counts_total - AhBo_count[region] - AoBh_count[region] - AhBh_count[region]

        num_pp = (
            totals["plus_plus"]
            - AhBo["plus_plus"][region]
            - AoBh["plus_plus"][region]
            - AhBh["plus_plus"][region]
        )
        jk_pp[region] = _safe_divide(num_pp, denom)

        num_xp = (
            totals["cross_plus"]
            - AhBo["cross_plus"][region]
            - AoBh["cross_plus"][region]
            - AhBh["cross_plus"][region]
        )
        jk_xp[region] = _safe_divide(num_xp, denom)

        num_xx = (
            totals["cross_cross"]
            - AhBo["cross_cross"][region]
            - AoBh["cross_cross"][region]
            - AhBh["cross_cross"][region]
        )
        jk_xx[region] = _safe_divide(num_xx, denom)

    return xi_pp_full, xi_xp_full, xi_xx_full, jk_pp, jk_xp, jk_xx
