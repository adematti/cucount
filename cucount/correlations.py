"""
Concise correlation helpers built on top of cucount.numpy.count2.

Public API
----------
- calculate_correlation(tracer_one, tracer_two, spin_tracer_one, spin_tracer_two,
                       battrs=None, theta_edges=None, min_theta=0.01, max_theta=1.0, nbins=20,
                       errors=None, n_jackknife_regions=16, seed=1234)

This computes the correlation for arbitrary spin combinations and, if requested,
jackknife errors using a lightweight region assignment. It normalizes spin
statistics by the pair counts (spin-0/0) so the return values are per-pair
averages in each angular bin. For spin=(0,0), the raw per-bin pair counts are
returned (no Landy–Szalay correction here).
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Union, Any

import numpy as np

from .numpy import count2, BinAttrs

# Light-weight type alias for documentation only (avoid strict type-checkers complaining)
CorrType = Union[np.ndarray, Dict[str, np.ndarray]]


def _ensure_battrs(
    battrs: Optional[BinAttrs],
    theta_edges: Optional[np.ndarray],
    min_theta: float,
    max_theta: float,
    nbins: int,
) -> BinAttrs:
    """Return a BinAttrs, constructing one if needed.

    Parameters
    ----------
    theta_edges : array of float, optional
        Angular bin edges in radians. If provided, overrides min/max/nbins.
    min_theta, max_theta : float
        Angular bin range in degrees if theta_edges is None.
    nbins : int
        Number of bins if theta_edges is None.
    """
    if battrs is not None:
        return battrs

    if theta_edges is None:
        # default: construct log-spaced bins in degrees then convert to radians
        edges_deg = np.logspace(np.log10(min_theta), np.log10(max_theta), nbins + 1)
        theta_edges = edges_deg * (np.pi / 180.0)

    return BinAttrs(theta=theta_edges)


def _normalize_result(result: CorrType, pair_counts: np.ndarray) -> CorrType:
    """Normalize correlations by pair counts per bin when applicable.

    For array-like results (e.g., spin=(0,0) returns pair counts), pass through.
    For dict-like results (e.g., spin=(0,2) or (2,2)), divide each component by
    the scalar pair count in the same bin, guarding zeros.
    """
    # If result is a dict of arrays, normalize each by pair counts
    if isinstance(result, dict):
        pc_safe = np.where(pair_counts > 0, pair_counts, np.inf)
        out: Dict[str, np.ndarray] = {}
        for k, v in result.items():
            out[k] = np.divide(v, pc_safe, out=np.zeros_like(v), where=np.isfinite(1.0 / pc_safe))
        return out
    # If it's a 1D array and looks like already per-pair statistic, try to normalize
    # Here assume spin=(0,0) which is pure counts; leave as-is.
    return result


def _extract_sky_ra(particles) -> Optional[np.ndarray]:
    """Attempt to extract RA angle (radians) from a Particles object.

    Returns
    -------
    array or None
        RA values in radians if found, else None.
    """
    # Try common attribute names used when creating Particles(positions, weights, sky_coords, ...)
    for attr in ("sky_coords", "sky", "skycoord", "skycoords"):
        if hasattr(particles, attr):
            sky = getattr(particles, attr)
            if sky is not None and hasattr(sky, "__array__"):
                arr = np.asarray(sky)
                if arr.ndim == 2 and arr.shape[1] >= 1:
                    return arr[:, 0]  # RA in radians
    return None


def _assign_regions(particles, n_regions: int, seed: int) -> np.ndarray:
    """Assign each particle to one of n_regions for jackknife.

    Uses RA-based equal-width segmentation if sky coordinates are available;
    falls back to random chunking otherwise.
    """
    n = len(particles.weights)
    ra = _extract_sky_ra(particles)
    if ra is not None:
        # Map RA in [0, 2pi) to bins 0..n_regions-1
        ra_mod = np.mod(ra, 2 * np.pi)
        # equal-width bins in angle
        edges = np.linspace(0.0, 2 * np.pi, n_regions + 1)
        # np.digitize returns 1..n_regions; subtract 1 => 0..n_regions-1; rightmost edge maps to n_regions
        labels = np.digitize(ra_mod, edges, right=False) - 1
        labels = np.clip(labels, 0, n_regions - 1)
        return labels.astype(int)
    # Fallback: random balanced assignment
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_regions, dtype=int), repeats=(n // n_regions))
    # Distribute the remainder
    remainder = n - labels.size
    if remainder > 0:
        labels = np.concatenate([labels, rng.choice(n_regions, size=remainder, replace=False)])
    rng.shuffle(labels)
    return labels


def _compute_stat(
    tracer_one,
    tracer_two,
    spin1: int,
    spin2: int,
    battrs: BinAttrs,
) -> Tuple[CorrType, np.ndarray]:
    """Compute correlation statistic and base pair counts for normalization."""
    result = count2(tracer_one, tracer_two, battrs=battrs, spin1=spin1, spin2=spin2)
    pair_counts = count2(tracer_one, tracer_two, battrs=battrs, spin1=0, spin2=0)
    normed = _normalize_result(result, pair_counts)
    return normed, pair_counts


def _jk_stats(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jackknife mean and error from stacked samples.

    Parameters
    ----------
    samples : array, shape (n_jk, n_bins)
        Leave-one-out estimates per region.

    Returns
    -------
    mean : array, shape (n_bins,)
    err  : array, shape (n_bins,)
        Jackknife standard error per bin.
    """
    n_jk = samples.shape[0]
    mean = samples.mean(axis=0)
    # (N-1)/N sum (x_i - mean)^2
    fac = (n_jk - 1) / n_jk if n_jk > 1 else 0.0
    var = fac * np.sum((samples - mean) ** 2, axis=0)
    err = np.sqrt(var)
    return mean, err


def calculate_correlation(
    tracer_one,
    tracer_two,
    spin_tracer_one: int,
    spin_tracer_two: int,
    *,
    battrs: Optional[BinAttrs] = None,
    theta_edges: Optional[np.ndarray] = None,
    min_theta: float = 0.01,
    max_theta: float = 1.0,
    nbins: int = 20,
    errors: Optional[str] = None,
    n_jackknife_regions: int = 16,
    seed: int = 1234,
    region_labels_one: Optional[np.ndarray] = None,
    region_labels_two: Optional[np.ndarray] = None,
) -> Tuple[CorrType, Optional[CorrType]]:
    """Compute correlations for arbitrary spins, with optional jackknife errors.

    Parameters
    ----------
    tracer_one, tracer_two : cucount.numpy.Particles
        Input tracers. For auto-correlations, pass the same object for both.
    spin_tracer_one, spin_tracer_two : int
        Spin of each tracer. Typical values are 0 (scalar) or 2 (spin-2).
    battrs : BinAttrs, optional
        Pre-constructed bin attributes. If not provided, theta bins are built
        from theta_edges or min/max/nbins (angles in degrees).
    theta_edges : array-like, optional
        Angular bin edges in radians. Overrides min/max/nbins when provided.
    min_theta, max_theta : float
        Angular bin range in degrees (used if theta_edges is None).
    nbins : int
        Number of angular bins (used if theta_edges is None).
    errors : {None, 'jackknife'}
        If 'jackknife', compute leave-one-out jackknife errors over simple
        sky-based regions (RA segmentation) or random chunks if sky coords
        are unavailable.
    n_jackknife_regions : int
        Number of jackknife regions.
    seed : int
        Random seed for fallback random jackknife partitioning.
    region_labels_one, region_labels_two : array-like of int, optional
        Optional explicit region labels for tracer_one and tracer_two. If provided,
        they must have the same length as the number of particles in each tracer
        and take integer values in [0, n_jackknife_regions-1]. When present,
        these labels are used directly instead of automatic RA-based partitioning.

    Returns
    -------
    corr : array or dict of arrays
        Correlation per bin, normalized by pair counts when applicable.
        - spin=(0,0): returns pair counts (no LS estimator here)
        - spin=(0,2): dict with keys 'plus', 'cross'
        - spin=(2,2): dict with keys like 'plus_plus', 'cross_plus', 'cross_cross'
    err : same structure as corr or None
        Jackknife standard error per bin if errors=='jackknife', else None.
    """
    battrs = _ensure_battrs(battrs, theta_edges, min_theta, max_theta, nbins)

    # Compute full-sample statistic first (with original weights)
    corr_full, _ = _compute_stat(tracer_one, tracer_two, spin_tracer_one, spin_tracer_two, battrs)

    if errors is None or errors.lower() != "jackknife":
        return corr_full, None

    # Prepare jackknife partitions
    nreg = int(n_jackknife_regions)
    if nreg < 2:
        # Not enough regions to jackknife; return None for errors
        return corr_full, None

    # Save original weights to restore after each iteration
    w1_orig = np.array(tracer_one.weights, copy=True)
    w2_orig = np.array(tracer_two.weights, copy=True)

    if region_labels_one is not None:
        labels1 = np.asarray(region_labels_one, dtype=int)
    else:
        labels1 = _assign_regions(tracer_one, nreg, seed)
    if region_labels_two is not None:
        labels2 = np.asarray(region_labels_two, dtype=int)
    else:
        labels2 = _assign_regions(tracer_two, nreg, seed)

    # Basic validation
    if labels1.ndim != 1 or labels2.ndim != 1:
        raise ValueError("region_labels must be 1D arrays if provided")
    if labels1.shape[0] != len(tracer_one.weights) or labels2.shape[0] != len(tracer_two.weights):
        raise ValueError("region_labels length must match number of particles for each tracer")

    # Helper to compute one leave-one-out estimate with weights zeroed in region r
    def loo_once(region: int) -> CorrType:
        # Restore original
        tracer_one.weights[:] = w1_orig
        tracer_two.weights[:] = w2_orig
        # Zero-out weights for region in both tracers
        if labels1.size:
            tracer_one.weights[labels1 == region] = 0.0
        if labels2.size:
            tracer_two.weights[labels2 == region] = 0.0
        # Compute statistic on this resample
        stat, _ = _compute_stat(tracer_one, tracer_two, spin_tracer_one, spin_tracer_two, battrs)
        return stat

    # Collect jackknife samples
    # Dict- vs array-shaped outputs: branch the accumulator accordingly
    sample0 = loo_once(0)
    if isinstance(sample0, dict):
        # Initialize per-key stacks
        stacks: Dict[str, list] = {k: [v] for k, v in sample0.items()}
        for r in range(1, nreg):
            sr = loo_once(r)
            for k in stacks.keys():
                stacks[k].append(sr[k])
        # Stack and compute jk error per key
        corr_err: Dict[str, np.ndarray] = {}
        corr_mean: Dict[str, np.ndarray] = {}
        for k, lst in stacks.items():
            arr = np.stack(lst, axis=0)
            mean, err = _jk_stats(arr)
            corr_mean[k] = mean
            corr_err[k] = err
    else:
        # Array case
        arr0 = np.asarray(sample0)
        arrs = [arr0]
        for r in range(1, nreg):
            arrs.append(np.asarray(loo_once(r)))
        stack = np.stack(arrs, axis=0)
        corr_mean, corr_err = _jk_stats(stack)

    # Restore original weights at the end
    tracer_one.weights[:] = w1_orig
    tracer_two.weights[:] = w2_orig

    return corr_full, (corr_err if not isinstance(sample0, dict) else corr_err)
