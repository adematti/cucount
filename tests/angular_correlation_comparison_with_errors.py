#!/usr/bin/env python3
"""
Angular correlation comparison with error bars for w_gg, galaxy-shear, and cosmic shear.

Computes error estimates for both cucount and TreeCorr:
- cucount: Jackknife resampling with k-means spatial regions (default: 16 regions)
- TreeCorr: Shot noise variance based on pair counts

While the error methodologies differ, both provide uncertainty estimates that can be
compared to assess consistency between the two implementations.

Measures:
- ξ_gg(θ): Galaxy-galaxy angular correlations using DESI LRG
- ξ_g+(θ), ξ_gx(θ): Galaxy-shape correlations (galaxy-galaxy lensing)
- ξ_++(θ), ξ_+x(θ), ξ_××(θ): Shape-shape correlations (cosmic shear)

Data:
- Lenses: DESI LRG catalog
- Sources: UNIONS shape catalog

Usage:
    # Run all correlations with error bars (full catalogs, jackknife by default)
    python angular_correlation_comparison_with_errors.py

    # Run only galaxy-galaxy clustering
    python angular_correlation_comparison_with_errors.py --correlations gg

    # Use downsampled catalog for faster testing
    python angular_correlation_comparison_with_errors.py --use-downsample

    # Skip error computation for even faster testing
    python angular_correlation_comparison_with_errors.py --no-errors

    # Use different number of jackknife regions (cucount only)
    python angular_correlation_comparison_with_errors.py --n-jackknife-regions 25

    # Subsample catalogs for testing
    python angular_correlation_comparison_with_errors.py --max-lenses 10000 --max-sources 50000
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from cucount.numpy import Particles, BinAttrs
from cucount.correlations import calculate_correlation

# Try to import TreeCorr (optional)
try:
    import treecorr
    TREECORR_AVAILABLE = True
except ImportError:
    TREECORR_AVAILABLE = False
    print("Warning: TreeCorr not available. Running cucount only mode.")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# Default paths
DESI_DATA = '/sps/euclid/Users/cmurray/DESI/catalogs/LRG_NGC_clustering.dat.fits'
DESI_RAND = '/sps/euclid/Users/cmurray/DESI/catalogs/LRG_NGC_0_clustering.ran.fits'
UNIONS_DATA = '/sps/euclid/Users/cmurray/UNIONS/unions_shapepipe_2024_v1.4.1.fits'  # Full catalog by default
UNIONS_DATA_DOWNSAMPLE = '/sps/euclid/Users/cmurray/UNIONS/unions_shapepipe_2024_v1.4.1_downsample.fits'

# Angular bins
MIN_THETA = 0.01  # degrees
MAX_THETA = 1.0
NBINS = 20
N_JACKKNIFE_REGIONS = 4


def load_desi_catalog(path):
    """Load DESI LRG catalog"""
    with fits.open(path) as hdul:
        data = hdul[1].data
        ra = data['RA']
        dec = data['DEC']
        weights = data['WEIGHT'] * data['WEIGHT_FKP']
    print(f"  Loaded {len(ra)} objects from {path.split('/')[-1]}")
    return ra, dec, weights


def load_unions_catalog(path, max_sources=None):
    """Load UNIONS shape catalog with optional subsampling"""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        total_sources = len(data)

        # Subsample during loading if requested
        if max_sources is not None and total_sources > max_sources:
            idx = np.random.choice(total_sources, max_sources, replace=False)
            idx.sort()  # Sort indices for faster disk access
            print(f"  Subsampling {max_sources} from {total_sources:,} sources...")

            # Load only the needed columns for subsampled indices
            ra = np.array(data['RA'][idx])
            dec = np.array(data['Dec'][idx])
            e1 = np.array(data['e1'][idx])
            e2 = np.array(data['e2'][idx])
            weights = np.array(data['w'][idx])
        else:
            # Load all data
            ra = np.array(data['RA'])
            dec = np.array(data['Dec'])
            e1 = np.array(data['e1'])
            e2 = np.array(data['e2'])
            weights = np.array(data['w'])

    print(f"  Loaded {len(ra):,} source galaxies from {path.split('/')[-1]}")
    print(f"  e1 range: [{e1.min():.3f}, {e1.max():.3f}]")
    print(f"  e2 range: [{e2.min():.3f}, {e2.max():.3f}]")
    return ra, dec, weights, e1, e2


def get_cartesian(ra, dec):
    """Convert RA/Dec to unit sphere Cartesian coordinates"""
    conv = np.pi / 180.
    theta, phi = dec * conv, ra * conv
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.column_stack([x, y, z])


def create_particles(ra, dec, weights, ellipticities=None):
    """Create cucount Particles object from sky coordinates

    Parameters
    ----------
    ra : array
        Right ascension in degrees
    dec : array
        Declination in degrees
    weights : array
        Particle weights
    ellipticities : array, optional
        Ellipticity components (e1, e2) as [N, 2] array

    Returns
    -------
    Particles
        cucount Particles object
    """
    # Convert to 3D unit sphere Cartesian coordinates
    positions = get_cartesian(ra, dec)

    # Sky coordinates for spin projections (RA, Dec in radians)
    sky_coords = np.column_stack([ra * np.pi/180, dec * np.pi/180])

    if ellipticities is not None:
        # Create particles with sky coordinates and spin values (for spin-2 fields)
        return Particles(positions, weights, sky_coords, ellipticities)
    else:
        # Create particles with sky coordinates only (for spin-0 fields)
        return Particles(positions, weights, sky_coords)


def compute_wgg_cucount(lenses, randoms, battrs, compute_errors=True, n_jk_regions=16):
    """Compute w_gg using cucount with optional jackknife errors"""

    t0 = time.time()

    if compute_errors:
        print(f"  Computing with jackknife errors ({n_jk_regions} regions)...")
        xi, err = calculate_correlation(
            lenses, lenses,
            spin_one=0, spin_two=0,
            randoms_particles_one=randoms,
            randoms_particles_two=randoms,
            battrs=battrs,
            errors='jackknife',
            n_jackknife_regions=n_jk_regions
        )
    else:
        print("  Computing without errors...")
        xi, err = calculate_correlation(
            lenses, lenses,
            spin_one=0, spin_two=0,
            randoms_particles_one=randoms,
            randoms_particles_two=randoms,
            battrs=battrs,
            errors=None
        )

    elapsed = time.time() - t0
    return xi, err, elapsed


def compute_wg_cucount(lenses, sources, randoms, battrs, compute_errors=True, n_jk_regions=16):
    """Compute galaxy-shear correlations using cucount with optional jackknife errors"""

    t0 = time.time()

    if compute_errors:
        print(f"  Computing with jackknife errors ({n_jk_regions} regions)...")
        (xi_plus, xi_cross), (err_plus, err_cross) = calculate_correlation(
            lenses, sources,
            spin_one=0, spin_two=2,
            randoms_particles_one=randoms,
            battrs=battrs,
            errors='jackknife',
            n_jackknife_regions=n_jk_regions
        )
    else:
        print("  Computing without errors...")
        (xi_plus, xi_cross), _ = calculate_correlation(
            lenses, sources,
            spin_one=0, spin_two=2,
            randoms_particles_one=randoms,
            battrs=battrs,
            errors=None
        )
        err_plus = err_cross = None

    elapsed = time.time() - t0
    return xi_plus, xi_cross, err_plus, err_cross, elapsed


def compute_wss_cucount(sources, battrs, compute_errors=True, n_jk_regions=16):
    """Compute shape-shape correlations using cucount with optional jackknife errors"""

    t0 = time.time()

    if compute_errors:
        print(f"  Computing with jackknife errors ({n_jk_regions} regions)...")
        (xi_pp, xi_xp, xi_xx), (err_pp, err_xp, err_xx) = calculate_correlation(
            sources, sources,
            spin_one=2, spin_two=2,
            battrs=battrs,
            errors='jackknife',
            n_jackknife_regions=n_jk_regions
        )
    else:
        print("  Computing without errors...")
        (xi_pp, xi_xp, xi_xx), _ = calculate_correlation(
            sources, sources,
            spin_one=2, spin_two=2,
            battrs=battrs,
            errors=None
        )
        err_pp = err_xp = err_xx = None

    elapsed = time.time() - t0
    return xi_pp, xi_xp, xi_xx, err_pp, err_xp, err_xx, elapsed


def compute_correlations_treecorr(lenses_data, sources_data, randoms_data, theta_edges_deg,
                                  correlations, compute_errors=True, n_jk_regions=16,
                                  bin_slop=0.01, metric='Euclidean', seed=1234):
    """Compute correlations using TreeCorr with variance estimates

    Parameters
    ----------
    compute_errors : bool
        If True, compute variance estimates (shot noise based)
    n_jk_regions : int
        Number of jackknife regions (for cucount comparison; TreeCorr uses different method)

    Notes
    -----
    TreeCorr computes shot-noise based variances automatically. For true jackknife errors,
    TreeCorr requires defining patches explicitly which is more complex. For comparison
    purposes, we use TreeCorr's built-in variance estimation which is based on pair counts.
    """

    if not TREECORR_AVAILABLE:
        raise ImportError("TreeCorr is not available")

    # Convert theta edges to arcmin for TreeCorr
    min_sep = theta_edges_deg[0] * 60.0
    max_sep = theta_edges_deg[-1] * 60.0
    nbins = len(theta_edges_deg) - 1

    results = {}
    timings = {}

    if compute_errors:
        print(f"  Computing with variance estimates (shot noise based)...")

    if 'gg' in correlations:
        print("  TreeCorr: Computing ξ_gg...")
        t0 = time.time()

        lens_cat = treecorr.Catalog(ra=lenses_data['ra'], dec=lenses_data['dec'],
                                    w=lenses_data['weights'], ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms_data['ra'], dec=randoms_data['dec'],
                                    w=randoms_data['weights'], ra_units='deg', dec_units='deg')

        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        dd.process(lens_cat)

        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        rr.process(rand_cat)

        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        dr.process(lens_cat, rand_cat)

        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        results['treecorr_xi_gg'] = xi

        if compute_errors:
            results['treecorr_err_gg'] = np.sqrt(varxi)
        else:
            results['treecorr_err_gg'] = None

        timings['gg'] = time.time() - t0

    if 'gs' in correlations and sources_data is not None:
        print("  TreeCorr: Computing γ_t, γ_×...")
        t0 = time.time()

        lens_cat = treecorr.Catalog(ra=lenses_data['ra'], dec=lenses_data['dec'],
                                    w=lenses_data['weights'], ra_units='deg', dec_units='deg')
        source_cat = treecorr.Catalog(ra=sources_data['ra'], dec=sources_data['dec'],
                                      w=sources_data['weights'], g1=sources_data['e1'],
                                      g2=sources_data['e2'], ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms_data['ra'], dec=randoms_data['dec'],
                                    w=randoms_data['weights'], ra_units='deg', dec_units='deg')

        ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        ng.process(lens_cat, source_cat)

        rg = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        rg.process(rand_cat, source_cat)

        gamma_t, gamma_x, varxi = ng.calculateXi(rg=rg)
        results['treecorr_xi_g_plus'] = gamma_t
        results['treecorr_xi_g_cross'] = gamma_x

        if compute_errors:
            results['treecorr_err_g_plus'] = np.sqrt(varxi)
            results['treecorr_err_g_cross'] = np.sqrt(varxi)  # TreeCorr returns same variance for both
        else:
            results['treecorr_err_g_plus'] = None
            results['treecorr_err_g_cross'] = None

        timings['gs'] = time.time() - t0

    if 'ss' in correlations and sources_data is not None:
        print("  TreeCorr: Computing ξ_++, ξ_××...")
        t0 = time.time()

        source_cat = treecorr.Catalog(ra=sources_data['ra'], dec=sources_data['dec'],
                                      w=sources_data['weights'], g1=sources_data['e1'],
                                      g2=sources_data['e2'], ra_units='deg', dec_units='deg')

        gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop,
                                    metric=metric)
        gg.process(source_cat)

        # Convert ξ_+, ξ_- to ξ_++, ξ_××
        results['treecorr_xi_plus_plus'] = (gg.xip + gg.xim) / 2.0
        results['treecorr_xi_cross_cross'] = (gg.xip - gg.xim) / 2.0

        if compute_errors:
            # Convert variances
            results['treecorr_err_plus_plus'] = np.sqrt((gg.varxip + gg.varxim) / 4.0)
            results['treecorr_err_cross_cross'] = np.sqrt((gg.varxip + gg.varxim) / 4.0)
        else:
            results['treecorr_err_plus_plus'] = None
            results['treecorr_err_cross_cross'] = None

        timings['ss'] = time.time() - t0

    return results, timings


def compute_chi_squared(data1, data2, err1, err2):
    """Compute chi-squared statistic between two datasets"""
    if err1 is None or err2 is None:
        return None, None

    # Combined error
    err_combined = np.sqrt(err1**2 + err2**2)

    # Avoid division by zero
    mask = err_combined > 0

    if not np.any(mask):
        return None, None

    diff = data1[mask] - data2[mask]
    chi2 = np.sum((diff / err_combined[mask])**2)
    ndof = np.sum(mask)

    return chi2, ndof


def print_timing_table(cucount_timings, treecorr_timings=None):
    """Print a formatted timing comparison table"""

    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # Header
    if treecorr_timings:
        print(f"{'Correlation':<15} {'cucount (s)':<15} {'TreeCorr (s)':<15} {'Speedup':<10}")
        print("-" * 60)
    else:
        print(f"{'Correlation':<15} {'cucount (s)':<15}")
        print("-" * 30)

    # Row labels
    corr_labels = {
        'gg': 'ξ_gg',
        'gs': 'γ_t, γ_×',
        'ss': 'ξ_++, ξ_××'
    }

    # Print each correlation type
    for corr_type in ['gg', 'gs', 'ss']:
        if corr_type in cucount_timings:
            cucount_time = cucount_timings[corr_type]
            label = corr_labels[corr_type]

            if treecorr_timings and corr_type in treecorr_timings:
                treecorr_time = treecorr_timings[corr_type]
                speedup = treecorr_time / cucount_time if cucount_time > 0 else 0
                print(f"{label:<15} {cucount_time:>10.3f}     {treecorr_time:>10.3f}     {speedup:>6.1f}×")
            else:
                print(f"{label:<15} {cucount_time:>10.3f}")

    print("=" * 60)


def print_comparison_statistics(results, correlations):
    """Print statistical comparison between cucount and TreeCorr"""

    print("\n" + "=" * 60)
    print("Statistical Comparison (cucount vs TreeCorr)")
    print("=" * 60)

    comparisons = []

    if 'gg' in correlations and 'xi_gg' in results and 'treecorr_xi_gg' in results:
        chi2, ndof = compute_chi_squared(
            results['xi_gg'], results['treecorr_xi_gg'],
            results.get('err_gg'), results.get('treecorr_err_gg')
        )
        comparisons.append(('ξ_gg', chi2, ndof))

    if 'gs' in correlations:
        if 'xi_g_plus' in results and 'treecorr_xi_g_plus' in results:
            chi2, ndof = compute_chi_squared(
                results['xi_g_plus'], results['treecorr_xi_g_plus'],
                results.get('err_g_plus'), results.get('treecorr_err_g_plus')
            )
            comparisons.append(('γ_t', chi2, ndof))

        if 'xi_g_cross' in results and 'treecorr_xi_g_cross' in results:
            chi2, ndof = compute_chi_squared(
                results['xi_g_cross'], results['treecorr_xi_g_cross'],
                results.get('err_g_cross'), results.get('treecorr_err_g_cross')
            )
            comparisons.append(('γ_×', chi2, ndof))

    if 'ss' in correlations:
        if 'xi_plus_plus' in results and 'treecorr_xi_plus_plus' in results:
            chi2, ndof = compute_chi_squared(
                results['xi_plus_plus'], results['treecorr_xi_plus_plus'],
                results.get('err_plus_plus'), results.get('treecorr_err_plus_plus')
            )
            comparisons.append(('ξ_++', chi2, ndof))

        if 'xi_cross_cross' in results and 'treecorr_xi_cross_cross' in results:
            chi2, ndof = compute_chi_squared(
                results['xi_cross_cross'], results['treecorr_xi_cross_cross'],
                results.get('err_cross_cross'), results.get('treecorr_err_cross_cross')
            )
            comparisons.append(('ξ_××', chi2, ndof))

    if comparisons:
        print(f"{'Quantity':<10} {'χ²':<12} {'DoF':<8} {'χ²/DoF':<12} {'Agreement':<15}")
        print("-" * 60)

        for label, chi2, ndof in comparisons:
            if chi2 is not None and ndof is not None:
                chi2_per_dof = chi2 / ndof
                # Rule of thumb: χ²/DoF ~ 1 is good agreement
                if chi2_per_dof < 1.5:
                    agreement = "Excellent"
                elif chi2_per_dof < 2.0:
                    agreement = "Good"
                elif chi2_per_dof < 3.0:
                    agreement = "Moderate"
                else:
                    agreement = "Poor"

                print(f"{label:<10} {chi2:>10.2f}  {ndof:>6d}  {chi2_per_dof:>10.3f}  {agreement:<15}")
            else:
                print(f"{label:<10} {'N/A':<10}  {'N/A':<6}  {'N/A':<10}  {'No errors':<15}")
    else:
        print("No overlapping measurements found for comparison.")

    print("=" * 60)


def plot_correlations(results, theta_centers, correlations, output_prefix='correlation'):
    """Plot all correlation functions with error bars"""

    n_plots = len(correlations)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Plot w_gg
    if 'gg' in correlations:
        ax = axes[idx]

        # cucount
        if 'xi_gg' in results:
            if results.get('err_gg') is not None:
                ax.errorbar(theta_centers, results['xi_gg'], yerr=results['err_gg'],
                           fmt='o', color='blue', label='cucount', markersize=6,
                           linewidth=2, capsize=3, alpha=0.8)
            else:
                ax.loglog(theta_centers, results['xi_gg'], 'b-', marker='o',
                         label='cucount', markersize=6, linewidth=2)

        # TreeCorr
        if 'treecorr_xi_gg' in results:
            if results.get('treecorr_err_gg') is not None:
                ax.errorbar(theta_centers, results['treecorr_xi_gg'],
                           yerr=results['treecorr_err_gg'],
                           fmt='s', color='red', label='TreeCorr', markersize=4,
                           linewidth=1.5, capsize=3, alpha=0.6)
            else:
                ax.loglog(theta_centers, results['treecorr_xi_gg'], 'r--', marker='s',
                         label='TreeCorr', markersize=4, linewidth=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('ξ_gg(θ)', fontsize=12)
        ax.set_title('Galaxy-Galaxy Clustering', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        idx += 1

    # Plot galaxy-shear
    if 'gs' in correlations:
        ax = axes[idx]

        # cucount
        if 'xi_g_plus' in results:
            if results.get('err_g_plus') is not None:
                ax.errorbar(theta_centers, np.abs(results['xi_g_plus']),
                           yerr=results['err_g_plus'],
                           fmt='o', color='blue', label='cucount γ_t', markersize=5,
                           linewidth=2, capsize=3, alpha=0.8)
                ax.errorbar(theta_centers, np.abs(results['xi_g_cross']),
                           yerr=results['err_g_cross'],
                           fmt='s', color='cyan', label='cucount γ_×', markersize=4,
                           linewidth=2, capsize=3, alpha=0.8)
            else:
                ax.loglog(theta_centers, np.abs(results['xi_g_plus']), 'b-', marker='o',
                         label='cucount γ_t', markersize=5, linewidth=2)
                ax.loglog(theta_centers, np.abs(results['xi_g_cross']), 'b--', marker='s',
                         label='cucount γ_×', markersize=4, linewidth=2)

        # TreeCorr
        if 'treecorr_xi_g_plus' in results:
            if results.get('treecorr_err_g_plus') is not None:
                ax.errorbar(theta_centers, np.abs(results['treecorr_xi_g_plus']),
                           yerr=results['treecorr_err_g_plus'],
                           fmt='o', color='red', label='TreeCorr γ_t', markersize=4,
                           linewidth=1.5, capsize=2, alpha=0.6)
                ax.errorbar(theta_centers, np.abs(results['treecorr_xi_g_cross']),
                           yerr=results['treecorr_err_g_cross'],
                           fmt='s', color='orange', label='TreeCorr γ_×', markersize=3,
                           linewidth=1.5, capsize=2, alpha=0.6)
            else:
                ax.loglog(theta_centers, np.abs(results['treecorr_xi_g_plus']), 'r-',
                         marker='o', label='TreeCorr γ_t', markersize=4, linewidth=1.5, alpha=0.7)
                ax.loglog(theta_centers, np.abs(results['treecorr_xi_g_cross']), 'r--',
                         marker='s', label='TreeCorr γ_×', markersize=3, linewidth=1.5, alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('|γ_t(θ)|, |γ_×(θ)|', fontsize=12)
        ax.set_title('Galaxy-Shear (GGL)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        idx += 1

    # Plot shape-shape
    if 'ss' in correlations:
        ax = axes[idx]

        # cucount
        if 'xi_plus_plus' in results:
            if results.get('err_plus_plus') is not None:
                ax.errorbar(theta_centers, np.abs(results['xi_plus_plus']),
                           yerr=results['err_plus_plus'],
                           fmt='o', color='blue', label='cucount ξ_++', markersize=5,
                           linewidth=2, capsize=3, alpha=0.8)
                ax.errorbar(theta_centers, np.abs(results['xi_cross_cross']),
                           yerr=results['err_cross_cross'],
                           fmt='^', color='cyan', label='cucount ξ_××', markersize=4,
                           linewidth=2, capsize=3, alpha=0.8)
            else:
                ax.loglog(theta_centers, np.abs(results['xi_plus_plus']), 'b-', marker='o',
                         label='cucount ξ_++', markersize=5, linewidth=2)
                ax.loglog(theta_centers, np.abs(results['xi_cross_cross']), 'b:', marker='^',
                         label='cucount ξ_××', markersize=4, linewidth=2)

        # TreeCorr
        if 'treecorr_xi_plus_plus' in results:
            if results.get('treecorr_err_plus_plus') is not None:
                ax.errorbar(theta_centers, np.abs(results['treecorr_xi_plus_plus']),
                           yerr=results['treecorr_err_plus_plus'],
                           fmt='o', color='red', label='TreeCorr ξ_++', markersize=4,
                           linewidth=1.5, capsize=2, alpha=0.6)
                ax.errorbar(theta_centers, np.abs(results['treecorr_xi_cross_cross']),
                           yerr=results['treecorr_err_cross_cross'],
                           fmt='^', color='orange', label='TreeCorr ξ_××', markersize=3,
                           linewidth=1.5, capsize=2, alpha=0.6)
            else:
                ax.loglog(theta_centers, np.abs(results['treecorr_xi_plus_plus']), 'r-',
                         marker='o', label='TreeCorr ξ_++', markersize=4, linewidth=1.5, alpha=0.7)
                ax.loglog(theta_centers, np.abs(results['treecorr_xi_cross_cross']), 'r:',
                         marker='^', label='TreeCorr ξ_××', markersize=3, linewidth=1.5, alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('|ξ(θ)|', fontsize=12)
        ax.set_title('Shape-Shape (Cosmic Shear)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'{output_prefix}_all_correlations.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute angular correlations with error bars using cucount and TreeCorr',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--no-treecorr', action='store_true',
                        help='Skip TreeCorr comparison (cucount only)')
    parser.add_argument('--no-errors', action='store_true',
                        help='Skip error computation (faster for testing)')
    parser.add_argument('--data', type=str, default=DESI_DATA,
                        help='Path to DESI LRG catalog')
    parser.add_argument('--randoms', type=str, default=DESI_RAND,
                        help='Path to DESI random catalog')
    parser.add_argument('--sources', type=str, default=UNIONS_DATA,
                        help='Path to UNIONS shape catalog (default: full version)')
    parser.add_argument('--use-downsample', action='store_true',
                        help='Use downsampled UNIONS catalog for faster testing')
    parser.add_argument('--correlations', nargs='+', choices=['gg', 'gs', 'ss'],
                        default=['gg', 'gs', 'ss'],
                        help='Which correlations to compute')
    parser.add_argument('--min-theta', type=float, default=MIN_THETA,
                        help='Minimum theta in degrees')
    parser.add_argument('--max-theta', type=float, default=MAX_THETA,
                        help='Maximum theta in degrees')
    parser.add_argument('--nbins', type=int, default=NBINS,
                        help='Number of angular bins')
    parser.add_argument('--n-jackknife-regions', type=int, default=N_JACKKNIFE_REGIONS,
                        help='Number of jackknife regions')
    parser.add_argument('--output-prefix', type=str, default='angular_correlation_with_errors',
                        help='Prefix for output files')
    parser.add_argument('--max-lenses', type=int, default=None,
                        help='Maximum number of lens galaxies')
    parser.add_argument('--max-sources', type=int, default=None,
                        help='Maximum number of source galaxies')
    parser.add_argument('--bin-slop', type=float, default=0.2,
                        help='TreeCorr bin slop parameter (tolerance for bin placement)')
    parser.add_argument('--treecorr-metric', type=str, default='Euclidean',
                        choices=['Euclidean', 'Arc'],
                        help='TreeCorr metric to use for distance calculations')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for jackknife region assignment')

    args = parser.parse_args()

    # Use downsampled catalog if requested
    if args.use_downsample:
        args.sources = UNIONS_DATA_DOWNSAMPLE
        print(f"Using downsampled UNIONS catalog: {args.sources}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if TreeCorr should be run
    run_treecorr = not args.no_treecorr and TREECORR_AVAILABLE
    compute_errors = not args.no_errors

    print("=" * 60)
    print("Angular Correlation Analysis with Error Bars")
    print("=" * 60)
    print(f"Mode: {'cucount + TreeCorr comparison' if run_treecorr else 'cucount only'}")
    if compute_errors:
        print(f"Errors: cucount jackknife ({args.n_jackknife_regions} regions) + TreeCorr shot noise")
    else:
        print(f"Errors: Disabled")
    print(f"Correlations: {', '.join(args.correlations)}")
    print("=" * 60)

    # Create theta bins
    theta_edges_deg = np.logspace(np.log10(args.min_theta), np.log10(args.max_theta),
                                   args.nbins + 1)
    theta_centers_deg = np.sqrt(theta_edges_deg[:-1] * theta_edges_deg[1:])  # geometric mean
    theta_edges_rad = theta_edges_deg * (np.pi / 180.0)

    # Create BinAttrs
    battrs = BinAttrs(theta=theta_edges_rad)

    print(f"\nTheta range: {args.min_theta:.3f} - {args.max_theta:.2f} degrees ({args.nbins} bins)")

    # Load catalogs
    print("\n" + "=" * 60)
    print("Loading Catalogs")
    print("=" * 60)

    data_ra, data_dec, data_w = load_desi_catalog(args.data)
    rand_ra, rand_dec, rand_w = load_desi_catalog(args.randoms)

    # Subsample if requested
    if args.max_lenses:
        n = min(args.max_lenses, len(data_ra))
        idx = np.random.choice(len(data_ra), n, replace=False)
        data_ra, data_dec, data_w = data_ra[idx], data_dec[idx], data_w[idx]
        print(f"  Subsampled lenses to {n}")

    # Load sources if needed
    sources = None
    src_ra = src_dec = src_w = src_e1 = src_e2 = None
    if any(c in args.correlations for c in ['gs', 'ss']):
        src_ra, src_dec, src_w, src_e1, src_e2 = load_unions_catalog(args.sources,
                                                                       max_sources=args.max_sources)

        # Create particles with ellipticities (use negative sign convention)
        sources = create_particles(src_ra, src_dec, src_w,
                                   ellipticities=np.column_stack([-src_e1, -src_e2]))

    # Create particle objects
    lenses = create_particles(data_ra, data_dec, data_w)
    randoms = create_particles(rand_ra, rand_dec, rand_w)

    print(f"\nCreated {len(data_ra)} lenses, {len(rand_ra)} randoms" +
          (f", {len(src_ra)} sources" if sources is not None else ""))

    # Compute correlations with cucount
    print("\n" + "=" * 60)
    print("Computing Correlations with cucount")
    print("=" * 60)

    results = {}
    cucount_timings = {}

    if 'gg' in args.correlations:
        print("\n→ ξ_gg(θ): Galaxy-galaxy clustering")
        xi_gg, err_gg, timing = compute_wgg_cucount(lenses, randoms, battrs,
                                                     compute_errors, args.n_jackknife_regions)
        results['xi_gg'] = xi_gg
        results['err_gg'] = err_gg
        cucount_timings['gg'] = timing
        print(f"  Range: [{xi_gg.min():.2e}, {xi_gg.max():.2e}]")
        if err_gg is not None:
            print(f"  Error range: [{err_gg.min():.2e}, {err_gg.max():.2e}]")
        print(f"  Time: {timing:.3f} s")

    if 'gs' in args.correlations and sources is not None:
        print("\n→ γ_t(θ), γ_×(θ): Galaxy-shear (GGL)")
        xi_plus, xi_cross, err_plus, err_cross, timing = compute_wg_cucount(
            lenses, sources, randoms, battrs, compute_errors, args.n_jackknife_regions
        )
        results['xi_g_plus'] = xi_plus
        results['xi_g_cross'] = xi_cross
        results['err_g_plus'] = err_plus
        results['err_g_cross'] = err_cross
        cucount_timings['gs'] = timing
        print(f"  γ_t range: [{xi_plus.min():.2e}, {xi_plus.max():.2e}]")
        print(f"  γ_× range: [{xi_cross.min():.2e}, {xi_cross.max():.2e}]")
        if err_plus is not None:
            print(f"  γ_t error range: [{err_plus.min():.2e}, {err_plus.max():.2e}]")
            print(f"  γ_× error range: [{err_cross.min():.2e}, {err_cross.max():.2e}]")
        print(f"  Time: {timing:.3f} s")

    if 'ss' in args.correlations and sources is not None:
        print("\n→ ξ_++(θ), ξ_+×(θ), ξ_××(θ): Shape-shape (cosmic shear)")
        xi_pp, xi_xp, xi_xx, err_pp, err_xp, err_xx, timing = compute_wss_cucount(
            sources, battrs, compute_errors, args.n_jackknife_regions
        )
        results['xi_plus_plus'] = xi_pp
        results['xi_cross_plus'] = xi_xp
        results['xi_cross_cross'] = xi_xx
        results['err_plus_plus'] = err_pp
        results['err_cross_plus'] = err_xp
        results['err_cross_cross'] = err_xx
        cucount_timings['ss'] = timing
        print(f"  ξ_++ range: [{xi_pp.min():.2e}, {xi_pp.max():.2e}]")
        print(f"  ξ_+× range: [{xi_xp.min():.2e}, {xi_xp.max():.2e}]")
        print(f"  ξ_×× range: [{xi_xx.min():.2e}, {xi_xx.max():.2e}]")
        if err_pp is not None:
            print(f"  ξ_++ error range: [{err_pp.min():.2e}, {err_pp.max():.2e}]")
            print(f"  ξ_+× error range: [{err_xp.min():.2e}, {err_xp.max():.2e}]")
            print(f"  ξ_×× error range: [{err_xx.min():.2e}, {err_xx.max():.2e}]")
        print(f"  Time: {timing:.3f} s")

    # Compute with TreeCorr if requested
    treecorr_timings = None
    if run_treecorr:
        print("\n" + "=" * 60)
        print("Computing Correlations with TreeCorr")
        print("=" * 60)

        lenses_data = {'ra': data_ra, 'dec': data_dec, 'weights': data_w}
        randoms_data = {'ra': rand_ra, 'dec': rand_dec, 'weights': rand_w}
        sources_data = None
        if any(c in args.correlations for c in ['gs', 'ss']):
            sources_data = {'ra': src_ra, 'dec': src_dec, 'weights': src_w,
                           'e1': src_e1, 'e2': src_e2}

        treecorr_results, treecorr_timings = compute_correlations_treecorr(
            lenses_data, sources_data, randoms_data, theta_edges_deg, args.correlations,
            compute_errors=compute_errors, n_jk_regions=args.n_jackknife_regions,
            bin_slop=args.bin_slop, metric=args.treecorr_metric, seed=args.seed
        )
        results.update(treecorr_results)

    # Print timing comparison
    print_timing_table(cucount_timings, treecorr_timings)

    # Print statistical comparison if errors were computed
    if compute_errors and run_treecorr:
        print_comparison_statistics(results, args.correlations)

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    outfile = os.path.join(OUTPUT_DIR, f'{args.output_prefix}_results.npz')
    np.savez(outfile, theta_centers=theta_centers_deg, theta_edges=theta_edges_deg, **results)
    print(f"Saved results to {outfile}")

    # Create plots
    print("\n" + "=" * 60)
    print("Creating Plots")
    print("=" * 60)

    plot_correlations(results, theta_centers_deg, args.correlations, args.output_prefix)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
