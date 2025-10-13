#!/usr/bin/env python3
"""
Angular correlation comparison for w_gg, galaxy-shear, and cosmic shear with optional TreeCorr validation.

Measures:
- ξ_gg(θ): Galaxy-galaxy angular correlations using DESI LRG
- ξ_g+(θ), ξ_gx(θ): Galaxy-shape correlations (galaxy-galaxy lensing)
- ξ_++(θ), ξ_+x(θ), ξ_××(θ): Shape-shape correlations (cosmic shear)

Data:
- Lenses: DESI LRG catalog
- Sources: UNIONS shape catalog

Usage:
    # Run all correlations with TreeCorr comparison
    python angular_correlation_comparison.py

    # Run only galaxy-galaxy clustering (cucount only, no TreeCorr)
    python angular_correlation_comparison.py --correlations gg --no-treecorr

    # Run galaxy-shear with custom angular bins
    python angular_correlation_comparison.py --correlations gs --no-treecorr --min-theta 0.02 --max-theta 2.0 --nbins 15

    # Subsample catalogs for testing
    python angular_correlation_comparison.py --max-lenses 10000 --max-sources 50000

    # Use full UNIONS catalog (slower)
    python angular_correlation_comparison.py --use-full-sources
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from cucount.numpy import count2, Particles, BinAttrs

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
UNIONS_DATA = '/sps/euclid/Users/cmurray/UNIONS/unions_shapepipe_2024_v1.4.1_downsample.fits'
UNIONS_DATA_FULL = '/sps/euclid/Users/cmurray/UNIONS/unions_shapepipe_2024_v1.4.1.fits'

# Angular bins
MIN_THETA = 0.01  # degrees
MAX_THETA = 1.0
NBINS = 20

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

def compute_wgg_cucount(lenses, randoms, theta_edges_rad):
    """Compute w_gg using cucount with Landy-Szalay estimator"""

    t0 = time.time()

    # Compute pair counts (no spin)
    print("  Computing DD...")
    DD = count2(lenses, lenses, battrs=BinAttrs(theta=theta_edges_rad))
    print("  Computing DR...")
    DR = count2(lenses, randoms, battrs=BinAttrs(theta=theta_edges_rad))
    print("  Computing RR...")
    RR = count2(randoms, randoms, battrs=BinAttrs(theta=theta_edges_rad))

    # Normalization
    DD_norm = np.sum(lenses.weights)**2 - np.sum(lenses.weights**2)
    DR_norm = np.sum(lenses.weights) * np.sum(randoms.weights)
    RR_norm = np.sum(randoms.weights)**2 - np.sum(randoms.weights**2)

    # Landy-Szalay estimator
    xi = ((DD / DD_norm) - 2 * (DR / DR_norm) + (RR / RR_norm)) / (RR / RR_norm)

    elapsed = time.time() - t0
    return xi, elapsed

def compute_wg_cucount(lenses, sources, randoms, theta_edges_rad):
    """Compute galaxy-shear correlations using cucount"""

    t0 = time.time()

    battrs = BinAttrs(theta=theta_edges_rad)

    # Position-shape correlations
    print("  Computing PS...")
    PS_result = count2(lenses, sources, battrs=battrs, spin1=0, spin2=2)
    PS_eplus = PS_result['plus']
    PS_ecross = PS_result['cross']

    # Random-shape correlations
    print("  Computing RS...")
    RS_result = count2(randoms, sources, battrs=battrs, spin1=0, spin2=2)
    RS_eplus = RS_result['plus']
    RS_ecross = RS_result['cross']

    # Regular pair counts for normalization
    print("  Computing pair counts...")
    PS_count = count2(lenses, sources, battrs=battrs, spin1=0, spin2=0)
    RS_count = count2(randoms, sources, battrs=battrs, spin1=0, spin2=0)

    # Galaxy-galaxy lensing estimator
    gamma_t = np.where(RS_count > 0, (PS_eplus / PS_count - RS_eplus / RS_count), 0)
    gamma_x = np.where(RS_count > 0, (PS_ecross / PS_count - RS_ecross / RS_count), 0)

    elapsed = time.time() - t0
    return gamma_t, gamma_x, elapsed

def compute_wss_cucount(sources, theta_edges_rad):
    """Compute shape-shape correlations using cucount"""

    t0 = time.time()

    battrs = BinAttrs(theta=theta_edges_rad)

    # Shape-shape correlation
    print("  Computing SS...")
    xi_ss = count2(sources, sources, battrs=battrs, spin1=2, spin2=2)

    # Get normalizing pair counts
    print("  Computing pair counts...")
    SS_counts = count2(sources, sources, battrs=battrs, spin1=0, spin2=0)

    # Normalize by pair counts
    xi_plus_plus = np.where(SS_counts > 0, xi_ss['plus_plus'] / SS_counts, 0)
    xi_cross_plus = np.where(SS_counts > 0, xi_ss['cross_plus'] / SS_counts, 0)
    xi_cross_cross = np.where(SS_counts > 0, xi_ss['cross_cross'] / SS_counts, 0)

    elapsed = time.time() - t0
    return xi_plus_plus, xi_cross_plus, xi_cross_cross, elapsed

def compute_correlations_treecorr(lenses_data, sources_data, randoms_data, theta_edges_deg, correlations, bin_slop=0.01, metric='Euclidean'):
    """Compute correlations using TreeCorr for comparison

    Parameters
    ----------
    metric : str
        Distance metric for TreeCorr ('Euclidean' or 'Arc')
    """

    if not TREECORR_AVAILABLE:
        raise ImportError("TreeCorr is not available")

    # Convert theta edges to arcmin for TreeCorr
    min_sep = theta_edges_deg[0] * 60.0
    max_sep = theta_edges_deg[-1] * 60.0
    nbins = len(theta_edges_deg) - 1

    results = {}
    timings = {}

    if 'gg' in correlations:
        print("  TreeCorr: Computing ξ_gg...")
        t0 = time.time()

        lens_cat = treecorr.Catalog(ra=lenses_data['ra'], dec=lenses_data['dec'],
                                    w=lenses_data['weights'], ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms_data['ra'], dec=randoms_data['dec'],
                                    w=randoms_data['weights'], ra_units='deg', dec_units='deg')

        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        dd.process(lens_cat)

        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        rr.process(rand_cat)

        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        dr.process(lens_cat, rand_cat)

        xi, _ = dd.calculateXi(rr=rr, dr=dr)
        results['treecorr_xi_gg'] = xi
        timings['gg'] = time.time() - t0

    if 'gs' in correlations and sources_data is not None:
        print("  TreeCorr: Computing γ_t, γ_×...")
        t0 = time.time()

        lens_cat = treecorr.Catalog(ra=lenses_data['ra'], dec=lenses_data['dec'],
                                    w=lenses_data['weights'], ra_units='deg', dec_units='deg')
        source_cat = treecorr.Catalog(ra=sources_data['ra'], dec=sources_data['dec'],
                                      w=sources_data['weights'], g1=sources_data['e1'], g2=sources_data['e2'],
                                      ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms_data['ra'], dec=randoms_data['dec'],
                                    w=randoms_data['weights'], ra_units='deg', dec_units='deg')

        ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        ng.process(lens_cat, source_cat)

        rg = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        rg.process(rand_cat, source_cat)

        gamma_t, gamma_x, _ = ng.calculateXi(rg=rg)
        results['treecorr_xi_g_plus'] = gamma_t
        results['treecorr_xi_g_cross'] = gamma_x
        timings['gs'] = time.time() - t0

    if 'ss' in correlations and sources_data is not None:
        print("  TreeCorr: Computing ξ_++, ξ_××...")
        t0 = time.time()

        source_cat = treecorr.Catalog(ra=sources_data['ra'], dec=sources_data['dec'],
                                      w=sources_data['weights'], g1=sources_data['e1'], g2=sources_data['e2'],
                                      ra_units='deg', dec_units='deg')

        gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    sep_units='arcmin', bin_type='Log', bin_slop=bin_slop, metric=metric)
        gg.process(source_cat)

        # Convert ξ_+, ξ_- to ξ_++, ξ_××
        results['treecorr_xi_plus_plus'] = (gg.xip + gg.xim) / 2.0
        results['treecorr_xi_cross_cross'] = (gg.xip - gg.xim) / 2.0
        timings['ss'] = time.time() - t0

    return results, timings

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

def plot_correlations(results, theta_centers, correlations, output_prefix='correlation'):
    """Plot all correlation functions"""

    n_plots = len(correlations)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Plot w_gg
    if 'gg' in correlations:
        ax = axes[idx]
        if 'xi_gg' in results:
            ax.loglog(theta_centers, results['xi_gg'], 'b-', marker='o',
                     label='cucount', markersize=6, linewidth=2)
        if 'treecorr_xi_gg' in results:
            ax.loglog(theta_centers, results['treecorr_xi_gg'], 'r--', marker='s',
                     label='TreeCorr', markersize=4, linewidth=2)
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('ξ_gg(θ)', fontsize=12)
        ax.set_title('Galaxy-Galaxy Clustering', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        idx += 1

    # Plot galaxy-shear
    if 'gs' in correlations:
        ax = axes[idx]
        if 'xi_g_plus' in results:
            ax.loglog(theta_centers, results['xi_g_plus'], 'b-', marker='o',
                     label='cucount γ_t', markersize=5, linewidth=2)
            ax.loglog(theta_centers, results['xi_g_cross'], 'b--', marker='s',
                     label='cucount γ_×', markersize=4, linewidth=2)
        if 'treecorr_xi_g_plus' in results:
            ax.loglog(theta_centers, results['treecorr_xi_g_plus'], 'r-', marker='o',
                     label='TreeCorr γ_t', markersize=4, linewidth=1.5, alpha=0.7)
            ax.loglog(theta_centers, results['treecorr_xi_g_cross'], 'r--', marker='s',
                     label='TreeCorr γ_×', markersize=3, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('γ_t(θ), γ_×(θ)', fontsize=12)
        ax.set_title('Galaxy-Shear (GGL)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        idx += 1

    # Plot shape-shape
    if 'ss' in correlations:
        ax = axes[idx]
        if 'xi_plus_plus' in results:
            ax.loglog(theta_centers, results['xi_plus_plus'], 'b-', marker='o',
                     label='cucount ξ_++', markersize=5, linewidth=2)
            ax.loglog(theta_centers, results['xi_cross_plus'], 'b--', marker='s',
                     label='cucount ξ_+×', markersize=4, linewidth=2)
            ax.loglog(theta_centers, results['xi_cross_cross'], 'b:', marker='^',
                     label='cucount ξ_××', markersize=4, linewidth=2)
        if 'treecorr_xi_plus_plus' in results:
            ax.loglog(theta_centers, results['treecorr_xi_plus_plus'], 'r-', marker='o',
                     label='TreeCorr ξ_++', markersize=4, linewidth=1.5, alpha=0.7)
            ax.loglog(theta_centers, results['treecorr_xi_cross_cross'], 'r:', marker='^',
                     label='TreeCorr ξ_××', markersize=3, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('θ [degrees]', fontsize=12)
        ax.set_ylabel('ξ(θ)', fontsize=12)
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
        description='Compute angular correlations with cucount and optionally compare with TreeCorr',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--no-treecorr', action='store_true',
                        help='Skip TreeCorr comparison (cucount only)')
    parser.add_argument('--data', type=str, default=DESI_DATA,
                        help='Path to DESI LRG catalog')
    parser.add_argument('--randoms', type=str, default=DESI_RAND,
                        help='Path to DESI random catalog')
    parser.add_argument('--sources', type=str, default=UNIONS_DATA,
                        help='Path to UNIONS shape catalog (default: downsampled version)')
    parser.add_argument('--use-full-sources', action='store_true',
                        help='Use full UNIONS catalog instead of downsampled version')
    parser.add_argument('--correlations', nargs='+', choices=['gg', 'gs', 'ss'],
                        default=['gg', 'gs', 'ss'],
                        help='Which correlations to compute')
    parser.add_argument('--min-theta', type=float, default=MIN_THETA,
                        help='Minimum theta in degrees')
    parser.add_argument('--max-theta', type=float, default=MAX_THETA,
                        help='Maximum theta in degrees')
    parser.add_argument('--nbins', type=int, default=NBINS,
                        help='Number of angular bins')
    parser.add_argument('--output-prefix', type=str, default='angular_correlation',
                        help='Prefix for output files')
    parser.add_argument('--max-lenses', type=int, default=None,
                        help='Maximum number of lens galaxies')
    parser.add_argument('--max-sources', type=int, default=None,
                        help='Maximum number of source galaxies')
    parser.add_argument('--bin-slop', type=float, default=0.2,
                        help='TreeCorr bin slop parameter (tolerance for bin placement)')
    parser.add_argument('--treecorr-metric', type=str, default='Euclidean', choices=['Euclidean', 'Arc'],
                        help='TreeCorr metric to use for distance calculations')

    args = parser.parse_args()

    # Use full catalog if requested
    if args.use_full_sources:
        args.sources = UNIONS_DATA_FULL
        print(f"Using full UNIONS catalog: {args.sources}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if TreeCorr should be run
    run_treecorr = not args.no_treecorr and TREECORR_AVAILABLE

    print("=" * 60)
    print("Angular Correlation Analysis")
    print("=" * 60)
    print(f"Mode: {'cucount + TreeCorr comparison' if run_treecorr else 'cucount only'}")
    print(f"Correlations: {', '.join(args.correlations)}")
    print("=" * 60)

    # Create theta bins
    theta_edges_deg = np.logspace(np.log10(args.min_theta), np.log10(args.max_theta), args.nbins + 1)
    theta_centers_deg = np.sqrt(theta_edges_deg[:-1] * theta_edges_deg[1:])  # geometric mean
    theta_edges_rad = theta_edges_deg * (np.pi / 180.0)

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
        src_ra, src_dec, src_w, src_e1, src_e2 = load_unions_catalog(args.sources, max_sources=args.max_sources)

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
        results['xi_gg'], cucount_timings['gg'] = compute_wgg_cucount(lenses, randoms, theta_edges_rad)
        print(f"  Range: [{results['xi_gg'].min():.2e}, {results['xi_gg'].max():.2e}]")
        print(f"  Time: {cucount_timings['gg']:.3f} s")

    if 'gs' in args.correlations and sources is not None:
        print("\n→ γ_t(θ), γ_×(θ): Galaxy-shear (GGL)")
        results['xi_g_plus'], results['xi_g_cross'], cucount_timings['gs'] = compute_wg_cucount(lenses, sources, randoms, theta_edges_rad)
        print(f"  γ_t range: [{results['xi_g_plus'].min():.2e}, {results['xi_g_plus'].max():.2e}]")
        print(f"  γ_× range: [{results['xi_g_cross'].min():.2e}, {results['xi_g_cross'].max():.2e}]")
        print(f"  Time: {cucount_timings['gs']:.3f} s")

    if 'ss' in args.correlations and sources is not None:
        print("\n→ ξ_++(θ), ξ_+×(θ), ξ_××(θ): Shape-shape (cosmic shear)")
        results['xi_plus_plus'], results['xi_cross_plus'], results['xi_cross_cross'], cucount_timings['ss'] = compute_wss_cucount(sources, theta_edges_rad)
        print(f"  ξ_++ range: [{results['xi_plus_plus'].min():.2e}, {results['xi_plus_plus'].max():.2e}]")
        print(f"  ξ_+× range: [{results['xi_cross_plus'].min():.2e}, {results['xi_cross_plus'].max():.2e}]")
        print(f"  ξ_×× range: [{results['xi_cross_cross'].min():.2e}, {results['xi_cross_cross'].max():.2e}]")
        print(f"  Time: {cucount_timings['ss']:.3f} s")

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

        treecorr_results, treecorr_timings = compute_correlations_treecorr(lenses_data, sources_data, randoms_data,
                                                                           theta_edges_deg, args.correlations,
                                                                           bin_slop=args.bin_slop, metric=args.treecorr_metric)
        results.update(treecorr_results)

    # Print timing comparison
    print_timing_table(cucount_timings, treecorr_timings)

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
