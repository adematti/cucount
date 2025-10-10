#!/usr/bin/env python3
"""
Concise TreeCorr comparison for w_gg with DESI LRG data.
Computes galaxy clustering with both cucount and TreeCorr.
"""

import os
import numpy as np
import treecorr
import matplotlib.pyplot as plt
from astropy.io import fits
from cucount.numpy import count2, Particles, BinAttrs

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# Paths
DESI_DATA = '/sps/euclid/Users/cmurray/DESI/catalogs/LRG_NGC_clustering.dat.fits'
DESI_RAND = '/sps/euclid/Users/cmurray/DESI/catalogs/LRG_NGC_0_clustering.ran.fits'

# Angular bins
MIN_THETA = 0.01  # degrees
MAX_THETA = 1.0
NBINS = 20

def load_catalog(path):
    """Load RA, Dec, weights from FITS catalog"""
    with fits.open(path) as hdul:
        data = hdul[1].data
        ra = data['RA']
        dec = data['DEC']
        weights = data['WEIGHT'] * data['WEIGHT_FKP']
    print(f"  Loaded {len(ra)} objects from {path.split('/')[-1]}")
    return ra, dec, weights

def get_cartesian(ra, dec):
    """Convert RA/Dec to unit sphere Cartesian coordinates"""
    conv = np.pi / 180.
    theta, phi = dec * conv, ra * conv
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.column_stack([x, y, z])

def create_particles(ra, dec, weights):
    """Create cucount Particles object (no sky_coords for spin-0)"""
    positions = get_cartesian(ra, dec)
    return Particles(positions, weights)

def compute_wgg_cucount(data_ra, data_dec, data_w, rand_ra, rand_dec, rand_w, theta_edges_rad):
    """Compute w_gg using cucount with Landy-Szalay estimator"""

    # Create particles
    lenses = create_particles(data_ra, data_dec, data_w)
    randoms = create_particles(rand_ra, rand_dec, rand_w)

    # Create bins
    battrs = BinAttrs(theta=theta_edges_rad)

    # Compute pair counts (no spin)
    DD = count2(lenses, lenses, battrs=battrs)
    DR = count2(lenses, randoms, battrs=battrs)
    RR = count2(randoms, randoms, battrs=battrs)

    # Normalization
    DD_norm = np.sum(data_w)**2 - np.sum(data_w**2)
    DR_norm = np.sum(data_w) * np.sum(rand_w)
    RR_norm = np.sum(rand_w)**2 - np.sum(rand_w**2)

    # Landy-Szalay estimator
    xi = ((DD / DD_norm) - 2 * (DR / DR_norm) + (RR / RR_norm)) / (RR / RR_norm)

    return xi, DD, DR, RR

def compute_wgg_treecorr(data_ra, data_dec, data_w, rand_ra, rand_dec, rand_w):
    """Compute w_gg using TreeCorr with Landy-Szalay estimator"""

    # Create catalogs
    data_cat = treecorr.Catalog(ra=data_ra, dec=data_dec, w=data_w,
                                 ra_units='deg', dec_units='deg')
    rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, w=rand_w,
                                 ra_units='deg', dec_units='deg')

    # Setup correlation
    config = {
        'min_sep': MIN_THETA * 60,  # convert to arcmin
        'max_sep': MAX_THETA * 60,
        'nbins': NBINS,
        'sep_units': 'arcmin',
        'bin_type': 'Log'
    }

    # Compute DD, DR, RR
    dd = treecorr.NNCorrelation(**config)
    dd.process(data_cat)

    dr = treecorr.NNCorrelation(**config)
    dr.process(data_cat, rand_cat)

    rr = treecorr.NNCorrelation(**config)
    rr.process(rand_cat)

    # Landy-Szalay: xi = (DD - 2*DR + RR) / RR
    xi, varxi = dd.calculateXi(rr=rr, dr=dr)

    # Get theta centers in degrees
    theta = dd.rnom / 60.0  # arcmin to degrees

    return theta, xi, varxi, dd, dr, rr

def plot_comparison(theta, xi_cucount, xi_treecorr, varxi_treecorr):
    """Plot cucount vs TreeCorr comparison"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: w_gg comparison
    ax = axes[0]
    ax.loglog(theta, xi_cucount, 'b-', marker='o', label='cucount', markersize=4)
    ax.loglog(theta, xi_treecorr, 'r--', marker='s', label='TreeCorr', markersize=4)
    ax.set_xlabel('θ [degrees]')
    ax.set_ylabel('w_gg(θ)')
    ax.set_title('Galaxy Clustering')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Ratio
    ax = axes[1]
    ratio = xi_cucount / xi_treecorr
    ax.semilogx(theta, ratio, 'g-', marker='o', markersize=4)
    ax.axhline(1.0, color='k', linestyle='-', alpha=0.7, label='Perfect agreement')
    ax.axhline(1.1, color='r', linestyle=':', alpha=0.5, label='±10%')
    ax.axhline(0.9, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('θ [degrees]')
    ax.set_ylabel('cucount / TreeCorr')
    ax.set_title('Ratio')
    ax.set_ylim([0.5, 1.5])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'treecorr_comparison.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")
    plt.close()

def main():
    print("=" * 60)
    print("TreeCorr vs cucount comparison for w_gg")
    print("=" * 60)

    print("\nLoading DESI LRG catalogs...")
    data_ra, data_dec, data_w = load_catalog(DESI_DATA)
    rand_ra, rand_dec, rand_w = load_catalog(DESI_RAND)

    # Create theta bins
    theta_edges_deg = np.logspace(np.log10(MIN_THETA), np.log10(MAX_THETA), NBINS + 1)
    theta_centers_deg = np.sqrt(theta_edges_deg[:-1] * theta_edges_deg[1:])  # geometric mean
    theta_edges_rad = theta_edges_deg * (np.pi / 180.0)

    print("\n" + "=" * 60)
    print("Computing w_gg with cucount...")
    print("=" * 60)

    xi_cucount, DD, DR, RR = compute_wgg_cucount(
        data_ra, data_dec, data_w,
        rand_ra, rand_dec, rand_w,
        theta_edges_rad
    )

    print("\n" + "=" * 60)
    print("Computing w_gg with TreeCorr...")
    print("=" * 60)
    theta_treecorr, xi_treecorr, varxi_treecorr, dd, dr, rr = compute_wgg_treecorr(
        data_ra, data_dec, data_w,
        rand_ra, rand_dec, rand_w
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"{'Theta (deg)':<12} {'cucount':<14} {'TreeCorr':<14} {'Ratio':<10}")
    print("-" * 60)
    for i in range(len(theta_centers_deg)):
        ratio = xi_cucount[i] / xi_treecorr[i]
        print(f"{theta_centers_deg[i]:<12.4f} {xi_cucount[i]:<14.6e} {xi_treecorr[i]:<14.6e} {ratio:<10.4f}")

    # Plot comparison
    print("\n" + "=" * 60)
    print("Creating plots...")
    print("=" * 60)
    plot_comparison(theta_centers_deg, xi_cucount, xi_treecorr, varxi_treecorr)

    # Save results
    outfile = os.path.join(OUTPUT_DIR, 'treecorr_comparison_results.npz')
    np.savez(outfile,
             theta=theta_centers_deg,
             xi_cucount=xi_cucount,
             xi_treecorr=xi_treecorr,
             varxi_treecorr=varxi_treecorr,
             DD=DD, DR=DR, RR=RR)
    print(f"Saved results to {outfile}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
