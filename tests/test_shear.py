import numpy as np


def generate_ellipticities(size, sigma_e=0.3, seed=42):
    """Generate random galaxy ellipticities e1, e2."""
    rng = np.random.RandomState(seed=seed)
    # Random orientation angles (0 to pi)
    phi = np.random.uniform(0, np.pi, size)
    e_mag = rng.normal(0, sigma_e, size)
    e_mag = np.clip(np.abs(e_mag), 0, 0.9)
    # Compute components
    e1 = e_mag * np.cos(2 * phi)
    e2 = e_mag * np.sin(2 * phi)
    return e1, e2


def generate_catalogs(size=100, limits=((0., 10.), (0., 10.)), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        # Uniform in RA
        ra = np.random.uniform(limits[0], size)
        # Uniform in sin(Dec)
        sin_dec = np.random.uniform(*np.sin(np.array(limits) * np.pi / 180.), size)
        dec = np.arcsin(sin_dec)
        positions = [ra, dec]
        ellipticies = list(generate_ellipticities(size, seed=seed))
        weights = []
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        # weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + ellipticies + weights)
    return toret


def test_treecorr():
    from cucount.numpy import BinAttrs, Particles, count2
    import treecorr

    def get_cartesian(ra, dec):
        """Convert RA/Dec to unit sphere Cartesian coordinates"""
        conv = np.pi / 180.
        theta, phi = dec * conv, ra * conv
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        return np.column_stack([x, y, z])

    def create_cucount_particles(catalog, with_ellipticies=False):
        ra, dec, e1, e2, weights = catalog
        # Convert to 3D unit sphere Cartesian coordinates
        positions = get_cartesian(ra, dec)
        # Sky coordinates for spin projections (RA, Dec in radians)
        sky_coords = np.column_stack([ra * np.pi/180, dec * np.pi/180])
        return Particles(positions, weights, sky_coords, ellipticities=np.column_stack([e1, e2]) if with_ellipticies else None)

    def create_treecorr_catalog(catalog):
        ra, dec, e1, e2, weights = catalog
        return treecorr.Catalog(ra=ra, dec=dec, weights=weights, g1=e1, g2=e2, ra_units='deg', dec_units='deg')

    edges = np.logspace(-3., -1., 10)
    battrs = BinAttrs(theta=edges)
    catalogs = generate_catalogs(size=1000)

    # Convert theta edges to arcmin for TreeCorr
    min_sep = edges[0] * 60.0
    max_sep = edges[-1] * 60.0
    nbins = len(edges) - 1

    # gg
    particles = [create_cucount_particles(catalog, with_ellipticies=False) for catalog in catalogs]
    counts = count2(*particles, battrs=battrs, spin1=0, spin2=0)

    tc = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='arcmin', bin_type='Log', bin_slop=0., metric='Euclidean')
    tc.process(*[create_treecorr_catalog(catalog) for catalog in catalogs])
    print(tc.__dict__)

    # gs
    particles = [create_cucount_particles(catalogs[0], with_ellipticies=False),
                 create_cucount_particles(catalogs[0], with_ellipticies=True)]
    counts = count2(*particles, battrs=battrs, spin1=0, spin2=2)

    tc = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='arcmin', bin_type='Log', bin_slop=0., metric='Euclidean')
    tc.process(*[create_treecorr_catalog(catalog) for catalog in catalogs])
    print(tc.__dict__)

    # ss
    particles = [create_cucount_particles(catalog, with_ellipticies=True) for catalog in catalogs]
    counts = count2(*particles, battrs=battrs, spin1=2, spin2=2)

    tc = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='arcmin', bin_type='Log', bin_slop=0., metric='Euclidean')
    tc.process(*[create_treecorr_catalog(catalog) for catalog in catalogs])
    print(tc.__dict__)



if __name__ == '__main__':

    test_treecorr()