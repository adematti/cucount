import numpy as np

from cucount.numpy import Particles, MeshAttrs, WeightAttrs, BinAttrs


class BoxSubsampler(object):
    """
    Spatial subsampler that divides a 3D box into regular subregions.

    This subsampler partitions the bounding box into a regular grid of subboxes
    and assigns particle labels based on which subbox contains them. Supports
    periodic and non-periodic boundary conditions.

    Attributes
    ----------
    particles : Particles, optional
        Input particle positions and metadata.
    nsplits : array-like of shape (3,)
        Number of divisions along each axis (x, y, z).
    mattrs : MeshAttrs
        Mesh attributes defining the bounding box.
    edges : list of array
        Bin edges for each spatial dimension.
    """

    def __init__(self, particles=None, nsplits: int = 8, mattrs: MeshAttrs = None):
        """
        Initialize :class:`BoxSubsampler`.

        Parameters
        ----------
        particles : Particles, optional
            Input particles. Required if ``mattrs`` is not provided.
        nsplits : int or array-like of shape (3,), default=8
            Number of divisions per axis. If scalar, interpreted as the cube root
            (e.g., ``nsplits=8`` → 2×2×2 divisions). If array-like, must have length 3
            with one element per axis.
        mattrs : MeshAttrs, optional
            Mesh attributes (box center, size, periodicity). If ``None``, computed
            from ``particles``.

        Examples
        --------
        Create a 2×2×2 grid from particles:

        >>> particles = Particles(positions)
        >>> subsampler = BoxSubsampler(particles, nsplits=8)
        >>> labels = subsampler.label(particles)

        Use explicit mesh attributes:

        >>> mattrs = MeshAttrs(boxcenter=[500, 500, 500], boxsize=[1000, 1000, 1000])
        >>> subsampler = BoxSubsampler(nsplits=[4, 4, 4], mattrs=mattrs)
        """
        self.particles = particles
        self.nsplits = nsplits

        if mattrs is None:
            assert particles is not None, 'Provide particles when mattrs is not provided'
            mattrs = MeshAttrs(particles, battrs=BinAttrs(s=np.linspace(0., 1., 2)))
        self.mattrs = mattrs

        ndim = 3
        if np.ndim(nsplits) == 0:
            _nsplits = nsplits
            nsplits = np.full(ndim, int(np.rint(_nsplits ** (1. / ndim))))
            assert _nsplits == np.prod(nsplits), \
                f'nsplits={_nsplits} must be a {ndim}-th power of an integer'

        nsplits = np.ravel(nsplits)
        assert nsplits.size == ndim, \
            f'nsplits must be a list/tuple of size {ndim}, got {nsplits.size}'
        assert np.issubdtype(nsplits.dtype, np.integer) and np.all(nsplits), \
            f'nsplits must contain positive integers, got {nsplits}'

        self.nsplits = nsplits
        offset = self.mattrs.boxcenter - self.mattrs.boxsize / 2.
        self.edges = [o + np.linspace(0, b, n + 1)
                     for o, b, n in zip(offset, self.mattrs.boxsize, self.nsplits)]

    def label(self, positions):
        """
        Assign subregion labels to input positions.

        Positions are assigned labels based on which subregion they occupy in the
        regular grid. With periodic boundary conditions, positions are wrapped
        before assignment.

        Parameters
        ----------
        positions : array of shape (n_particles, 3) or Particles
            Particle positions. If ``Particles`` object, extracts ``.positions`` attribute.

        Returns
        -------
        labels : ndarray of shape (n_particles,)
            Integer labels in range [0, prod(nsplits)) corresponding to subregion indices.

        Examples
        --------
        >>> subsampler = BoxSubsampler(particles, nsplits=8)
        >>> labels = subsampler.label(particles)
        >>> unique_labels = np.unique(labels)
        >>> print(f"Found {len(unique_labels)} subregions")
        """
        positions = positions.positions if isinstance(positions, Particles) else positions
        indices = []
        offset = self.mattrs.boxcenter - self.mattrs.boxsize / 2.

        for edges, o, b, p in zip(self.edges, offset, self.mattrs.boxsize, positions.T):
            # Wrap periodic coordinates
            if self.mattrs.periodic:
                p = (p - o) % b + o

            # Find bin indices
            tmp = np.searchsorted(edges, p, side='right') - 1

            # Validate bounds
            if not np.all((tmp >= 0) & (tmp < len(edges) - 1)):
                raise ValueError(f'Some input positions outside bounding box '
                               f'[{edges[0]:.3f}, {edges[-1]:.3f}]')
            indices.append(tmp)

        return np.ravel_multi_index(tuple(indices), self.nsplits, mode='raise', order='C')


class KMeansSubsampler(object):
    """
    Subsampler using k-means clustering to group particles into regions.

    This subsampler uses scikit-learn's KMeans algorithm to partition particles
    into ``nsplits`` clusters based on their spatial positions. Optionally supports
    Healpix pixelation for angular coordinates to reduce computation.

    Attributes
    ----------
    nsplits : int
        Number of clusters (subregions).
    kmeans : sklearn.cluster.KMeans
        Fitted KMeans clustering model.
    nside : int or None
        Healpix nside parameter (angular mode only).
    nest : bool
        Healpix pixel ordering (nested=True or ring=False).
    random_state : int or RandomState
        Random seed for KMeans initialization.
    """

    def __init__(self, particles, nsplits=8, nside=None, mode='cartesian',
                 random_state=None, wattrs: WeightAttrs = None, **kwargs):
        """
        Initialize :class:`KMeansSubsampler`.

        Parameters
        ----------
        particles : Particles
            Input particles with positions and optional weights.
        nsplits : int, default=8
            Number of clusters to create.
        nside : int, optional
            Healpix nside parameter for angular pixelation. Only valid with
            ``mode='angular'``. Smaller values reduce memory/runtime but coarsen
            angular resolution. If ``None``, no pixelation is performed.
        mode : {'cartesian', 'angular'}, default='cartesian'
            Coordinate system. 'cartesian' uses 3D Euclidean positions;
            'angular' expects unit vectors on a sphere.
        random_state : int or np.random.RandomState, optional
            Random seed for KMeans centroid initialization. Ensures reproducibility.
        wattrs : WeightAttrs, optional
            Weight attributes for particles. If ``None``, uniform weights are used.
        **kwargs
            Additional arguments passed to :class:`sklearn.cluster.KMeans`.
            Defaults: ``n_init=10``.

        Examples
        --------
        Create k-means subsampler with 8 clusters:

        >>> particles = Particles(positions)
        >>> subsampler = KMeansSubsampler(particles, nsplits=8, random_state=42)
        >>> labels = subsampler.label(particles)

        Angular mode with Healpix pixelation:

        >>> subsampler = KMeansSubsampler(particles, nsplits=16, mode='angular',
        ...                                nside=128, random_state=42)
        >>> labels = subsampler.label(particles)
        """
        mode = mode.lower()
        self.nsplits = int(nsplits)
        self.nside = nside

        if self.nside is not None and mode != 'angular':
            raise ValueError(f'Healpix (nside={self.nside}) can only be used with mode="angular"')

        self.nest = False
        self.random_state = random_state

        if wattrs is None:
            wattrs = WeightAttrs()

        weights = wattrs(particles) * np.ones(particles.size)
        kwargs.setdefault('n_init', 10)
        from sklearn import cluster

        if self.nside is not None:
            self.nside = int(self.nside)
            import healpy as hp

            # Convert Cartesian positions to Healpix pixels
            pix = hp.vec2pix(self.nside, *particles.positions.T, nest=self.nest)

            # Aggregate weights by pixel
            weights = np.bincount(pix, weights=weights, minlength=hp.nside2npix(self.nside))

            # Keep only non-empty pixels
            pix = np.flatnonzero(weights)
            weights = weights[pix]

            # Convert pixel indices back to 3D unit vectors
            positions = np.column_stack(hp.pix2vec(self.nside, pix, nest=self.nest))
        else:
            positions = particles.positions

        # Fit KMeans clustering
        self.kmeans = cluster.KMeans(n_clusters=self.nsplits,
                                     random_state=self.random_state, **kwargs)
        self.kmeans.fit(positions, sample_weight=weights)

    def label(self, positions):
        """
        Assign cluster labels to input positions.

        Positions are assigned to the nearest cluster center identified during fitting.
        For angular mode, positions are first pixelated at the same resolution.

        Parameters
        ----------
        positions : array of shape (n_particles, 3) or Particles
            Particle positions. If ``Particles`` object, extracts ``.positions`` attribute.
            For angular mode, should be unit vectors on a sphere.

        Returns
        -------
        labels : ndarray of shape (n_particles,)
            Cluster labels in range [0, nsplits).

        Examples
        --------
        >>> subsampler = KMeansSubsampler(particles, nsplits=8)
        >>> labels = subsampler.label(new_positions)
        >>> cluster_counts = np.bincount(labels)
        >>> print(f"Cluster sizes: {cluster_counts}")
        """
        positions = positions.positions if isinstance(positions, Particles) else positions

        if self.nside is not None:
            import healpy as hp
            # Pixelate angular positions before prediction
            pix = hp.vec2pix(self.nside, *positions.T, nest=self.nest)
            positions = np.column_stack(hp.pix2vec(self.nside, pix, nest=self.nest))

        return self.kmeans.predict(positions)