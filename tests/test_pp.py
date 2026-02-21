import numpy as np
from scipy import special

from cucount.numpy import count2, Particles, BinAttrs, SelectionAttrs, WeightAttrs, popcount, reformat_bitarrays, joint_occurences, setup_logging


@np.vectorize
def spherical_bessel(x, ell=0):

    ABS, SIN, COS = np.abs, np.sin, np.cos

    absx = ABS(x)
    threshold_even = 4e-2
    threshold_odd = 2e-2
    if (ell == 0):
        if (absx < threshold_even):
            x2 = x * x
            return 1 - x2 / 6 + (x2 * x2) / 120
        return SIN(x) / x
    if (ell == 2):
        x2 = x * x
        if (absx < threshold_even): return x2 / 15 - (x2 * x2) / 210
        return (3 / x2 - 1) * SIN(x) / x - 3 * COS(x) / x2
    if (ell == 4):
        x2 = x * x
        x4 = x2 * x2
        if (absx < threshold_even): return x4 / 945
        invx = 1 / x
        invx2 = invx**2
        invx3 = invx2 * invx
        invx4 = invx2 * invx2
        #return SIN(x) * (invx - 45.0 * invx3 + 105.0 * invx4) - COS(x) * (10.0 * invx - 105.0 * invx3);
        return 5 * (2 * invx2 - 21 * invx4) * COS(x) + (invx - 45 * invx3 + 105 * invx2 * invx3) * SIN(x);
        #return 5 * (2 * x2 - 21) * COS(x) / x4 + (x4 - 45 * x2 + 105) * SIN(x) / (x * x4)
    if (ell == 1):
        if (absx < threshold_odd): return x / 3 - x * x * x / 30
        return SIN(x) / (x * x) - COS(x) / x
    if (ell == 3):
        if (absx < threshold_odd): return x * x * x / 105
        x2 = x * x
        return (x2 - 15) * COS(x) / (x * x2) - 3 * (2 * x2 - 5) * SIN(x) / (x2 * x2)



def test_legendre_bessel():
    mu = np.linspace(-1., 1., 1000)
    x = np.geomspace(1e-9, 100, 1000)
    for ell in range(5):
        print(ell)

        assert np.allclose(spherical_bessel(x, ell), special.spherical_jn(ell, x, derivative=False), atol=1e-7, rtol=1e-3)


def generate_catalogs(size=100, boxsize=(1000,) * 3, offset=(1000., 0., 0.), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = []
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        weights += [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        #weights += [np.full(size, 0xffffffff, dtype=np.uint64) for i in range(n_bitwise_weights)]
        toret.append(positions + weights)
    return toret


def diff(position1, position2):
    return [p2 - p1 for p1, p2 in zip(position1, position2)]


def midpoint(position1, position2):
    return [p2 + p1 for p1, p2 in zip(position1, position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1 * x2 for x1, x2 in zip(position1, position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2) / (norm(position1) * norm(position2))


def wiip(weights, nrealizations=None, noffset=1, default=0.):
    denom = noffset + popcount(*weights)
    mask = denom == 0
    denom[mask] = 1.
    toret = nrealizations / denom
    toret[mask] = default
    return toret


def wpip_single(weights1, weights2, nrealizations=None, noffset=1, default=0., correction=None):
    denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1, weights2))
    if denom == 0:
        weight = default
    else:
        weight = nrealizations / denom
        if correction is not None:
            c = tuple(sum(bin(w).count('1') for w in weights) for weights in [weights1, weights2])
            weight /= correction[c]
    return weight


def wiip_single(weights, nrealizations=None, noffset=1, default=0.):
    denom = noffset + popcount(*weights)
    return default if denom == 0 else nrealizations / denom


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default=0., correction=None, weight_type='auto'):
    weight = 1
    if nrealizations is not None:
        weight *= wpip_single(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default=default, correction=correction)
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        costheta = sum(x1 * x2 for x1, x2 in zip(xyz1, xyz2)) / (norm(xyz1) * norm(xyz2))
        if (sep_twopoint_weights[0] <= costheta < sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='right', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta]) / (sep_twopoint_weights[ind_costheta + 1] - sep_twopoint_weights[ind_costheta])
            weight *= (1 - frac) * twopoint_weights[ind_costheta] + frac * twopoint_weights[ind_costheta + 1]
    if weight_type == 'inverse_bitwise_minus_individual':
        # print(1./nrealizations * weight, 1./nrealizations * wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default=default)\
        #          * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default=default))
        weight -= wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default=default)\
                  * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default=default)
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    return weight


def ref_theta_correlation(edges, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    counts = np.zeros(len(edges) - 1, dtype='f8')
    sep = np.zeros(len(edges) - 1, dtype='f8')
    poles = [np.zeros(len(edges) - 1, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    if theta_limits is not None:
        costheta_limits = np.cos(np.deg2rad(theta_limits)[::-1])
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
                #if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): costheta = 1.
                #else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
                #if costheta <= costheta_limits[0] or costheta > costheta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if dist < edges[0] or dist >= edges[-1]: continue
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            counts[ind] += weight
            sep[ind] += weight * dist
            for ill, ell in enumerate(ells):
                poles[ill][ind] += weight * (2 * ell + 1) * legendre[ill](mu)
    return np.asarray(poles), sep / counts


def ref_theta_spectrum(modes, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    poles = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            for ill, ell in enumerate(ells):
                poles[ill] += (-1j)**ell * weight * (2 * ell + 1) * legendre[ill](mu) * special.spherical_jn(ell, modes * dist)
    return np.asarray(poles)


def test_thetacut():
    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])

    edges = np.linspace(0., 30, 11)
    #edges = np.array([0.2, 0.3, 0.5, 1.])
    size = int(1e2)
    boxsize = (20,) * 3

    list_options = []
    list_options.append({'space': 'spectrum', 'ells': (0, 2)})
    list_options.append({'space': 'correlation', 'ells': (0, 2)})

    for options in list_options:
        options = options.copy()
        print(options)
        nthreads = options.pop('nthreads', None)
        weights_one = options.pop('weights_one', [])
        n_individual_weights = options.pop('n_individual_weights', 1)
        n_bitwise_weights = options.pop('n_bitwise_weights', 0)
        data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
        data1 = [np.concatenate([d, d]) for d in data1]  # that will get us some pairs at sep = 0
        rng = np.random.RandomState(seed=42)
        mask = rng.uniform(0., 1., size) > 0.1
        data2[3:] = [dd * mask for dd in data2[3:]]  # set some weights to zero
        selection_attrs = options.pop('selection_attrs', {'theta': (0., 0.05)})
        autocorr = options.pop('autocorr', False)
        options.setdefault('boxsize', None)
        options.setdefault('los', 'x' if options['boxsize'] is not None else 'firstpoint')
        bitwise_type = options.pop('bitwise_type', None)
        iip = options.pop('iip', False)
        dtype = options.pop('dtype', None)
        ells = options.get('ells', (0,))
        los = options.get('los')
        space = options.pop('space', 'correlation')

        ref_options = options.copy()
        weight_attrs = ref_options.pop('weight_attrs', {}).copy()

        def setdefaultnone(di, key, value):
            if di.get(key, None) is None:
                di[key] = value

        setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
        setdefaultnone(weight_attrs, 'noffset', 1)
        set_default = 'default' in weight_attrs
        if set_default:
            for w in data1[3:3 + n_bitwise_weights] + data2[3:3 + n_bitwise_weights]: w[:] = 0  # set to zero to make sure default is used
        setdefaultnone(weight_attrs, 'default', 0)
        data1_ref, data2_ref = data1.copy(), data2.copy()
        # data1_ref = [mpi.gather(d, mpiroot=None, mpicomm=mpicomm) for d in data1_ref]
        # data2_ref = [mpi.gather(d, mpiroot=None, mpicomm=mpicomm) for d in data2_ref]

        def dataiip(data):
            kwargs = {name: weight_attrs[name] for name in ['nrealizations', 'noffset', 'default']}
            return data[:3] + [wiip(data[3:3 + n_bitwise_weights], **kwargs)] + data[3 + n_bitwise_weights:]

        if n_bitwise_weights == 0:
            weight_attrs['nrealizations'] = None
        if iip:
            data1_ref = dataiip(data1_ref)
            data2_ref = dataiip(data2_ref)
        if iip == 1:
            data1 = dataiip(data1)
        elif iip == 2:
            data2 = dataiip(data2)
        if iip:
            n_bitwise_weights = 0
            weight_attrs['nrealizations'] = None

        if dtype is not None:
            for ii in range(len(data1_ref)):
                if np.issubdtype(data1_ref[ii].dtype, np.floating):
                    data1_ref[ii] = np.asarray(data1_ref[ii], dtype=dtype)
                    data2_ref[ii] = np.asarray(data2_ref[ii], dtype=dtype)

        twopoint_weights = ref_options.pop('twopoint_weights', None)
        if twopoint_weights is not None:
            twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))

        if n_bitwise_weights and weight_attrs.get('normalization', None) == 'counter':
            nalways = weight_attrs.get('nalways', 0)
            noffset = weight_attrs.get('noffset', 1)
            nrealizations = weight_attrs['nrealizations']
            joint = joint_occurences(nrealizations, noffset=weight_attrs['noffset'] + nalways, default=weight_attrs['default'])
            correction = np.zeros((nrealizations,) * 2, dtype='f8')
            for c1 in range(correction.shape[0]):
                for c2 in range(correction.shape[1]):
                    correction[c1][c2] = joint[c1 - nalways][c2 - nalways] if c2 <= c1 else joint[c2 - nalways][c1 - nalways]
                    correction[c1][c2] /= (nrealizations / (noffset + c1) * nrealizations / (noffset + c2))
            weight_attrs['correction'] = correction
        weight_attrs.pop('normalization', None)

        sep_ref = None
        if space == 'correlation':
            poles_ref, sep_ref = ref_theta_correlation(edges, data1_ref, data2=data2_ref if not autocorr else None, autocorr=autocorr, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, selection_attrs=selection_attrs, **ref_options, **weight_attrs)
        else:
            poles_ref = ref_theta_spectrum(edges, data1_ref, data2=data2_ref if not autocorr else None, autocorr=autocorr, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, selection_attrs=selection_attrs, **ref_options, **weight_attrs)

        itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
        tol = {'atol': 1e-8, 'rtol': 1e-2} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-5}

        if bitwise_type is not None and n_bitwise_weights > 0:

            def update_bit_type(data):
                return data[:3] + reformat_bitarrays(*data[3:3 + n_bitwise_weights], dtype=bitwise_type) + data[3 + n_bitwise_weights:]

            data1 = update_bit_type(data1)
            data2 = update_bit_type(data2)

        for label, catalog in zip([1, 2], [data1, data2]):
            if label in weights_one:
                catalog.append(np.ones_like(catalog[0]))

        def run(pass_none=False, pass_zero=False, **kwargs):
            positions1 = data1[:3]
            positions2 = data2[:3]
            weights1 = data1[3]
            weights2 = data2[3]

            def get_zero(arrays):
                if isinstance(arrays, list):
                    return [array[:0] if array is not None else None for array in arrays]
                elif arrays is not None:
                    return arrays[:0]
                return None

            if pass_zero:
                positions1 = get_zero(positions1)
                positions2 = get_zero(positions2)
                weights1 = get_zero(weights1)
                weights2 = get_zero(weights2)

            positions1 = np.column_stack(positions1)
            positions2 = np.column_stack(positions2)
            print('zero weight', np.sum(weights2 == 0.))
            particles1 = Particles(positions1, weights1)
            particles2 = Particles(positions2, weights2)
            if space == 'correlation':
                battrs = BinAttrs(**{'s': edges, 'pole': (np.array(ells), los)})
            else:
                battrs = BinAttrs(**{'k': edges, 'pole': (np.array(ells), los)})
            for var in selection_attrs:
                sattrs = SelectionAttrs(**{var: (selection_attrs[var][0], selection_attrs[var][1])})
            return count2(particles1, particles2, battrs=battrs, sattrs=sattrs)['weight']

        test = run()
        print(test)
        print(poles_ref.T)
        assert np.allclose(test.T, poles_ref, **tol)


def test_corrfunc_cutsky(mode='smu'):
    import time
    print(f'Test in mode {mode}')

    size = int(1e7)
    boxsize = (3000,) * 3
    sep = np.linspace(0., 0.1, 100)
    twopoint_weights = (sep, 1. + np.linspace(0., 1., sep.size))

    data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=1, n_bitwise_weights=2, seed=44)
    positions1, weights1 = np.column_stack(data1[:3]), data1[3:]
    positions2, weights2 = np.column_stack(data2[:3]), data2[3:]

    los = 'midpoint'
    if mode == 'smu':
        edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
        battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
        #assert np.allclose(battrs.edges('s'), np.column_stack([edges[0][:-1], edges[0][1:]]))
        #assert np.allclose(battrs.edges('mu'), np.column_stack([edges[1][:-1], edges[1][1:]]))
    elif mode == 'rppi':
        edges = (np.linspace(1., 51, 101), np.linspace(-50., 50., 101))
        battrs = BinAttrs(rp=(edges[0], los), pi=(edges[1], los))
    elif mode == 'theta':
        edges = 10**np.arange(-4, -1.5 + 0.1, 0.1)
        battrs = BinAttrs(theta=edges)
    else:
        raise NotImplementedError
    nthreads = 4

    def run_cucount(backend='numpy'):
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        if backend == 'jax':
            from cucount.jax import Particles, WeightAttrs, MeshAttrs, count2
            kw = dict()
        else:
            from cucount.numpy import Particles, WeightAttrs, MeshAttrs, count2
            kw = dict(nthreads=nthreads)
        t0 = time.time()
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        wattrs = WeightAttrs(bitwise=dict(weights=particles1.get('bitwise_weight')), angular=dict(sep=twopoint_weights[0], weight=twopoint_weights[1]))
        test = count2(particles1, particles2, battrs=battrs, wattrs=wattrs, **kw)['weight']
        print(f'cucount: {time.time() - t0:.2f} s')
        return test

    def run_corrfunc():
        from pycorr import TwoPointCounter
        t0 = time.time()
        kw = dict(nthreads=nthreads, gpu=True) if mode == 'smu' else dict(nthreads=8, gpu=False)
        ref = TwoPointCounter(mode, edges=edges, positions1=positions1, weights1=weights1, positions2=positions2, weights2=weights2, los=los, position_type='pos', weight_attrs={'normalization': 'counter'}, twopoint_weights=twopoint_weights, **kw).wcounts
        print(f'Corrfunc: {time.time() - t0:.2f} s')
        return ref

    tol = {'atol': 1e-8, 'rtol': 3e-5}
    if mode == 'theta': tol = {'atol': 1e-8, 'rtol': 1e-3} # there can be a jump from a bin from another
    #print(test.ravel())
    ref = run_corrfunc()
    test_jax = run_cucount(backend='jax')
    test_numpy = run_cucount(backend='numpy')
    assert np.allclose(test_jax, ref, **tol)
    assert np.allclose(test_numpy, ref, **tol)


def test_corrfunc_cubic(mode='smu'):
    import time
    print(f'Test in mode {mode}')

    size = int(1e7)
    boxsize = (3000,) * 3

    data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=1, n_bitwise_weights=0, seed=44)
    positions1, weights1 = np.column_stack(data1[:3]), data1[3:]
    positions2, weights2 = np.column_stack(data2[:3]), data2[3:]

    los = 'z'
    if mode == 'smu':
        edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
        battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
        #assert np.allclose(battrs.edges('s'), np.column_stack([edges[0][:-1], edges[0][1:]]))
        #assert np.allclose(battrs.edges('mu'), np.column_stack([edges[1][:-1], edges[1][1:]]))
    elif mode == 'rppi':
        edges = (np.linspace(1., 51, 101), np.linspace(-100., 100., 101))
        battrs = BinAttrs(rp=(edges[0], los), pi=(edges[1], los))
    else:
        raise NotImplementedError
    nthreads = 4

    def run_cucount(backend='numpy', nthreads=nthreads):
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        if backend == 'jax':
            from cucount.jax import Particles, WeightAttrs, MeshAttrs, count2
            kw = dict()
        else:
            from cucount.numpy import Particles, WeightAttrs, MeshAttrs, count2
            kw = dict(nthreads=nthreads)
        t0 = time.time()
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        mattrs = MeshAttrs(particles1, particles2, boxsize=boxsize, battrs=battrs, periodic=True)
        print(mattrs)
        test = count2(particles1, particles2, battrs=battrs, mattrs=mattrs, **kw)['weight']
        print(f'cucount: {time.time() - t0:.2f} s')
        return test

    def run_corrfunc():
        from pycorr import TwoPointCounter
        t0 = time.time()
        kw = dict(nthreads=nthreads, gpu=True) if mode == 'smu' else dict(nthreads=8, gpu=False)
        ref = TwoPointCounter(mode, edges=edges, positions1=positions1, weights1=weights1, positions2=positions2, weights2=weights2, los=los, position_type='pos', boxsize=boxsize, **kw).wcounts
        print(f'Corrfunc: {time.time() - t0:.2f} s')
        return ref

    tol = {'atol': 1e-8, 'rtol': 3e-5}  # there can be a jump from a bin from another
    #print(test.ravel())
    ref = run_corrfunc()
    test_numpy = run_cucount(backend='numpy')
    assert np.allclose(test_numpy, ref, **tol)
    test_jax = run_cucount(backend='jax')
    assert np.allclose(test_jax, ref, **tol)


def test_spectrum(backend='numpy'):
    if backend == 'jax':
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        from cucount.jax import count2, Particles, BinAttrs, SelectionAttrs
    else:
        from cucount.numpy import count2, Particles, BinAttrs, SelectionAttrs
    size = int(1e7)
    boxsize = (3000,) * 3

    sattrs = SelectionAttrs(theta=(0., 0.05))
    ells = (0, 2, 4)

    data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=1, seed=44)
    positions1, weights1 = np.column_stack(data1[:3]), data1[3:]
    positions2, weights2 = np.column_stack(data2[:3]), data2[3:]
    particles1 = Particles(positions1, weights1)
    particles2 = Particles(positions2, weights2)

    k = np.linspace(0.001, 0.1, 10)
    battrs = BinAttrs(k=k, pole=(ells, 'firstpoint'))
    counts_k = count2(particles1, particles2, battrs=battrs, sattrs=sattrs)['weight'].T

    sedges = np.arange(0., sum(b**2 for b in boxsize)**0.5, 0.1)
    battrs = BinAttrs(s=sedges, pole=(ells, 'firstpoint'))
    counts_s = count2(particles1, particles2, battrs=battrs, sattrs=sattrs)['weight'].T

    def correlation_to_spectrum(k, sedges, counts, ells=ells):
        from scipy import special
        sedges = np.column_stack([sedges[:-1], sedges[1:]])
        smid = np.mean(sedges, axis=-1)
        #volume = 4. * np.pi / 3. * (sedges[..., 1]**3 - sedges[..., 0]**3)
        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append((-1)**(ell // 2) * np.sum(counts[ill] * special.spherical_jn(ell, k[:, None] * smid[None, :]), axis=-1))
        return spectrum

    spectrum = correlation_to_spectrum(k, sedges, counts_s, ells=ells)
    from matplotlib import pyplot as plt
    ax = plt.gca()
    for ill, ell in enumerate(ells):
        color = f'C{ill:d}'
        ax.plot(k, counts_k[ill], color=color, linestyle='-')
        ax.plot(k, spectrum[ill], color='k', linestyle='--')
    plt.show()


def test_jax(distributed=False):
    los = 'midpoint'
    edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
    size = int(1e6)
    boxsize = (3000,) * 3
    data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=1, n_bitwise_weights=0, seed=42)
    positions1, weights1 = np.column_stack(data1[:3]), data1[3]
    positions2, weights2 = np.column_stack(data2[:3]), data2[3]

    def count_numpy():
        from cucount.numpy import count2, Particles, BinAttrs, SelectionAttrs
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
        return count2(particles1, particles2, battrs=battrs)['weight']

    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    res_numpy = count_numpy()
    if distributed: jax.distributed.initialize()

    def count_jax_manual():
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, PartitionSpec as P

        from cucount.jax import count2, Particles, BinAttrs, SelectionAttrs, MeshAttrs, setup_logging
        #setup_logging("error")
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
        mattrs = MeshAttrs(particles1, particles2, battrs=battrs)
        count = lambda *particles: count2(particles, battrs=battrs, mattrs=mattrs)
        if distributed:
            sharding_mesh = Mesh(jax.devices(), ('x',))
            count = shard_map(lambda *particles: jax.lax.psum(count2(*particles, battrs=battrs, mattrs=mattrs), sharding_mesh.axis_names), mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names), P(None)), out_specs=P(None))
        toret = count(particles1, particles2)
        return toret['weight']

    def count_jax():
        from cucount.jax import count2, Particles, BinAttrs, SelectionAttrs, create_sharding_mesh, setup_logging
        #setup_logging("error")
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
        with create_sharding_mesh():
            toret = count2(particles1, particles2, battrs=battrs)
        return toret['weight']

    res_jax = count_jax()
    res_jax_manual = count_jax_manual()
    if distributed: jax.distributed.shutdown()
    assert np.allclose(res_jax, res_jax_manual)
    assert np.allclose(res_jax, res_numpy)


def test_readme():
    import numpy as np
    from cucount.numpy import count2, Particles, BinAttrs, setup_logging

    setup_logging("info")
    # Prepare catalogs
    size = int(1e5)
    boxsize = np.array((3000.,) * 3)
    rng = np.random.RandomState(seed=42)

    def generate_catalog(rng, size):
        offset = boxsize
        positions = rng.uniform(0., 1., (size, 3)) * boxsize + offset
        weights = rng.uniform(0., 1., size)
        return positions, weights

    positions1, weights1 = generate_catalog(rng, size)
    positions2, weights2 = generate_catalog(rng, size)

    # Define binning and line-of-sight
    edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
    los = 'midpoint'

    # Compute pair counts
    particles1 = Particles(positions1, weights1)
    particles2 = Particles(positions2, weights2)
    battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
    counts = count2(particles1, particles2, battrs=battrs)['weight']
    print(counts.sum(axis=-1))


def test_readme2():
    # Prepare catalogs
    size = int(1e5)
    boxsize = np.array((3000.,) * 3)
    rng = np.random.RandomState(seed=42)

    def generate_catalog(rng, size):
        offset = boxsize
        positions = rng.uniform(0., 1., (size, 3)) * boxsize + offset
        weights = rng.uniform(0., 1., size)
        return positions, weights

    positions1, weights1 = generate_catalog(rng, size)
    positions2, weights2 = generate_catalog(rng, size)
    edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
    los = 'midpoint'

    import jax
    jax.config.update("jax_enable_x64", True)
    # Initialize distributed environment (if needed)
    jax.distributed.initialize()
    from cucount.jax import count2, Particles, BinAttrs, create_sharding_mesh

    battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

    # Run distributed pair counts
    with create_sharding_mesh():
        # Pass exchange=True if input is distributed over multiple processes
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        counts = count2(particles1, particles2, battrs=battrs)


def test_popcount():
    # Create a lookup table for set bits per byte
    _popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

    def popcount_ref(*arrays):
        """
        Return number of 1 bits in each value of input array.
        Inspired from https://github.com/numpy/numpy/issues/16325.
        """
        # if not np.issubdtype(array.dtype, np.unsignedinteger):
        #     raise ValueError('input array must be an unsigned int dtype')
        toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
        for array in arrays[1:]: toret += popcount(array)
        return toret

    from jax import numpy as jnp

    def popcount(*arrays):
        return sum(jnp.bitwise_count(array) for array in arrays)

    rng = np.random.RandomState(seed=42)
    size = 100
    arrays = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(6)]
    count_ref = popcount_ref(*arrays)
    count = popcount(*arrays)
    assert np.allclose(count, count_ref)


def test_analytic():
    boxsize = 2000.
    from cucount.numpy import count2_analytic, BinAttrs, MeshAttrs

    for battrs in [BinAttrs(s=np.linspace(0., 200., 21)),
                   BinAttrs(s=np.linspace(0., 200., 21), mu=(np.linspace(-1., 1., 21), 'x')),
                   BinAttrs(rp=(np.linspace(0., 200., 21), 'x'), pi=(np.linspace(-1., 1., 21), 'x')),
                   BinAttrs(rp=(np.linspace(0., 200., 21), 'x'))]:
        counts = count2_analytic(battrs=battrs, mattrs=boxsize)
        assert counts.shape == battrs.shape


def test_lsstypes():

    from pathlib import Path
    from cucount.numpy import count2, count2_analytic, Particles, BinAttrs, WeightAttrs, MeshAttrs, setup_logging
    import lsstypes as types
    from lsstypes import Count2, Count2Correlation

    boxsize = (3000.,) * 3

    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, n_bitwise_weights=2, seed=42)
    randoms, _ = generate_catalogs(2 * size, boxsize=boxsize, n_individual_weights=1, seed=84)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]
    random_positions, random_weights = np.column_stack(randoms[:3]), randoms[3:]
    data = Particles(data_positions, data_weights)
    randoms = Particles(random_positions, random_weights)

    battrs = BinAttrs(s=np.linspace(0., 200., 21), mu=(np.linspace(-1., 1., 21), 'midpoint'))
    wattrs = WeightAttrs(bitwise=dict(weights=data.get('bitwise_weight')))
    mattrs = None

    # Helper to convert to lsstypes Count2
    def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray) -> Count2:
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return Count2(counts=counts, norm=norm, **coords, **edges, coords=list(coords))

    # Hepler to get counts as Count2
    def get_counts(*particles: Particles) -> Count2:
        autocorr = len(particles) == 1
        counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
        if autocorr:
            norm = wattrs(particles[0]).sum()**2 - wattrs(*(particles * 2)).sum()
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        return to_lsstypes(battrs, counts, norm)

    DD = get_counts(data)
    DR = get_counts(data.clone(weights=wattrs(data)), randoms)
    RD = DR.clone(value=DR.value()[:, ::-1])  # reverse mu for RD
    RR = get_counts(randoms)
    # Note: you can also "sum" DR, RD and RR counts over multiple random catalogs to reduce noise
    # DR = types.sum(list_of_DR_counts)
    # RR = types.sum(list_of_RR_counts)

    # For reconstructed 2PCF, you can provide DD, DS, SD, SS, RR counts
    correlation = Count2Correlation(estimator='landyszalay', DD=DD, DR=DR, RD=RD, RR=RR)
    dirname = Path('_tests')
    fn = dirname / 'test_lsstypes_landyszalay.h5'
    dirname.mkdir(exist_ok=True)
    correlation.write(fn)
    correlation = types.read(fn)
    correlation.project(ells=[0, 2, 4]).plot(fn=dirname / 'test_lsstypes_landyszalay.png')

    # Periodic box geometry
    data, _ = generate_catalogs(1000, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]
    data = Particles(data_positions, data_weights)

    battrs = BinAttrs(s=np.linspace(0., 200., 21), mu=(np.linspace(-1., 1., 21), 'midpoint'))
    wattrs = WeightAttrs()
    mattrs = MeshAttrs(data, boxsize=boxsize, battrs=battrs, periodic=True)

    DD = get_counts(data)
    # In case of periodic geometry; analytic RR
    RR = to_lsstypes(battrs, count2_analytic(battrs=battrs, mattrs=boxsize), norm=1.)
    correlation = Count2Correlation(estimator='natural', DD=DD, RR=RR)
    fn = dirname / 'test_lsstypes_natural.h5'
    correlation.write(fn)
    correlation = types.read(fn)
    correlation.project(ells=[0, 2, 4]).plot(fn=dirname / 'test_lsstypes_natural.png')


if __name__ == '__main__':

    setup_logging()


    test_analytic()
    test_lsstypes()
    #test_thetacut()
    for mode in ['smu', 'rppi', 'theta']:
        test_corrfunc_cutsky(mode)
    for mode in ['smu', 'rppi']:
        test_corrfunc_cubic(mode)
    #test_spectrum()
    #test_jax(distributed=True)
    #test_readme()
    #test_readme2()
    #test_popcount()