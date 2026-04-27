import numpy as np

from test_pp import generate_catalogs
import time


def test_cucount():

    from cucount.numpy import count2, count3close, Particles, BinAttrs, SelectionAttrs, setup_logging

    boxsize = (3000.,) * 3
    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]
    data = Particles(data_positions, data_weights)
    battrs = BinAttrs(theta=np.linspace(0., 1, 100))
    sattrs = SelectionAttrs(theta=(0., 0.05))
    t0 = time.time()
    count2(data, data, battrs=battrs, sattrs=sattrs)
    print(f'count2 {time.time() - t0:.2f}')
    t0 = time.time()
    counts = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)['weight']
    print(f'count3 {time.time() - t0:.2f}')


def test_jax():

    boxsize = (3000.,) * 3
    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]

    def test_numpy(binning='theta'):
        from cucount.numpy import Particles, count3close, BinAttrs, SelectionAttrs
        particles = Particles(positions=data_positions, weights=data_weights)
        data = Particles(data_positions, data_weights)
        if binning == 'theta':
            battrs = BinAttrs(theta=np.linspace(0., 1, 100))
        else:
            battrs = BinAttrs(s=np.linspace(0., 10, 101), pole=(np.array([0, 2]), 'firstpoint'))
        sattrs = SelectionAttrs(theta=(0., 0.05))
        toret = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)['weight']
        return toret

    def test_jax(binning='theta'):
        import jax
        jax.config.update("jax_enable_x64", True)
        from cucount.jax import Particles, count3close, BinAttrs, SelectionAttrs, create_sharding_mesh
        data = Particles(data_positions, data_weights)
        if binning == 'theta':
            battrs = BinAttrs(theta=np.linspace(0., 1, 100))
        else:
            battrs = BinAttrs(s=np.linspace(0., 10, 101), pole=(np.array([0, 2]), 'firstpoint'))
        sattrs = SelectionAttrs(theta=(0., 0.05))
        with create_sharding_mesh():
            toret = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)['weight']
        return toret

    for binning in ['theta', 's']:
        counts_numpy = test_numpy(binning=binning)
        counts_jax = test_jax(binning=binning)
        assert np.allclose(counts_jax, counts_numpy)


def test_symmetry():

    boxsize = (3000.,) * 3
    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]

    def count_close_pair(close_pair=(1, 2)):
        from cucount.numpy import Particles, count3close, BinAttrs, SelectionAttrs
        particles = Particles(positions=data_positions, weights=data_weights)
        data = Particles(data_positions, data_weights)
        battrs = BinAttrs(theta=np.linspace(0., 0.5, 5))
        sattrs = SelectionAttrs(theta=(0., 0.05))
        kw = dict()
        kw[f'sattrs{close_pair[0]:d}{close_pair[1]:d}'] = sattrs
        return count3close(data, data, data, close_pair=close_pair, battrs12=battrs, battrs13=battrs, battrs23=battrs, **kw)['weight']

    results12 = count_close_pair(close_pair=(1, 2))
    results13 = count_close_pair(close_pair=(1, 3))
    results23 = count_close_pair(close_pair=(2, 3))
    assert np.allclose(results13, np.swapaxes(results12, 1, 0))
    assert np.allclose(results23, np.swapaxes(results12, 2, 0))


def test_triposh():
    from scipy import special
    from cucount.numpy import Particles, count3close, BinAttrs, SelectionAttrs, triposh_to_poles, triposh_transform_matrix

    def normalize(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.divide(x, n, out=np.zeros_like(x), where=n > 0)

    def build_local_frame(ez):
        ez = ez / np.linalg.norm(ez)
        ref = np.array([0., 0., 1.]) if abs(ez[2]) < 0.9 else np.array([1., 0., 0.])
        ex = ref - np.dot(ref, ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        return ez, ex, ey

    def proj_labels(ells1, ells2):
        labels = []
        for ell1 in ells1:
            for ell2 in ells2:
                ell1, ell2 = int(ell1), int(ell2)
                mmax = min(ell1, ell2)
                for m in range(mmax + 1):
                    labels.append((ell1, ell2, m, "re"))
                for m in range(1, mmax + 1):
                    labels.append((ell1, ell2, m, "im"))
        return labels

    def brute_count3close_firstpoint_with_zero_r_zaxis(
        positions,
        weights,
        sedges,
        ells1=(0, 2),
        ells2=(0, 2),
        theta_max=0.05,
    ):
        pos = np.asarray(positions, dtype="f8")
        w = np.asarray(weights, dtype="f8")
        if w.ndim == 2:
            w = w[:, 0]

        ells1 = tuple(map(int, ells1))
        ells2 = tuple(map(int, ells2))

        n = len(pos)
        spos = normalize(pos)

        nbins = len(sedges) - 1
        labels = proj_labels(ells1, ells2)
        out = np.zeros((nbins, nbins, len(labels)), dtype="f8")

        cos_theta_min = np.cos(theta_max)

        def Y(ell, mm, xhat):
            mu = xhat[2]
            phi = np.arctan2(xhat[1], xhat[0])
            fac = special.factorial(ell - abs(mm), exact=False) / special.factorial(ell + abs(mm), exact=False)
            amp = np.sqrt((2 * ell + 1) / (4. * np.pi)) * np.sqrt(fac)
            return amp * special.lpmv(abs(mm), ell, mu) * np.exp(1j * mm * phi)

        for i in range(n):
            ez, ex, ey = build_local_frame(spos[i])

            for j in range(n):
                if np.dot(spos[i], spos[j]) < cos_theta_min:
                    continue

                r12_vec = pos[j] - pos[i]
                r12 = np.linalg.norm(r12_vec)

                ib12 = np.searchsorted(sedges, r12, side="right") - 1
                if ib12 < 0 or ib12 >= nbins:
                    continue

                if r12 == 0.:
                    rhat12_local = np.array([0., 0., 1.])
                else:
                    rhat12 = r12_vec / r12
                    rhat12_local = np.array([
                        np.dot(rhat12, ex),
                        np.dot(rhat12, ey),
                        np.dot(rhat12, ez),
                    ])

                for k in range(n):
                    if np.dot(spos[i], spos[k]) < cos_theta_min:
                        continue

                    r13_vec = pos[k] - pos[i]
                    r13 = np.linalg.norm(r13_vec)

                    ib13 = np.searchsorted(sedges, r13, side="right") - 1
                    if ib13 < 0 or ib13 >= nbins:
                        continue

                    if r13 == 0.:
                        rhat13_local = np.array([0., 0., 1.])
                    else:
                        rhat13 = r13_vec / r13
                        rhat13_local = np.array([
                            np.dot(rhat13, ex),
                            np.dot(rhat13, ey),
                            np.dot(rhat13, ez),
                        ])

                    tw = w[i] * w[j] * w[k]

                    ip = 0
                    for ell1 in ells1:
                        for ell2 in ells2:
                            mmax = min(ell1, ell2)

                            for m in range(mmax + 1):
                                val = Y(ell1, m, rhat12_local).conjugate() * Y(ell2, m, rhat13_local)
                                out[ib12, ib13, ip + m] += tw * val.real

                                if m > 0:
                                    out[ib12, ib13, ip + mmax + m] += tw * val.imag

                            ip += 2 * mmax + 1
        return out, labels

    boxsize = (3000.,) * 3
    size = 200

    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    positions = np.column_stack(data[:3])
    weights = np.asarray(data[3])
    sedges = np.linspace(0., 10., 101)
    theta_max = 0.05

    particles = Particles(positions=positions, weights=weights)
    sattrs = SelectionAttrs(theta=(0., theta_max))

    triposh_ells = [(0, 0, 0), (2, 0, 2)]
    ells1, ells2 = triposh_to_poles(triposh_ells)
    battrs12 = BinAttrs(s=sedges, pole=(ells1, "firstpoint"))
    battrs13 = BinAttrs(s=sedges, pole=(ells2, "firstpoint"))
    out_ells, matrix = triposh_transform_matrix(battrs12, battrs13, ells=triposh_ells)

    cuda = count3close(particles, particles, particles, battrs12=battrs12, battrs13=battrs13, sattrs12=sattrs, sattrs13=sattrs)["weight"]
    cuda.dot(matrix.T)

    ref, labels = brute_count3close_firstpoint_with_zero_r_zaxis(positions, weights, sedges, ells1=tuple(ells1), ells2=tuple(ells2), theta_max=theta_max)
    assert np.allclose(cuda, ref, rtol=5e-5, atol=5e-5)


def test_jax():

    boxsize = (2000.,) * 3
    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]

    def test_jax(binning='s'):
        import jax
        jax.config.update("jax_enable_x64", True)
        jax.distributed.initialize()
        from cucount.jax import Particles, count3close, BinAttrs, SelectionAttrs, create_sharding_mesh
        if binning == 'theta':
            battrs = BinAttrs(theta=np.linspace(0., 1, 100))
        else:
            battrs = BinAttrs(s=np.linspace(0., 10, 101), pole=(np.array([0, 2]), 'firstpoint'))
        sattrs = SelectionAttrs(theta=(0., 0.05))
        with create_sharding_mesh():
            data = Particles(data_positions, data_weights, exchange=True)
            toret = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)#['weight']
        return toret

    for binning in ['s']:
        counts_jax = test_jax(binning=binning)



if __name__ == '__main__':

    #test_cucount()
    #test_jax()
    #test_symmetry()
    test_triposh()
    #import os
    #os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    #test_jax()