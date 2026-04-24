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
    print(results23.sum(axis=-1))
    print(np.swapaxes(results12, 2, 0).sum(axis=-1))
    assert np.allclose(results23, np.swapaxes(results12, 2, 0))


def test_triposh():

    import numpy as np
    from cucount.numpy import Particles, count3close, BinAttrs, SelectionAttrs

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
    
    
    def pbar_lmax4(ell, mu):
        """Same normalized associated Legendre rows as CUDA compute_pbar_row_lmax4."""
        x = np.clip(mu, -1., 1.)
        s2 = max(0., 1. - x * x)
        s = np.sqrt(s2)
    
        if ell == 0:
            return np.array([1.])
        if ell == 1:
            return np.array([x, -np.sqrt(1/2) * s])
        if ell == 2:
            return np.array([
                0.5 * (3*x*x - 1),
                -np.sqrt(3/2) * x * s,
                np.sqrt(3/8) * s2,
            ])
        if ell == 3:
            return np.array([
                0.5 * (5*x**3 - 3*x),
                -np.sqrt(3/16) * (5*x*x - 1) * s,
                np.sqrt(15/8) * x * s2,
                -np.sqrt(5/16) * s2 * s,
            ])
        if ell == 4:
            return np.array([
                (35*x**4 - 30*x*x + 3) / 8,
                -np.sqrt(5/16) * x * (7*x*x - 3) * s,
                np.sqrt(5/128) * (7*x*x - 1) * s2,
                -np.sqrt(35/40) * x * s2 * s,
                np.sqrt(35/128) * s2 * s2,
            ])
        raise ValueError("This reference only supports ell <= 4")
    
    
    def proj_labels(ells1, ells2):
        labels = []
        for ell1 in ells1:
            for ell2 in ells2:
                mmax = min(ell1, ell2)
                for m in range(mmax + 1):
                    labels.append((ell1, ell2, m, "re"))
                for m in range(1, mmax + 1):
                    labels.append((ell1, ell2, m, "im"))
        return labels
    
    
    def brute_count3close_firstpoint(
        positions,
        weights,
        sedges,
        ells=(0, 2),
        theta_max=0.05,
    ):
        """
        Brute-force reference for:
          battrs12 = battrs13 = BinAttrs(s=sedges, pole=(ells, 'firstpoint'))
          sattrs12 = sattrs13 = SelectionAttrs(theta=(0, theta_max))
          no battrs23, no sattrs23.
    
        Mirrors the CUDA storage convention:
          for each ell1, ell2:
            Re m=0..mmax, then Im m=1..mmax.
        """
        pos = np.asarray(positions, dtype="f8")
        w = np.asarray(weights, dtype="f8")
        if w.ndim == 2:
            w = w[:, 0]
    
        n = len(pos)
        spos = normalize(pos)
        nbins = len(sedges) - 1
        labels = proj_labels(ells, ells)
        out = np.zeros((nbins, nbins, len(labels)), dtype="f8")
    
        cos_theta_min = np.cos(theta_max)
    
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
    
                rhat12 = r12_vec / r12 if r12 > 0 else np.zeros(3)
                mu12 = np.clip(np.dot(rhat12, ez), -1., 1.)
                x12 = np.dot(rhat12, ex)
                y12 = np.dot(rhat12, ey)
                rho12 = np.sqrt(max(0., 1. - mu12 * mu12))
    
                p1 = {ell: pbar_lmax4(ell, mu12) for ell in ells}
    
                for k in range(n):
                    if np.dot(spos[i], spos[k]) < cos_theta_min:
                        continue
    
                    r13_vec = pos[k] - pos[i]
                    r13 = np.linalg.norm(r13_vec)
                    ib13 = np.searchsorted(sedges, r13, side="right") - 1
                    if ib13 < 0 or ib13 >= nbins:
                        continue
    
                    rhat13 = r13_vec / r13 if r13 > 0 else np.zeros(3)
                    mu13 = np.clip(np.dot(rhat13, ez), -1., 1.)
                    x13 = np.dot(rhat13, ex)
                    y13 = np.dot(rhat13, ey)
                    rho13 = np.sqrt(max(0., 1. - mu13 * mu13))
    
                    cdphi, sdphi = 1., 0.
                    if rho12 > 1e-12 and rho13 > 1e-12:
                        inv = 1. / (rho12 * rho13)
                        cdphi = np.clip((x12 * x13 + y12 * y13) * inv, -1., 1.)
                        sdphi = np.clip((x12 * y13 - y12 * x13) * inv, -1., 1.)
    
                    p2 = {ell: pbar_lmax4(ell, mu13) for ell in ells}
                    tw = w[i] * w[j] * w[k]
    
                    ip = 0
                    for ell1 in ells:
                        for ell2 in ells:
                            mmax = min(ell1, ell2)
                            for m in range(mmax + 1):
                                amp = tw * p1[ell1][m] * p2[ell2][m]
                                out[ib12, ib13, ip + m] += amp * np.cos(m * np.arctan2(sdphi, cdphi))
                                if m > 0:
                                    out[ib12, ib13, ip + mmax + m] += amp * np.sin(m * np.arctan2(sdphi, cdphi))
                            ip += 2 * mmax + 1
    
        return out, labels
    
    rng = np.random.default_rng(42)
    # Keep this small: brute force is O(N^3).
    n = 200
    boxsize = (3000.,) * 3
    data, _ = generate_catalogs(n, boxsize, n_individual_weights=1, seed=42)
    positions = np.column_stack(data[:3])
    weights = np.asarray(data[3:]).T
    
    sedges = np.linspace(0., 10., 101)
    ells = np.array([0, 2])
    particles = Particles(positions=positions, weights=weights)
    battrs = BinAttrs(s=sedges, pole=(ells, "firstpoint"))
    sattrs = SelectionAttrs(theta=(0., 0.05))
    
    cuda = count3close(particles, particles, particles, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)["weight"]
    
    ref, labels = brute_count3close_firstpoint(
        positions,
        weights,
        sedges,
        ells=tuple(ells),
        theta_max=0.05,
    )
    
    cuda = np.asarray(cuda)
    print("CUDA shape:", cuda.shape)
    print("REF  shape:", ref.shape)
    print("projection labels:", labels)
    absdiff = np.abs(cuda - ref)
    reldiff = absdiff / np.maximum(1., np.abs(ref))
    print("max abs diff:", absdiff.max())
    print("max rel diff:", reldiff.max())
    idx = np.unravel_index(np.argmax(absdiff), absdiff.shape)
    print("worst index:", idx)
    print("label:", labels[idx[-1]])
    print("cuda:", cuda[idx])
    print("ref :", ref[idx])
    np.testing.assert_allclose(cuda, ref, rtol=5e-5, atol=5e-5)


if __name__ == '__main__':

    #test_cucount()
    #test_jax()
    test_symmetry()