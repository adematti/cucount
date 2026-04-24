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
            battrs = BinAttrs(s=np.linspace(0., 100, 101), pole=(0, 2))
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
            battrs = BinAttrs(s=np.linspace(0., 100, 101), pole=(0, 2))
        sattrs = SelectionAttrs(theta=(0., 0.05))
        with create_sharding_mesh():
            toret = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)['weight']
        return toret

    counts_numpy = test_numpy()
    counts_jax = test_jax()
    assert np.allclose(counts_jax, counts_numpy)


if __name__ == '__main__':

    #test_cucount()
    test_jax()