import numpy as np

from test_pp import generate_catalogs


def test_cucount():

    from cucount.numpy import count2, count3close, Particles, BinAttrs, SelectionAttrs, MeshAttrs, setup_logging
    import lsstypes as types

    boxsize = (3000.,) * 3

    # Cutsky geometry
    size = int(1e6)
    data, _ = generate_catalogs(size, boxsize, n_individual_weights=1, seed=42)
    data_positions, data_weights = np.column_stack(data[:3]), data[3:]
    data = Particles(data_positions, data_weights)
    battrs = BinAttrs(theta=np.linspace(0., 1, 100))
    sattrs = SelectionAttrs(theta=(0., 0.05))
    import time
    t0 = time.time()
    count2(data, data, battrs=battrs, sattrs=sattrs)
    print(f'count2 {time.time() - t0:.2f}')
    t0 = time.time()
    counts = count3close(data, data, data, battrs12=battrs, battrs13=battrs, sattrs12=sattrs, sattrs13=sattrs)['weight']
    print(counts)
    print(f'count3 {time.time() - t0:.2f}')


if __name__ == '__main__':

    test_cucount()