# cucount: CUDA-Powered Pair Counts

**cucount** is a high-performance CUDA implementation for computing pair counts (positions - spins), and triplet counts, optimized for GPUs. It provides both NumPy and JAX interfaces depending on your workflow.

> ⚠️ A CUDA-capable GPU is required.

---

## 📦 Installation

You can install `cucount` directly via pip:

```bash
pip install git+https://github.com/adematti/cucount.git
```

The JAX API (through FFI) will be built automatically if the `jax.ffi` library is found during installation.

---

## 🧮 NumPy API

Use the NumPy API if you're not using JAX. All data stays on the **host (CPU)** and is internally transferred to the GPU.

### Example

```python
import numpy as np
from cucount.numpy import count2, Particles, BinAttrs

# Prepare catalogs
size = int(1e6)
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

# Compute pair counts, with 4 threads (i.e. on 4 GPU)
# If you want to go multi-node, MPI is a good option
particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
counts = count2(particles1, particles2, battrs=battrs, nthreads=4)
# counts is a dictionary with key "weight"
```

---

## ⚡ JAX API

Use the JAX interface if JAX is already part of your codebase.

### Why a separate JAX API?

1. **JAX preallocates GPU memory**, which can cause `cudaMalloc` to fail if using the NumPy backend.
2. **Passing device arrays** (from JAX) directly avoids host-device transfers.
3. **JAX's distributed capabilities** (e.g., `shard_map`) are well-suited for scaling.

---

### 🚀 Single-Device JAX Example

```python
import jax
from jax import config
config.update("jax_enable_x64", True)  # Currently only double precision is supported

from cucount.jax import count2, Particles, BinAttrs

# Assume positions1, positions2, weights1, weights2 are already defined
edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
los = 'midpoint'
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)

counts = count2(particles1, particles2, battrs=battrs)
# counts is a dictionary with key "weight"
```

---

### 🧩 Multi-Device (Distributed) JAX Example

Using [shard\_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), you can parallelize over multiple devices:

```python
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
```

---

## Angular Upweights

Angular (PIP) upweights are passed through `WeightAttrs(angular=...)`, as a function of the angular
separation θ **in degrees**. Tabulate it either at sample points (`sep`, linearly interpolated,
1 outside the range) or in bins (`edges`, piecewise-constant, 1 outside the range).
`weight` may be N-dimensional, in which case provide one `sep`/`edges` array per dimension.

```python
import numpy as np
from cucount.numpy import count2, Particles, BinAttrs, WeightAttrs

# Angular upweights, tabulated at sep (in degrees)
sep = np.linspace(0., 0.1, 100)
weight = 1. + np.linspace(0., 1., sep.size)

battrs = BinAttrs(s=np.linspace(1., 201., 201), mu=(np.linspace(-1., 1., 201), 'midpoint'))
wattrs = WeightAttrs(angular=dict(sep=sep, weight=weight))

particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
counts = count2(particles1, particles2, battrs=battrs, wattrs=wattrs)
```

Angular upweights typically come with bitwise (PIP) weights: pass integer arrays as weights to
`Particles` (they are automatically recognized as bitwise weights) and add `bitwise` to `WeightAttrs`.

```python
# weights1 = [individual_weights, bitwise_weight_0, bitwise_weight_1] (integer dtype => bitwise)
particles1 = Particles(positions1, weights1)
wattrs = WeightAttrs(bitwise=dict(weights=particles1.get('bitwise_weight')),
                     angular=dict(sep=sep, weight=weight))
counts = count2(particles1, particles2, battrs=battrs, wattrs=wattrs)
```

Same syntax with `cucount.jax`.

---

## Triplet Counts

`count3` computes *factorized* triplet counts: for each particle of catalog 1, catalog 2 is binned as
a function of the (1, 2) separation and catalog 3 as a function of the (1, 3) separation, accumulating
`w1 * w2(r12) * w3(r13)`. There is no binning nor selection on the (2, 3) separation.
`count3close` is the *close*-triplet version, which additionally accepts `battrs23` and (2, 3) selections.

```python
import numpy as np
from cucount.numpy import count3, count3close, Particles, BinAttrs, SelectionAttrs

particles = Particles(positions, weights)

sedges = np.linspace(0., 40., 11)
battrs12 = BinAttrs(s=sedges, pole=([0], 'firstpoint'))
battrs13 = BinAttrs(s=sedges, pole=([0, 2], 'firstpoint'))
# Restrict to close triplets: angular separation < 20 deg
sattrs = SelectionAttrs(theta=(0., 20.))

counts = count3(particles, particles, particles,
                battrs12=battrs12, battrs13=battrs13,
                sattrs12=sattrs, sattrs13=sattrs)
# counts is a dictionary with key "weight"

# Explicit triplets, optionally with binning in the (2, 3) separation
counts = count3close(particles, particles, particles,
                     battrs12=battrs12, battrs13=battrs13,
                     sattrs12=sattrs, sattrs13=sattrs)
```

For periodic boxes, `count3_analytic(battrs12, battrs13, mattrs)` gives the analytic (random) counts.
As above, `cucount.jax` provides the same functions for device arrays.

---

## 📦 lsstypes Wrapper

`cucount.types` wraps the low-level counters and returns
[lsstypes](https://github.com/cosmodesi/lsstypes) containers instead of raw arrays:
it handles the normalization (sum of weights, self-pair removal, jackknife splits) for you,
and the result carries the coordinates, edges and metadata --- ready to be combined into a
correlation function estimator, written to disk and plotted.

Because normalization is computed internally, pass **one** `Particles` instance for an autocorrelation
(instead of repeating it twice).

```python
import numpy as np
from cucount.numpy import Particles, BinAttrs, WeightAttrs, MeshAttrs, SplitAttrs
from cucount.types import count2, count2_analytic
import lsstypes as types

battrs = BinAttrs(s=np.linspace(0., 200., 21), mu=(np.linspace(-1., 1., 21), 'midpoint'))
wattrs = WeightAttrs()

data = Particles(data_positions, data_weights)
randoms = Particles(random_positions, random_weights)

DD = count2(data, battrs=battrs, wattrs=wattrs)['weight']              # autocorrelation
DR = count2(data, randoms, battrs=battrs, wattrs=wattrs)['weight']     # cross-correlation
RR = count2(randoms, battrs=battrs, wattrs=wattrs)['weight']
RD = DR.clone(value=DR.value()[:, ::-1])  # reverse mu for RD

correlation = types.Count2Correlation(estimator='landyszalay', DD=DD, DR=DR, RD=RD, RR=RR)
correlation.write('correlation.h5')
correlation.project(ells=[0, 2, 4]).plot()
```

For a periodic box, replace `RR` by the analytic counts:

```python
mattrs = MeshAttrs(data, boxsize=boxsize, battrs=battrs, periodic=True)
DD = count2(data, battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
RR = count2_analytic(battrs=battrs, mattrs=mattrs)
correlation = types.Count2Correlation(estimator='natural', DD=DD, RR=RR)
```

Passing `splits` to `Particles` together with `SplitAttrs` returns a `Count2Jackknife`,
which carries the per-split counts required for jackknife covariance estimates:

```python
data = data.clone(splits=splits)  # splits: integer array in [0, nsplits)
spattrs = SplitAttrs(mode='jackknife', nsplits=nsplits)
DD = count2(data, battrs=battrs, wattrs=wattrs, spattrs=spattrs, mattrs=mattrs)['weight']
```

`cucount.types` also provides `count3`, `count3close` and `count3_analytic`, returning
`Count3` / `Count3Poles` containers:

```python
from cucount.types import count3

counts = count3(data, battrs12=battrs12, battrs13=battrs13,
                sattrs12=sattrs, sattrs13=sattrs)['weight']
```

All these functions accept `cucount.jax` `Particles` as well, and dispatch to the JAX backend
automatically.

---

## 📎 References

- [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html)
- [shard\_map usage guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

---

## 🙏 Acknowledgment

Special thanks to **François Lanusse** for valuable advice on Python-JAX bindings.

---

## ❓ Questions or Feedback

Feel free to open an issue or discussion on the repository if you encounter problems or have suggestions.
