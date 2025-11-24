# cucount: CUDA-Powered Pair Counts

**cucount** is a high-performance CUDA implementation for computing pair counts (positions - spins), optimized for GPUs. It provides both NumPy and JAX interfaces depending on your workflow.

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

# Compute pair counts
particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
counts = count2(particles1, particles2, battrs=battrs)
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

particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

counts = count2(particles1, particles2, battrs=battrs)
# counts is a dictionary with key "weight"
```

---

### 🧩 Multi-Device (Distributed) JAX Example

Using [shard\_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), you can parallelize over multiple devices:

```python
import jax
from jax import config
config.update("jax_enable_x64", True)

# Initialize distributed environment (if needed)
jax.distributed.initialize()

from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from cucount.jax import count2, Particles, BinAttrs

# Assume particles1 and particles2 are already defined
edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
los = 'midpoint'
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

# Create sharding mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=('x',))

# Define parallel pair-count function
count2_parallel = shard_map(
    lambda p1, p2: jax.lax.psum(count2(p1, p2, battrs=battrs), axis_name='x'),
    mesh=mesh,
    in_specs=(P('x'), P(None)),  # Shard only one input
    out_specs=P(None)
)

# Run distributed pair counts
counts = count2_parallel(particles1, particles2)
```

---

## 🛠️ TODO

- Implement periodic boundary conditions.

---

## 📎 References

- [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html)
- [shard\_map usage guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

---

## 🙏 Acknowledgment

Special thanks to **François Lanusse** for valuable advice on Python–JAX bindings.

---

## ❓ Questions or Feedback

Feel free to open an issue or discussion on the repository if you encounter problems or have suggestions.
