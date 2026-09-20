# CUDA backend

The default backend: the `cucountlib.cuda` extension (count2, count3,
count3close on the GPU) and, when `jax.ffi` headers are found at build time,
the `cucountlib.ffi_cuda` module behind the JAX API. See the
[top-level README](../README.md) for usage; this directory only holds the
CUDA sources, mirroring [cpu/](../cpu/).

Backend-neutral code lives at the repository root and is compiled into these
modules too: `include/attrs.h` (the attribute classes and their shared
pybind registration), `include/cmath.h` (scalar per-pair math shared
with the CPU backend — Legendre, spherical Bessel, spin projection) and
`include/common.h` (type definitions).

## Build

Built by the root project by default:

```bash
cmake -S .. -B build            # or: pip install .
```

`-DCMAKE_CUDA_ARCHITECTURES=...` overrides the default `70 75 80` — required
with CUDA >= 13, which dropped `compute_70` (e.g. `80` for A100).
`-DCUCOUNT_BUILD_CUDA=OFF` skips this directory entirely; `cucount.numpy`
then still imports and serves `backend='cpu'`.

## Kernel limits

`ELLMAX` in `include/count3close.h` caps the multipoles the triplet kernels
compute (raising it also needs `MMAX_SIZE = ELLMAX + 1`); the Python frontend
mirrors it as `KERNEL_ELLMAX` and raises rather than letting the kernel clamp
silently.
