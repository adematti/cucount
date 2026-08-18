# CPU backend

A small, self-contained CPU `count2` used to prove out the mechanisms a full
CPU+CUDA cucount would need: [Google Highway](https://github.com/google/highway)
multi-ISA compile with runtime dispatch, compile-time binning specialisation,
`Float` templating, and SIMD traversal with scatter-accumulate.

It is additive and deletable. Nothing under `src/` or `include/`
is touched, and the CUDA build is unchanged when `-DCUCOUNT_BUILD_CPU=OFF`
is passed.

## Build

Standalone (no nvcc needed):

```bash
cmake -S cpu -B build/cpu
cmake --build build/cpu -j
```

Or as part of the root project: `cmake -S . -B build -DCUCOUNT_BUILD_CPU=ON`.

## Use

```python
import cpucount
counts = cpucount.count2(pos1, w1, pos2, w2, sedges,
                         muedges=None,          # None -> 1D in s
                         boxsize=(1000.,)*3,
                         bin='lin',             # 'lin' | 'log' | 'edges'
                         los='z',               # 'z' | 'midpoint'
                         periodic=True, float32=False,
                         scatter='scalar',      # 'scalar' | 'binmajor'
                         nthreads=16)
```

Every ordered pair is visited, matching the CUDA backend: an autocorrelation
counts each pair twice and includes self-pairs.

`cpucount.set_target('AVX2')` pins Highway to one ISA (`''` restores automatic
selection); `available_targets()` and `current_target()` report what is
reachable. This is how the tests check that every compiled ISA agrees and how
the benchmark measures SIMD-width scaling.

## Using it from the normal cucount API

Once built, `cucount.numpy.count2` can route to the CPU backend without any
change to calling code. Select with `CUCOUNT_BACKEND` or a `backend=` keyword:

| mode | behaviour |
|---|---|
| `cuda` | default |
| `cpu` | CPU backend, raising a precise reason if it cannot serve the request |
| `compare` | run both and raise if they disagree |

No backend is ever chosen implicitly.

```bash
CUCOUNT_BACKEND=compare pytest tests/   # differential-test the CPU backend
CUCOUNT_BACKEND=cpu pytest tests/       # report what it cannot yet serve
```

`compare` turns the existing suite into a differential test against CUDA with
no test edits. It also logs wall-clock for both backends:

```
INFO   compare: cpu 0.0970 s, cuda 0.0417 s -- cpu 2.32x slower
DEBUG  cpu backend: mesh 40.7 ms, pairs 53.9 ms
DEBUG  Time elapsed: 31.1 ms.          <- CUDA kernel only, from the C++ side
```

Enable with `cucount.numpy.setup_logging(logging.INFO)` (or `DEBUG` for the
per-phase split). Note the first CUDA call of a process includes context
creation, which can add ~100 ms; warm it up before reading the ratio. `nthreads` keeps its existing meaning (number of GPUs), so the
CPU backend takes its thread count from `CUCOUNT_CPU_NTHREADS`, defaulting to
the number available in the affinity mask.

Requests the CPU backend cannot serve — theta/pole/k binning, angular mesh,
bitwise or angular weights, spin, jackknife splits, selections — are declined
by name, so it never silently computes something different.

## Verify

```bash
pytest tests/test_cpu.py -q                       # correctness
```

`tests/test_cpu.py` drives the backend through the public numpy API, checking
the full config matrix against both a numpy O(N^2) reference and the CUDA
backend, plus thread-count invariance, cross-ISA agreement, and the scatter
and precision axes that the public API does not expose.

## Scope

Covered: `s` and `(s, mu)` binning; linear, log and arbitrary edges; `z` and
midpoint LOS; periodic and non-periodic; `float`/`double`; per-object weights.

Not covered, deliberately: triplet counts, angular mesh, theta/rp/pi/pole/k
binning, bitwise and angular weights, spin/shear, jackknife splits, JAX FFI,
multi-device.
