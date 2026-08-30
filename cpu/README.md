# CPU backend

A small, self-contained CPU `count2` used to prove out the mechanisms a full
CPU+CUDA cucount would need: [Google Highway](https://github.com/google/highway)
multi-ISA compile with runtime dispatch, compile-time binning specialisation,
`Float` templating, and SIMD traversal with scatter-accumulate.

It is additive and deletable: the CUDA build is unchanged when
`-DCUCOUNT_BUILD_CPU=OFF` is passed. Conversely, `-DCUCOUNT_BUILD_CUDA=OFF`
builds this backend (plus the CUDA-free `cucountlib.attrs` extension) without
nvcc, and `cucount.numpy` then serves `backend='cpu'` end to end.

## Build

Standalone (no nvcc needed):

```bash
cmake -S cpu -B build/cpu
cmake --build build/cpu -j
```

Or as part of the root project: `cmake -S . -B build -DCUCOUNT_BUILD_CPU=ON`.

Either way the module lands in a build-tree `cucountlib/` — a namespace
package that merges with an installed one — so the freshly built kernel is
importable without reinstalling:

```bash
PYTHONPATH=build/cpu:$PYTHONPATH python -c "from cucountlib import cpu"
```

One caveat: an *editable* install (`pip install -e .`) plants a
scikit-build-core finder that pins its own built extensions first regardless
of `PYTHONPATH`; run `python -s` (or set `PYTHONNOUSERSITE=1`) to let the
build tree win over a user-site editable install.

## Use

`cucountlib.cpu.count2` mirrors `cucountlib.cuda.count2`: it takes the same
`Particles`, `MeshAttrs`, `BinAttrs`, `WeightAttrs`, ... objects (from any of
the extensions -- pybind's foreign module_local loading casts them across) and
returns the same dict of named, shaped channels. The lowering to the kernel
(bin-policy classification, LOS mapping, packed-value columns) happens in C++
in `src/bind.cpp`:

```python
from cucountlib import cpu
counts = cpu.count2(particles1, particles2, mattrs, battrs,
                         nthreads=16,           # CPU threads
                         float32=False,
                         scatter='scalar')      # 'scalar' | 'binmajor'
# counts is {'weight': array} -- or the spin channels, named as CUDA names them
```

A low-level raw-array entry point, `cpu.count2_arrays`, keeps the old
`(pos1, w1, pos2, w2, sedges, ...)` signature for tests and benchmarks that
want to bypass the attrs layer.

Every ordered pair is visited, matching the CUDA backend: an autocorrelation
counts each pair twice and includes self-pairs.

`cpu.set_target('AVX2')` pins Highway to one ISA (`''` restores automatic
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
creation, which can add ~100 ms; warm it up before reading the ratio.

Per-call tuning goes through `count2(..., tuning={...})`, addressed to the
selected backend: the CPU backend accepts `nthreads` (CPU threads), `isa`
(pin one Highway target for the call) and `scatter`. Defaults remain
`CUCOUNT_CPU_NTHREADS` (else the affinity mask), automatic ISA selection,
and `'scalar'`. In `compare` mode the dict nests per backend:
`tuning={'cpu': {...}, 'cuda': {...}}`.

Requests the CPU backend cannot serve — theta/pole/k binning, angular mesh,
N-dimensional angular weights, jackknife splits, selections — are declined
by name, so it never silently computes something different.

## Verify

```bash
pytest tests/test_cpu.py -q                       # correctness
```

Performance regressions across kernel changes are checked with
[bench.py](bench.py) (kernel-only timings on fixed synthetic catalogues;
run once per build and compare the JSONs):

```bash
python cpu/bench.py --save base.json      # on the baseline build
python cpu/bench.py --against base.json   # on the candidate build
```

The BinMajor row doubles as the control: it shows the noise floor of the
measurement on that node. The dead-branch lesson is worth keeping: a
never-taken runtime branch in the inner loop cost the plain path 2-8%
until it became the compile-time ScalarTail split.

`tests/test_cpu.py` drives the backend through the public numpy API, checking
the full config matrix against both a numpy O(N^2) reference and the CUDA
backend, plus thread-count invariance, cross-ISA agreement, and the scatter
and precision axes that the public API does not expose.

## Scope

Covered: `s` and `(s, mu)` binning; linear, log and arbitrary edges; every
LOS (`z`, `x`, `y`, midpoint, firstpoint, endpoint); periodic and
non-periodic; `float`/`double`; per-object weights;
spin/shear (galaxy-shear and shear-shear channels, via the scalar projection
shared with CUDA in `include/pair_math.h` — the SIMD distance cull is
unchanged and surviving lanes take the shared per-pair math); bitwise (PIP),
negative and 1D angular weights (same scalar-tail pattern, via the shared
`pair_bitwise_weight` and `lookup_angular_weight`; the bit patterns ride
double storage whatever the working precision).

Multipoles: `(s, pole)` binning, with mu computed but not binned and
`(2 ell + 1) P_ell(mu)` accumulated into the pole axis (the fastest one),
through the `set_legendre` shared with CUDA. Pair selections on `s` and
`theta` ride the same scalar tail as a per-pair veto.

Not covered, deliberately: triplet counts, angular mesh, theta/rp/pi/k
binning, N-dimensional angular weights, jackknife splits, JAX FFI,
multi-device.
