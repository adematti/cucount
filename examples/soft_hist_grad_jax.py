"""What jax compiles (and re-uses) when you differentiate soft pair counts.

Run it on a GPU node, twice, with a shared compilation-cache directory::

    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache python examples/soft_hist_grad_jax.py
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache python examples/soft_hist_grad_jax.py

(``scripts/run_soft_grad_example.sh`` does exactly that.)  The second process
should find both executables in the on-disk cache and skip compilation.

The point of the example is the caching contract of the soft-histogram FFI pair:

* one ``jax.jit`` trace of ``value_and_grad`` contains **two** foreign calls --
  ``count2_soft_cuda`` (forward) and ``count2_soft_vjp_cuda`` (adjoint) -- so a
  gradient costs one compile, not one per call;
* the compilation cache is keyed on shapes *and* the static arguments
  (``soft_sigma``, ``soft_cutoff``, the bin edges, the mesh), never on the array
  *values*, so new data or a new histogram cotangent reuses the executable;
* changing the particle count or ``soft_sigma`` re-traces and recompiles;
* after the first call the steady-state cost of a gradient is one forward plus
  one adjoint kernel launch, and that is what a fit loop pays.
"""

import re
import time

import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs
import jax
import jax.numpy as jnp

from cucount import jax as cjax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

BOX = 400.0
SMAX = 150.0
EDGES = np.linspace(2.0, 120.0, 41)
SIGMA = 3.0
CUTOFF = 1e-13
N = 2048

traces = {'value_and_grad': 0, 'counts': 0}


def _timed(fn, *args):
    """Run fn(*args), sync, and return (result, seconds_including_compile)."""
    t0 = time.perf_counter()
    out = fn(*args)
    leaves = jax.tree_util.tree_leaves(out)
    for leaf in leaves:
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()
            break
    return out, time.perf_counter() - t0


def _counts(pos1, w1, pos2, w2, battrs, mattrs, sigma):
    """The soft histogram itself: counts[:] = sum_pairs w W(s)."""
    traces['counts'] += 1
    return cjax.count2_soft(cjax.Particles(pos1, w1),
                            cjax.Particles(pos2, w2),
                            battrs=battrs,
                            soft_sigma=sigma,
                            soft_cutoff=CUTOFF,
                            mattrs=mattrs,
                            backend='cuda')['weight']


def main():
    rng = np.random.default_rng(5)
    pos1 = rng.uniform(0., BOX, (N, 3))
    pos2 = rng.uniform(0., BOX, (N, 3))
    w1 = rng.uniform(0.5, 1.5, N)
    w2 = rng.uniform(0.5, 1.5, N)
    battrs = BinAttrs(s=EDGES)
    mattrs = MeshAttrs(cjax.Particles(pos1, w1), cjax.Particles(pos2, w2),
                       boxsize=BOX, battrs=battrs, periodic=True)

    target = np.zeros(EDGES.size - 1)
    target[10] = 1.0  # one-hot cotangent: d(that bin)/d(positions)

    def loss(pos1, w1, pos2, w2, cot):
        traces['value_and_grad'] += 1
        return (_counts(pos1, w1, pos2, w2, battrs, mattrs, SIGMA) * cot).sum()

    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

    print(f'backend: {jax.devices()[0]}, n = {N} per catalogue, '
          f'{EDGES.size - 1} bins, device_count = {jax.device_count()}')

    # 1. What a gradient contains: the adjoint always, the forward only when the
    #    primal value is actually used (XLA drops it for a bare jax.grad).
    #    StableHLO prints a custom call as `@count2_soft_cuda` in the region form
    #    and as `custom_call_target="..."` / `call_target_name = "..."` in the
    #    generic form, so accept all three spellings.
    _call_target = re.compile(
        r'(?:@|custom_call_target\s*=\s*"|call_target_name\s*=\s*")([A-Za-z0-9_]+)')

    def foreign_calls(fn, *args):
        text = jax.jit(fn).lower(*args).as_text()
        names = [n for n in _call_target.findall(text) if n.startswith('count2_soft')]
        fwd = sum(n == 'count2_soft_cuda' for n in names)
        adj = sum(n == 'count2_soft_vjp_cuda' for n in names)
        return f'forward x{fwd}, adjoint x{adj}'

    print(f'\n[1] jax.grad   -> {foreign_calls(jax.grad(loss, argnums=(0, 1)), pos1, w1, pos2, w2, target)}')
    print(f'    value_and_grad -> {foreign_calls(jax.value_and_grad(loss, argnums=(0, 1)), pos1, w1, pos2, w2, target)}')
    costs = jax.jit(jax.value_and_grad(loss, argnums=(0, 1))).lower(
        pos1, w1, pos2, w2, target).compile().cost_analysis()
    rows = costs.values() if isinstance(costs, dict) else costs
    rows = [r for r in rows if isinstance(r, dict)]
    if rows:
        print(f'    compiler estimate: '
              f'{sum(r.get("flops", 0) for r in rows):.3g} flops, '
              f'{sum(r.get("bytes accessed", 0) for r in rows):.3g} bytes')

    # 2. Cold vs warm: compile once, then the cache answers.
    _, t_cold = _timed(value_and_grad, pos1, w1, pos2, w2, target)
    _, t_warm = _timed(value_and_grad, pos1, w1, pos2, w2, target)
    print(f'\n[2] value_and_grad(first call, includes compile) = {t_cold * 1e3:8.2f} ms')
    print(f'    value_and_grad(second call, cache hit)      = {t_warm * 1e3:8.2f} ms')
    print(f'    traces so far: {traces}')

    # 3. New cotangent data -> no recompile.
    rng2 = np.random.default_rng(6)
    for k in range(3):
        cot = rng2.normal(size=EDGES.size - 1)
        _, dt = _timed(value_and_grad, pos1, w1, pos2, w2, cot)
        print(f'[3] new cotangent #{k}: {dt * 1e3:8.2f} ms  '
              f'(recompiled: {traces["value_and_grad"]}, '
              f'forward traces: {traces["counts"]})')

    # 4. New shapes and new static arguments do recompile.
    _, t_shape = _timed(value_and_grad, pos1[:N // 2], w1[:N // 2],
                        pos2[:N // 2], w2[:N // 2], target)
    print(f'\n[4] half the particles (new shape): {t_shape * 1e3:8.2f} ms, '
          f'traces: {traces["value_and_grad"]}')

    def loss_sharper(pos1, w1, pos2, w2, cot):
        return (cjax.count2_soft(cjax.Particles(pos1, w1), cjax.Particles(pos2, w2),
                                 battrs=battrs, soft_sigma=1.5, soft_cutoff=CUTOFF,
                                 mattrs=mattrs, backend='cuda')['weight'] * cot).sum()

    grad_sharper = jax.jit(jax.grad(loss_sharper, argnums=0))
    _, t_sigma = _timed(grad_sharper, pos1, w1, pos2, w2, target)
    _, t_sigma_warm = _timed(grad_sharper, pos1, w1, pos2, w2, target)
    print(f'    soft_sigma=1.5 (new static arg): {t_sigma * 1e3:8.2f} ms cold, '
          f'{t_sigma_warm * 1e3:8.2f} ms warm')

    # 5. Steady state: what a fit loop actually pays per step.
    value_grad_pos = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    forward_only = jax.jit(lambda *a: _counts(*a, battrs, mattrs, SIGMA))
    _timed(forward_only, pos1, w1, pos2, w2)  # warm: [5] compares steady states
    _, t_fwd = _timed(forward_only, pos1, w1, pos2, w2)
    times = []
    for _ in range(5):
        _, dt = _timed(value_grad_pos, pos1, w1, pos2, w2, target)
        times.append(dt)
    print(f'\n[5] steady state: forward-only {t_fwd * 1e3:.2f} ms, '
          f'value+grad median {np.median(times) * 1e3:.2f} ms '
          f'(ratio {np.median(times) / t_fwd:.2f}x, one extra kernel)')

    # 6. A short fit: stretch the second catalogue along z until the one-hot
    #    bin is matched.  Only floats move, so nothing recompiles.
    def model(a):
        z = pos2 * jnp.array([1.0, 1.0, a])
        return (_counts(pos1, w1, z, w2, battrs, mattrs, SIGMA) * target).sum()

    value_grad_a = jax.jit(jax.value_and_grad(model))
    a = jnp.array(1.0)
    lr = 0.05
    m = v = jnp.zeros(())
    print('\n[6] fitting an anisotropic stretch (Adam, 15 steps)')
    first = None
    for step in range(15):
        t0 = time.perf_counter()
        l, dl = value_grad_a(a)
        dl.block_until_ready()
        dt = time.perf_counter() - t0
        first = first or dt
        m = 0.9 * m + 0.1 * dl
        v = 0.999 * v + 0.001 * dl ** 2
        a = a - lr * (m / (1 - 0.9 ** (step + 1))) / (
            jnp.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
        if step in (0, 1, 2, 14):
            print(f'    step {step:2d}: target-window counts = {float(l):.6f}, '
                  f'd/da = {float(dl): .3e}, a = {float(a):.4f}, {dt * 1e3:.1f} ms')
    print(f'    first step {first * 1e3:.1f} ms, later steps '
          f'{dt * 1e3:.1f} ms -> the loop pays no compile after step 0')

    cache_dir = jax.config.jax_compilation_cache_dir
    print(f'\ncompilation cache dir: {cache_dir or "(none: set JAX_COMPILATION_CACHE_DIR)"}')
    print(f'total traces of the loss: {traces["value_and_grad"]} '
          f'(one per distinct shape/static-argument combination)')


if __name__ == '__main__':
    main()
