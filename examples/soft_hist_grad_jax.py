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

The run ends with a figure (``examples/figures/soft_hist_grad_jax.png``): the
soft histogram next to the hard one, what the kernel does to the bin contents,
the adjoint checked against central differences, and the loss of a short fit
that recovers a known weight amplitude without recompiling after step 0.
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


def _clustered(rng, n, nclusters=48, sigma=6.0):
    """A clustered catalogue: uniform clusters, Gaussian scatter inside.

    A uniform catalogue has scale-free pair statistics inside a periodic box,
    so an isotropic stretch barely changes its histogram -- useless for a fit
    demo.  Clusters make the counts respond to the stretch and give the
    objective landscape a real minimum.
    """
    centres = rng.uniform(0., BOX, (nclusters, 3))
    labels = rng.integers(0, nclusters, n)
    return (centres[labels] + rng.normal(0., sigma, (n, 3))) % BOX


def main():
    rng = np.random.default_rng(5)
    pos1 = _clustered(rng, N)
    pos2 = _clustered(rng, N)
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

    # 6. A short fit: recover an amplitude on the second catalogue.  The soft
    #    histogram is exactly linear in that amplitude, so the objective is a
    #    well-posed quadratic and the minimum sits exactly at a = 1.  Only
    #    floats move: JAX does not recompile inside the loop.
    A_TRUE = 1.0
    A_START = 0.62

    reference_vec = np.asarray(jax.jit(
        lambda ww: _counts(pos1, w1, pos2, ww, battrs, mattrs, SIGMA))(
            w2 * A_TRUE))
    scale = float(np.sqrt((reference_vec ** 2).sum()))

    def residual(a):
        c = _counts(pos1, w1, pos2, w2 * a, battrs, mattrs, SIGMA)
        return (((c - reference_vec) / scale) ** 2).sum()

    value_grad_a = jax.jit(jax.value_and_grad(residual))
    a = jnp.array(A_START)
    lr = 0.03
    m = v = jnp.zeros(())
    print(f'\n[6] recovering an amplitude a = {A_TRUE} from a = {A_START} '
          '(Adam, 15 steps)')
    first = None
    history = []
    for step in range(15):
        t0 = time.perf_counter()
        l, dl = value_grad_a(a)
        dl.block_until_ready()
        dt = time.perf_counter() - t0
        first = first or dt
        history.append((float(l), float(a), float(dl)))
        m = 0.9 * m + 0.1 * dl
        v = 0.999 * v + 0.001 * dl ** 2
        a = a - lr * (m / (1 - 0.9 ** (step + 1))) / (
            jnp.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
        if step in (0, 1, 2, 14):
            print(f'    step {step:2d}: L = {float(l):.3e}, '
                  f'dL/da = {float(dl): .3e}, a = {float(a):.4f}, '
                  f'{dt * 1e3:.1f} ms')
    print(f'    first step {first * 1e3:.1f} ms, later steps '
          f'{dt * 1e3:.1f} ms -> the loop pays no compile after step 0')

    # 7. The same story as a figure: what the soft kernel does to the counts,
    #    that the adjoint is the derivative it claims to be, and what a fit
    #    through it looks like.
    import os
    import tempfile

    os.environ.setdefault('MPLCONFIGDIR',
                          os.path.join(tempfile.gettempdir(), 'cucount-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hard_counts = np.asarray(jax.jit(
        lambda p1, w1, p2, w2: cjax.count2(cjax.Particles(p1, w1),
                                           cjax.Particles(p2, w2),
                                           battrs=battrs, mattrs=mattrs,
                                           backend='cuda')['weight'])(
        pos1, w1, pos2, w2))
    soft_counts = np.asarray(jax.jit(
        lambda p1, w1, p2, w2: _counts(p1, w1, p2, w2, battrs, mattrs, SIGMA))(
        pos1, w1, pos2, w2))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout='constrained')
    fig.suptitle('Soft counts and their adjoint through the XLA FFI custom call',
                 fontsize=11.5)
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])

    ax = axes[0, 0]
    ax.step(np.repeat(EDGES, 2)[1:-1], np.repeat(hard_counts, 2), where='mid',
            color='0.3', lw=1.6, label='hard binning')
    ax.plot(centres, hard_counts, 'o', ms=3.5, color='0.3')
    ax.plot(centres, soft_counts, '-', marker='o', ms=3.5, lw=1.4,
            color=plt.get_cmap('viridis')(0.45),
            label=rf'soft, $\sigma={SIGMA:g}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('pair counts')
    ax.set_title('Soft versus hard pair counts (jax FFI path)')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    ax = axes[0, 1]
    ax.axhline(1.0, color='0.3', lw=1.2)
    ax.plot(centres, soft_counts / hard_counts, '-', marker='o', ms=3.5,
            lw=1.4, color=plt.get_cmap('viridis')(0.45))
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('soft / hard counts')
    ax.set_title('The kernel smooths the bin contents')
    ax.grid(alpha=0.25)

    # The adjoint against central differences of the same scalar loss.
    cot = target
    loss_pos2 = jax.jit(lambda q: (_counts(pos1, w1, q, w2, battrs,
                                           mattrs, SIGMA) * cot).sum())
    grad_pos2 = np.asarray(jax.grad(loss_pos2)(pos2))
    flat = np.argsort(-np.abs(grad_pos2).ravel())[:8]
    picked, axis = np.unravel_index(flat, grad_pos2.shape)
    h = 1e-4
    fd = []
    for i, ax_i in zip(picked, axis):
        plus = np.array(pos2, copy=True)
        minus = np.array(pos2, copy=True)
        plus[i, ax_i] += h
        minus[i, ax_i] -= h
        fd.append((float(loss_pos2(plus)) - float(loss_pos2(minus))) / (2 * h))
    fd = np.array(fd)
    adj = grad_pos2[picked, axis]

    ax = axes[1, 0]
    ax.loglog(fd, adj, 'o', ms=5, color=plt.get_cmap('viridis')(0.45),
              label='adjoint from the custom VJP')
    lim = [min(np.abs(fd).min(), np.abs(adj).min()) * 0.5,
           max(np.abs(fd).max(), np.abs(adj).max()) * 2.]
    ax.loglog(lim, lim, '-', lw=1.2, color='0.3', label='$y = x$')
    ax.set_xlabel(r'central difference $dL/dp$')
    ax.set_ylabel(r'adjoint $dL/dp$')
    ax.set_title('The adjoint is the derivative of the same loss')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    ax = axes[1, 1]
    steps = np.arange(len(history))
    ax.semilogy(steps, [hh[0] for hh in history], 'o-', ms=3.5, lw=1.4,
                color=plt.get_cmap('viridis')(0.45), label=r'$L$ along the fit')
    ax.set_xlabel('Adam step')
    ax.set_ylabel(r'$L$')
    ax.set_title(r'Recovering an amplitude (no recompile after step 0)')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, 'soft_hist_grad_jax.png')
    fig.savefig(out, dpi=150)
    print(f'\n[7] figure saved to {out}')

    cache_dir = jax.config.jax_compilation_cache_dir
    print(f'\ncompilation cache dir: {cache_dir or "(none: set JAX_COMPILATION_CACHE_DIR)"}')
    print(f'total traces of the loss: {traces["value_and_grad"]} '
          f'(one per distinct shape/static-argument combination)')


if __name__ == '__main__':
    main()
