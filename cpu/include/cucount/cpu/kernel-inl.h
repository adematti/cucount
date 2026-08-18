// Per-target kernel. foreach_target.h re-includes this file once per SIMD
// target, so it uses the toggle-style include guard rather than #pragma once.
#if defined(CUCOUNT_CPU_KERNEL_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CUCOUNT_CPU_KERNEL_INL_H_
#undef CUCOUNT_CPU_KERNEL_INL_H_
#else
#define CUCOUNT_CPU_KERNEL_INL_H_
#endif

#include <chrono>

#include "cucount/cpu/mesh.h"
#include "cucount/cpu/types.h"
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
// Log() for the BIN_LOG policy; must be included per-target, like this file.
#include "hwy/contrib/math/math-inl.h"

#ifdef _OPENMP
#include <omp.h>
#endif

HWY_BEFORE_NAMESPACE();
namespace cucount {
namespace cpu {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Arithmetic uses operators; comparisons and masks use named functions, which
// is what keeps this source compilable on SVE/RVV where Highway cannot define
// operators (vectors there are compiler built-in sizeless types).

template <class D>
struct Binned {
    hn::VFromD<hn::RebindToSigned<D>> idx;
    hn::MFromD<D> ok;
};

// The range mask and the index arithmetic are computed by different routes
// (a direct comparison vs. a multiply, log or ladder), so rounding can hand a
// lane that passed the mask an index of exactly nbins. Clamping is cheaper
// than making the two agree exactly, and an unclamped index is an
// out-of-bounds store into the histogram.
template <class DI>
static HWY_INLINE hn::VFromD<DI> ClampIndex(DI di, hn::VFromD<DI> idx,
                                            size_t nbins) {
    using T = hn::TFromD<DI>;
    return hn::Min(hn::Max(idx, hn::Zero(di)),
                   hn::Set(di, static_cast<T>(nbins - 1)));
}

// Bin policies. Only Linear needs s itself; Log and Edges are evaluated
// against s^2, which lets the kernel skip the square root when ndim == 1.

struct BinLinear {
    static constexpr bool needs_s = true;

    template <class D, class Float>
    static Binned<D> Index(D d, hn::VFromD<D> s, hn::VFromD<D> /*s2*/,
                           const BinSpec<Float>& b) {
        const hn::RebindToSigned<D> di;
        const auto t = (s - hn::Set(d, b.lo)) * hn::Set(d, b.inv_step);
        const auto ok = hn::And(hn::Ge(s, hn::Set(d, b.lo)),
                                hn::Lt(s, hn::Set(d, b.hi)));
        return {ClampIndex(di, hn::ConvertTo(di, hn::Floor(t)), b.nbins), ok};
    }
};

struct BinLog {
    static constexpr bool needs_s = false;

    template <class D, class Float>
    static Binned<D> Index(D d, hn::VFromD<D> /*s*/, hn::VFromD<D> s2,
                           const BinSpec<Float>& b) {
        const hn::RebindToSigned<D> di;
        // inv_step already carries the factor 1/2 that turns log(s^2) into log(s).
        const auto t = (hn::Log(d, s2) - hn::Set(d, b.log_sq_lo)) *
                       hn::Set(d, b.inv_step);
        const auto ok = hn::And(hn::Ge(s2, hn::Set(d, b.sq_lo)),
                                hn::Lt(s2, hn::Set(d, b.sq_hi)));
        return {ClampIndex(di, hn::ConvertTo(di, hn::Floor(t)), b.nbins), ok};
    }
};

// Branchless comparison ladder against squared edges. O(nbins) per vector of
// pairs, but fully vectorised and with no data-dependent control flow, which
// beats a vectorised binary search at the bin counts these codes actually use.
struct BinEdges {
    static constexpr bool needs_s = false;

    template <class D, class Float>
    static Binned<D> Index(D d, hn::VFromD<D> /*s*/, hn::VFromD<D> s2,
                           const BinSpec<Float>& b) {
        const hn::RebindToSigned<D> di;
        const auto one = hn::Set(d, Float(1));
        auto cnt = hn::Zero(d);
        for (size_t k = 1; k < b.nbins; ++k) {
            cnt = cnt + hn::IfThenElseZero(hn::Ge(s2, hn::Set(d, b.sq[k])), one);
        }
        const auto ok = hn::And(hn::Ge(s2, hn::Set(d, b.sq_lo)),
                                hn::Lt(s2, hn::Set(d, b.sq_hi)));
        return {ClampIndex(di, hn::ConvertTo(di, cnt), b.nbins), ok};
    }
};

// mu is always linearly binned: every catalogue code in practice uses uniform
// mu bins, and one non-uniform axis is enough to exercise the policy seam.
template <class D, class Float>
static Binned<D> MuIndex(D d, hn::VFromD<D> mu, const BinSpec<Float>& b) {
    const hn::RebindToSigned<D> di;
    const auto t = (mu - hn::Set(d, b.lo)) * hn::Set(d, b.inv_step);
    const auto ok = hn::And(hn::Ge(mu, hn::Set(d, b.lo)),
                            hn::Lt(mu, hn::Set(d, b.hi)));
    return {ClampIndex(di, hn::ConvertTo(di, hn::Floor(t)), b.nbins), ok};
}

// Nearest-image convention. Valid because |dx| < 1.5 * boxsize always holds
// for points inside the box, which Round then folds to the [-L/2, L/2) branch.
template <class D>
static HWY_INLINE hn::VFromD<D> WrapPeriodic(D d, hn::VFromD<D> dx,
                                             hn::VFromD<D> box,
                                             hn::VFromD<D> inv_box) {
    return dx - box * hn::Round(dx * inv_box);
}

// Neighbour cell range along one axis. Cells are built at >= smax so one cell
// of padding suffices; fewer than 3 cells on an axis means the whole axis is
// swept instead, otherwise periodic wrapping would visit a cell twice.
struct AxisRange {
    int begin, end;  // inclusive; indices may be out of [0, n) and need wrapping
    bool wrap;
};

inline AxisRange axis_range(int i, int n, bool periodic) {
    if (n < 3) return {0, n - 1, false};
    if (!periodic) return {std::max(i - 1, 0), std::min(i + 1, n - 1), false};
    return {i - 1, i + 1, true};
}

template <class Float, int NDim, class SBin, LosKind LOS, bool Periodic,
          ScatterKind SC>
void Count2Impl(const Mesh<Float>& m1, const Mesh<Float>& m2,
                const BinSpec<Float>& sb, const BinSpec<Float>& mb,
                const double boxsize[3], double* out, int nthreads) {
    const hn::ScalableTag<Float> d;
    const hn::RebindToSigned<decltype(d)> di;
    const size_t L = hn::Lanes(d);

    const size_t nmu = (NDim == 2) ? mb.nbins : 1;
    const size_t nbins_total = sb.nbins * nmu;

    const Float bx[3] = {static_cast<Float>(boxsize[0]),
                         static_cast<Float>(boxsize[1]),
                         static_cast<Float>(boxsize[2])};
    const Float ibx[3] = {Float(1) / bx[0], Float(1) / bx[1], Float(1) / bx[2]};

    const int nx = m2.dims[0], ny = m2.dims[1], nz = m2.dims[2];
    const long ncells1 = static_cast<long>(m1.ncells());

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        std::vector<double> local(nbins_total, 0.0);
        // Replicated per-lane histogram for the BinMajor strategy. A flat
        // buffer rather than an array of vectors, because sizeless SVE/RVV
        // vector types cannot be stored in a container; and allocated through
        // Highway because aligned Load/Store need vector alignment, which
        // std::vector does not provide.
        hwy::AlignedFreeUniquePtr<Float[]> acc;
        if constexpr (SC == ScatterKind::BinMajor) {
            acc = hwy::AllocateAligned<Float>(nbins_total * L);
            std::fill(acc.get(), acc.get() + nbins_total * L, Float(0));
        }

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (long c1 = 0; c1 < ncells1; ++c1) {
            const size_t i0 = m1.start[c1], i1 = m1.start[c1 + 1];
            if (i0 == i1) continue;

            const int iz = static_cast<int>(c1 % m1.dims[2]);
            const int iy = static_cast<int>((c1 / m1.dims[2]) % m1.dims[1]);
            const int ix = static_cast<int>(c1 / (m1.dims[2] * m1.dims[1]));

            const AxisRange rx = axis_range(ix, nx, Periodic);
            const AxisRange ry = axis_range(iy, ny, Periodic);
            const AxisRange rz = axis_range(iz, nz, Periodic);

            for (int jx = rx.begin; jx <= rx.end; ++jx) {
                const int wx = rx.wrap ? wrap_index(jx, nx) : jx;
                for (int jy = ry.begin; jy <= ry.end; ++jy) {
                    const int wy = ry.wrap ? wrap_index(jy, ny) : jy;
                    for (int jz = rz.begin; jz <= rz.end; ++jz) {
                        const int wz = rz.wrap ? wrap_index(jz, nz) : jz;
                        const size_t c2 =
                            (static_cast<size_t>(wx) * ny + wy) * nz + wz;
                        const size_t j0 = m2.start[c2], j1 = m2.start[c2 + 1];
                        if (j0 == j1) continue;

                        for (size_t i = i0; i < i1; ++i) {
                            const auto x1 = hn::Set(d, m1.x[i]);
                            const auto y1 = hn::Set(d, m1.y[i]);
                            const auto z1 = hn::Set(d, m1.z[i]);
                            const auto w1 = hn::Set(d, m1.w[i]);

                            for (size_t j = j0; j < j1; j += L) {
                                const size_t n = std::min(L, j1 - j);
                                const auto active = hn::FirstN(d, n);

                                auto dx = hn::MaskedLoad(active, d, &m2.x[j]) - x1;
                                auto dy = hn::MaskedLoad(active, d, &m2.y[j]) - y1;
                                auto dz = hn::MaskedLoad(active, d, &m2.z[j]) - z1;
                                const auto w2 =
                                    hn::MaskedLoad(active, d, &m2.w[j]);

                                if constexpr (Periodic) {
                                    dx = WrapPeriodic(d, dx, hn::Set(d, bx[0]),
                                                      hn::Set(d, ibx[0]));
                                    dy = WrapPeriodic(d, dy, hn::Set(d, bx[1]),
                                                      hn::Set(d, ibx[1]));
                                    dz = WrapPeriodic(d, dz, hn::Set(d, bx[2]),
                                                      hn::Set(d, ibx[2]));
                                }

                                const auto s2 = dx * dx + dy * dy + dz * dz;

                                auto s = hn::Zero(d);
                                if constexpr (SBin::needs_s || NDim == 2)
                                    s = hn::Sqrt(s2);

                                const auto sres = SBin::Index(d, s, s2, sb);
                                auto ok = hn::And(active, sres.ok);
                                auto idx = sres.idx;

                                if constexpr (NDim == 2) {
                                    hn::VFromD<decltype(d)> num;
                                    hn::VFromD<decltype(d)> den;
                                    if constexpr (LOS == LosKind::AxisZ) {
                                        num = dz;
                                        den = s;
                                    } else {
                                        // Midpoint: los = p1 + p2, so
                                        // mu = (dr . los) / (|los| |dr|).
                                        const auto lx =
                                            hn::MaskedLoad(active, d, &m2.x[j]) + x1;
                                        const auto ly =
                                            hn::MaskedLoad(active, d, &m2.y[j]) + y1;
                                        const auto lz =
                                            hn::MaskedLoad(active, d, &m2.z[j]) + z1;
                                        num = dx * lx + dy * ly + dz * lz;
                                        den = s * hn::Sqrt(lx * lx + ly * ly +
                                                           lz * lz);
                                    }
                                    // s == 0 self-pairs would divide by zero;
                                    // they are pushed out of range instead.
                                    const auto safe = hn::Gt(den, hn::Zero(d));
                                    const auto mu = hn::IfThenElse(
                                        safe, num / hn::IfThenElse(
                                                        safe, den,
                                                        hn::Set(d, Float(1))),
                                        hn::Set(d, Float(-2)));
                                    const auto mres = MuIndex(d, mu, mb);
                                    ok = hn::And(ok, mres.ok);
                                    idx = idx * hn::Set(di, static_cast<
                                                        hn::TFromD<decltype(di)>>(
                                                        mb.nbins)) +
                                          mres.idx;
                                }

                                const auto wpair = w1 * w2;

                                if constexpr (SC == ScatterKind::Scalar) {
                                    // Invalid lanes are steered to bin 0 with
                                    // zero weight, so the store loop needs no
                                    // unpredictable branch.
                                    HWY_ALIGN hn::TFromD<decltype(di)>
                                        ibuf[HWY_MAX_LANES_D(decltype(di))];
                                    HWY_ALIGN Float
                                        wbuf[HWY_MAX_LANES_D(decltype(d))];
                                    hn::Store(hn::IfThenElseZero(
                                                  hn::RebindMask(di, ok), idx),
                                              di, ibuf);
                                    hn::Store(hn::IfThenElseZero(ok, wpair), d,
                                              wbuf);
                                    for (size_t k = 0; k < n; ++k) {
                                        local[static_cast<size_t>(ibuf[k])] +=
                                            static_cast<double>(wbuf[k]);
                                    }
                                } else {
                                    const auto wv = hn::IfThenElseZero(ok, wpair);
                                    for (size_t b = 0; b < nbins_total; ++b) {
                                        const auto hit = hn::RebindMask(
                                            d, hn::Eq(idx, hn::Set(di, static_cast<
                                                          hn::TFromD<decltype(di)>>(
                                                          b))));
                                        auto a = hn::Load(d, &acc[b * L]);
                                        a = a + hn::IfThenElseZero(hit, wv);
                                        hn::Store(a, d, &acc[b * L]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Flush the replicated histogram once per primary cell: the values
            // accumulated there span one cell's pairs only, so Float precision
            // is ample before widening into the double accumulator.
            if constexpr (SC == ScatterKind::BinMajor) {
                for (size_t b = 0; b < nbins_total; ++b) {
                    local[b] += static_cast<double>(
                        hn::ReduceSum(d, hn::Load(d, &acc[b * L])));
                    hn::Store(hn::Zero(d), d, &acc[b * L]);
                }
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        for (size_t b = 0; b < nbins_total; ++b) out[b] += local[b];
    }
}

// Dispatch helpers for the runtime config to compile-time template parameters.
// Highway does the SIMD dispatch at the outermost level; here, we're just dispatching
// on our own config parameters.

template <class Float, int NDim, class SBin, LosKind LOS, bool Periodic>
static void DispatchScatter(const Count2Args& a, const Mesh<Float>& m1,
                            const Mesh<Float>& m2, const BinSpec<Float>& sb,
                            const BinSpec<Float>& mb) {
    if (a.cfg.scatter == ScatterKind::Scalar) {
        Count2Impl<Float, NDim, SBin, LOS, Periodic, ScatterKind::Scalar>(
            m1, m2, sb, mb, a.boxsize, a.out, a.nthreads);
    } else {
        Count2Impl<Float, NDim, SBin, LOS, Periodic, ScatterKind::BinMajor>(
            m1, m2, sb, mb, a.boxsize, a.out, a.nthreads);
    }
}

template <class Float, int NDim, class SBin, LosKind LOS>
static void DispatchPeriodic(const Count2Args& a, const Mesh<Float>& m1,
                             const Mesh<Float>& m2, const BinSpec<Float>& sb,
                             const BinSpec<Float>& mb) {
    if (a.cfg.periodic) {
        DispatchScatter<Float, NDim, SBin, LOS, true>(a, m1, m2, sb, mb);
    } else {
        DispatchScatter<Float, NDim, SBin, LOS, false>(a, m1, m2, sb, mb);
    }
}

template <class Float, int NDim, class SBin>
static void DispatchLos(const Count2Args& a, const Mesh<Float>& m1,
                        const Mesh<Float>& m2, const BinSpec<Float>& sb,
                        const BinSpec<Float>& mb) {
    // With ndim == 1 there is no mu, so only one LOS instantiation is built.
    if (NDim == 1 || a.cfg.los == LosKind::AxisZ) {
        DispatchPeriodic<Float, NDim, SBin, LosKind::AxisZ>(a, m1, m2, sb, mb);
    } else {
        DispatchPeriodic<Float, NDim, SBin, LosKind::Midpoint>(a, m1, m2, sb, mb);
    }
}

template <class Float, int NDim>
static void DispatchSBin(const Count2Args& a, const Mesh<Float>& m1,
                         const Mesh<Float>& m2, const BinSpec<Float>& sb,
                         const BinSpec<Float>& mb) {
    switch (a.cfg.sbin) {
        case BinKind::Linear:
            DispatchLos<Float, NDim, BinLinear>(a, m1, m2, sb, mb);
            break;
        case BinKind::Log:
            DispatchLos<Float, NDim, BinLog>(a, m1, m2, sb, mb);
            break;
        case BinKind::Edges:
            DispatchLos<Float, NDim, BinEdges>(a, m1, m2, sb, mb);
            break;
    }
}

template <class Float>
static void DispatchNDim(const Count2Args& a) {
    const double smax = a.sedges[a.nsbins];

    BinSpec<Float> sb, mb;
    sb.set(a.sedges, a.nsbins, /*squared=*/true);
    if (a.cfg.sbin == BinKind::Linear) {
        sb.inv_step = static_cast<Float>(1.0 / (a.sedges[1] - a.sedges[0]));
    } else if (a.cfg.sbin == BinKind::Log) {
        // Half, because the policy feeds log(s^2) rather than log(s).
        sb.inv_step =
            static_cast<Float>(0.5 / std::log(a.sedges[1] / a.sedges[0]));
        sb.log_sq_lo = static_cast<Float>(std::log(a.sedges[0] * a.sedges[0]));
    }

    if (a.cfg.ndim == 2) {
        mb.set(a.muedges, a.nmubins, /*squared=*/false);
        mb.inv_step = static_cast<Float>(1.0 / (a.muedges[1] - a.muedges[0]));
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const Mesh<Float> m1 = build_mesh<Float>(a.pos1, a.w1, a.n1, a.boxsize,
                                             a.origin, smax);
    const Mesh<Float> m2 = build_mesh<Float>(a.pos2, a.w2, a.n2, a.boxsize,
                                             a.origin, smax);
    const auto t1 = Clock::now();

    if (a.cfg.ndim == 1) {
        DispatchSBin<Float, 1>(a, m1, m2, sb, mb);
    } else {
        DispatchSBin<Float, 2>(a, m1, m2, sb, mb);
    }

    if (a.timings) {
        using Sec = std::chrono::duration<double>;
        a.timings[0] = Sec(t1 - t0).count();
        a.timings[1] = Sec(Clock::now() - t1).count();
    }
}

void Count2Dispatch(const Count2Args& a) {
    if (a.cfg.float32) {
        DispatchNDim<float>(a);
    } else {
        DispatchNDim<double>(a);
    }
}

// Reports which ISA the dispatcher actually selected.
const char* KernelTarget() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace cpu
}  // namespace cucount
HWY_AFTER_NAMESPACE();

#endif  // include guard
