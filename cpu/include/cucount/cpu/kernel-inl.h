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
// Scalar per-pair math shared with the CUDA backend (plain inline templates,
// no SIMD attributes, so a single definition per program is fine even though
// this file is re-included per target).
#include "pair_math.h"
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

// Neighbour cell range along one axis, delta cells each way. A port of the
// CUDA set_cartesian_bounds, so the sweep adapts to whatever mesh MeshAttrs
// chose rather than requiring cells at least smax wide. A periodic axis that
// the window wraps all the way round is swept once instead, otherwise a cell
// would be visited twice.
struct AxisRange {
    int begin, end;  // inclusive; indices may be out of [0, n) and need wrapping
    bool wrap;
};

inline AxisRange axis_range(int i, int n, bool periodic, int delta) {
    if (!periodic) return {std::max(i - delta, 0), std::min(i + delta, n - 1), false};
    if (2 * delta + 1 >= n) return {0, n - 1, false};
    return {i - delta, i + delta, true};
}

// ScalarTail is compile-time: when false (plain w1 * w2 weighting) the whole
// per-lane tail below is compiled OUT of the inner loop, keeping its codegen
// identical to the pre-spin kernel -- carrying the dead branch cost the plain
// path 2-8% in the A/B benchmark.
template <class Float, int NDim, bool Poles, class SBin, LosKind LOS,
          bool Periodic, ScatterKind SC, bool ScalarTail>
void Count2Impl(const Count2Args& a, const Mesh<Float>& m1,
                const Mesh<Float>& m2, const BinSpec<Float>& sb,
                const BinSpec<Float>& mb) {
    const hn::ScalableTag<Float> d;
    const hn::RebindToSigned<decltype(d)> di;
    const size_t L = hn::Lanes(d);

    const size_t nmu = (NDim == 2) ? mb.nbins : 1;
    const size_t nbins_geom = sb.nbins * nmu;

    // Multipole axis (VAR_POLE): the fastest bin axis. mu is then computed
    // but never binned, and each pair adds (2 ell + 1) P_ell(mu) into nells
    // consecutive bins -- per channel, so poles and spin compose.
    const size_t nells = Poles ? a.nells : 1;
    const int* ells = a.ells;
    const int ellstep_legendre = a.ells_even ? 2 : 1;
    const int ellmin = (Poles && a.nells) ? ells[0] : 0;
    const int ellmax = (Poles && a.nells) ? ells[a.nells - 1] : 0;
    const size_t nbins_total = nbins_geom * nells;

    // mu is needed for the mu axis and for the multipoles alike.
    constexpr bool NeedMu = (NDim == 2) || Poles;

    // Spin channels replace the plain weight, matching the CUDA layout:
    // (plus, cross) for one spinning side, (plus*plus, cross*plus,
    // cross*cross) for two. The SIMD geometry cull is untouched; surviving
    // lanes take the shared scalar projection.
    const bool sp1 = !m1.e1.empty();
    const bool sp2 = !m2.e1.empty();
    const bool has_spin = sp1 || sp2;
    const size_t nw = 1 + (sp1 ? 1 : 0) + (sp2 ? 1 : 0);

    // PIP (bitwise) and negative weighting stop the pair weight factorizing
    // as w1 * w2, so like spin they ride the scalar tail; the application
    // order (individual, bitwise, negative subtraction, then spin channels)
    // mirrors the CUDA add_weight2. Both apply only when both sides carry
    // the columns, like the CUDA kernel.
    const size_t nbit = a.nbitwise;
    const bool has_bitwise = !m1.bw.empty() && !m2.bw.empty() && nbit;
    const bool has_negative = !m1.nw.empty() && !m2.nw.empty();
    const bool has_angular = a.angular_weight && a.angular_shape;
    // Pair selections veto a pair outright; they ride the tail so the plain
    // vector path needs no extra compare (and no extra instantiation).
    const bool sel_s = a.sel_s;
    const bool sel_theta = a.sel_theta;
    const double sel_s2_min = a.sel_s_min * a.sel_s_min;
    const double sel_s2_max = a.sel_s_max * a.sel_s_max;
    (void)(has_spin || has_bitwise || has_negative || has_angular ||
           sel_s || sel_theta);  // dispatch chose ScalarTail
    BitwiseWeight bitwise = {};
    IndexValue iv_bw = {};  // synthetic: columns at offset 0, stride nbit
    if (has_bitwise) {
        bitwise.default_value = a.bitwise_default;
        bitwise.nrealizations = a.bitwise_nrealizations;
        bitwise.noffset = a.bitwise_noffset;
        bitwise.p_nbits = a.bitwise_p_nbits;
        bitwise.p_correction_nbits = const_cast<double*>(a.bitwise_p_correction);
        iv_bw.size_bitwise_weight = nbit;
        iv_bw.size = nbit;
    }
    AngularWeight angular = {};
    if (has_angular) {
        angular.ndim = 1;
        angular.size = a.angular_shape;
        angular.weight = const_cast<double*>(a.angular_weight);
        angular.sep[0] = const_cast<double*>(a.angular_sep);
        angular.sep_is_edges[0] = a.angular_sep_is_edges;
        angular.bin[0] = static_cast<BIN_TYPE>(a.angular_bin);
        angular.shape[0] = a.angular_shape;
    }

    const double* boxsize = a.boxsize;
    double* out = a.out;
    const int nthreads = a.nthreads;

    const Float bx[3] = {static_cast<Float>(boxsize[0]),
                         static_cast<Float>(boxsize[1]),
                         static_cast<Float>(boxsize[2])};
    const Float ibx[3] = {Float(1) / bx[0], Float(1) / bx[1], Float(1) / bx[2]};

    const int nx = m2.dims[0], ny = m2.dims[1], nz = m2.dims[2];
    const long ncells1 = static_cast<long>(m1.ncells());

    // How many cells each way can hold a neighbour within smax, for the mesh
    // the caller chose. The raw-array entry point has no MeshAttrs, so smax
    // falls back to the last s edge there.
    const double smax_eff = (a.smax > 0.) ? a.smax : a.sedges[a.nsbins];
    const int dcell[3] = {
        std::max(1, static_cast<int>(std::ceil(smax_eff / bx[0] * nx))),
        std::max(1, static_cast<int>(std::ceil(smax_eff / bx[1] * ny))),
        std::max(1, static_cast<int>(std::ceil(smax_eff / bx[2] * nz)))};

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        std::vector<double> local(nw * nbins_total, 0.0);
        // Replicated per-lane histogram for the BinMajor strategy. A flat
        // buffer rather than an array of vectors, because sizeless SVE/RVV
        // vector types cannot be stored in a container; and allocated through
        // Highway because aligned Load/Store need vector alignment, which
        // std::vector does not provide.
        hwy::AlignedFreeUniquePtr<Float[]> acc;
        if constexpr (SC == ScatterKind::BinMajor) {
            acc = hwy::AllocateAligned<Float>(nbins_geom * L);
            std::fill(acc.get(), acc.get() + nbins_geom * L, Float(0));
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

            const AxisRange rx = axis_range(ix, nx, Periodic, dcell[0]);
            const AxisRange ry = axis_range(iy, ny, Periodic, dcell[1]);
            const AxisRange rz = axis_range(iz, nz, Periodic, dcell[2]);

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


                            // FirstPoint LOS: particle 1's unit-sphere
                            // position is constant across the j vector.
                            auto sx1 = hn::Zero(d), sy1 = hn::Zero(d),
                                 sz1 = hn::Zero(d);
                            if constexpr (LOS == LosKind::FirstPoint) {
                                sx1 = hn::Set(d, m1.sx[i]);
                                sy1 = hn::Set(d, m1.sy[i]);
                                sz1 = hn::Set(d, m1.sz[i]);
                            }

                            Float r1v[3] = {0, 0, 0};
                            Float e1v[2] = {0, 0};
                            if constexpr (ScalarTail) {
                                if (has_spin || has_angular || sel_theta) {
                                    r1v[0] = m1.sx[i];
                                    r1v[1] = m1.sy[i];
                                    r1v[2] = m1.sz[i];
                                }
                                if (sp1) {
                                    e1v[0] = m1.e1[i];
                                    e1v[1] = m1.e2[i];
                                }
                            }

                            for (size_t j = j0; j < j1; j += L) {
                                const size_t n = std::min(L, j1 - j);
                                const auto active = hn::FirstN(d, n);

                                // LoadN: MaskedLoad may fault past the tail
                                // on HWY_MEM_OPS_MIGHT_FAULT targets.
                                const auto x2 = hn::LoadN(d, &m2.x[j], n);
                                const auto y2 = hn::LoadN(d, &m2.y[j], n);
                                const auto z2 = hn::LoadN(d, &m2.z[j], n);
                                const auto w2 = hn::LoadN(d, &m2.w[j], n);
                                auto dx = x2 - x1;
                                auto dy = y2 - y1;
                                auto dz = z2 - z1;

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
                                if constexpr (SBin::needs_s || NeedMu)
                                    s = hn::Sqrt(s2);

                                const auto sres = SBin::Index(d, s, s2, sb);
                                auto ok = hn::And(active, sres.ok);
                                auto idx = sres.idx;

                                // Declared unconditionally but written only
                                // for poles; the compiler elides it entirely
                                // otherwise. Hoisting `mu` itself out of this
                                // block instead cost the 2D path 4-5%.
                                HWY_ALIGN Float
                                    mubuf[HWY_MAX_LANES_D(decltype(d))];
                                if constexpr (NeedMu) {
                                    hn::VFromD<decltype(d)> num;
                                    hn::VFromD<decltype(d)> den;
                                    if constexpr (LOS == LosKind::AxisZ) {
                                        num = dz;
                                        den = s;
                                    } else if constexpr (LOS ==
                                                         LosKind::AxisX) {
                                        num = dx;
                                        den = s;
                                    } else if constexpr (LOS ==
                                                         LosKind::AxisY) {
                                        num = dy;
                                        den = s;
                                    } else if constexpr (LOS ==
                                                         LosKind::FirstPoint) {
                                        num = dx * sx1 + dy * sy1 + dz * sz1;
                                        den = s;
                                    } else if constexpr (LOS ==
                                                         LosKind::EndPoint) {
                                        const auto sx2 =
                                            hn::LoadN(d, &m2.sx[j], n);
                                        const auto sy2 =
                                            hn::LoadN(d, &m2.sy[j], n);
                                        const auto sz2 =
                                            hn::LoadN(d, &m2.sz[j], n);
                                        num = dx * sx2 + dy * sy2 + dz * sz2;
                                        den = s;
                                    } else {
                                        // Midpoint: los = p1 + p2, so
                                        // mu = (dr . los) / (|los| |dr|).
                                        const auto lx = x2 + x1;
                                        const auto ly = y2 + y1;
                                        const auto lz = z2 + z1;
                                        num = dx * lx + dy * ly + dz * lz;
                                        den = s * hn::Sqrt(lx * lx + ly * ly +
                                                           lz * lz);
                                    }
                                    // Coincident points (s2 == 0) take mu = 0,
                                    // matching the CUDA kernel; a zero
                                    // denominator with s > 0 stays out of range.
                                    const auto safe = hn::Gt(den, hn::Zero(d));
                                    auto mu = hn::IfThenElse(
                                        safe, num / hn::IfThenElse(
                                                        safe, den,
                                                        hn::Set(d, Float(1))),
                                        hn::Set(d, Float(-2)));
                                    mu = hn::IfThenElse(hn::Eq(s2, hn::Zero(d)),
                                                        hn::Zero(d), mu);

                                    if constexpr (NDim == 2) {
                                        const auto mres = MuIndex(d, mu, mb);
                                        ok = hn::And(ok, mres.ok);
                                        idx = idx * hn::Set(di, static_cast<
                                                            hn::TFromD<decltype(di)>>(
                                                            mb.nbins)) +
                                              mres.idx;
                                    }
                                    if constexpr (Poles) hn::Store(mu, d, mubuf);
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
                                    if constexpr (!ScalarTail) {
                                        for (size_t k = 0; k < n; ++k) {
                                            local[static_cast<size_t>(ibuf[k])] +=
                                                static_cast<double>(wbuf[k]);
                                        }
                                    } else {
                                        // Surviving lanes take the shared
                                        // scalar per-pair weighting. Masked
                                        // lanes must be skipped by MASK, not
                                        // weight: a zero weight would still
                                        // subtract the negative product, and
                                        // a NaN projection poisons bin 0.
                                        HWY_ALIGN Float
                                            mbuf[HWY_MAX_LANES_D(decltype(d))];
                                        hn::Store(hn::IfThenElseZero(
                                                      ok, hn::Set(d, Float(1))),
                                                  d, mbuf);
                                        HWY_ALIGN Float
                                            s2buf[HWY_MAX_LANES_D(decltype(d))];
                                        if (sel_s) hn::Store(s2, d, s2buf);
                                        for (size_t k = 0; k < n; ++k) {
                                            if (mbuf[k] == Float(0)) continue;
                                            const size_t jj = j + k;
                                            if (sel_s) {
                                                const double s2k =
                                                    static_cast<double>(s2buf[k]);
                                                if (s2k < sel_s2_min ||
                                                    s2k > sel_s2_max) continue;
                                            }
                                            if (sel_theta) {
                                                const double ct =
                                                    double(r1v[0]) * m2.sx[jj] +
                                                    double(r1v[1]) * m2.sy[jj] +
                                                    double(r1v[2]) * m2.sz[jj];
                                                if (ct < a.sel_ct_min ||
                                                    ct > a.sel_ct_max) continue;
                                            }
                                            const size_t b =
                                                static_cast<size_t>(ibuf[k]);
                                            double w =
                                                static_cast<double>(wbuf[k]);
                                            if (has_bitwise) {
                                                w *= pairmath::pair_bitwise_weight(
                                                    &m1.bw[i * nbit],
                                                    &m2.bw[jj * nbit],
                                                    iv_bw, iv_bw, bitwise);
                                            }
                                            if (has_angular) {
                                                const double ct[1] = {
                                                    double(r1v[0]) * m2.sx[jj] +
                                                    double(r1v[1]) * m2.sy[jj] +
                                                    double(r1v[2]) * m2.sz[jj]};
                                                w *= pairmath::lookup_angular_weight<1>(
                                                    ct, angular);
                                            }
                                            if (has_negative) {
                                                w -= static_cast<double>(m1.nw[i]) *
                                                     static_cast<double>(m2.nw[jj]);
                                            }
                                            // Per-channel pair weights, in
                                            // the CUDA add_weight2 order
                                            // (cross*plus included).
                                            double chan[3] = {w, 0., 0.};
                                            size_t nchan = 1;
                                            if (has_spin) {
                                                const Float r2v[3] = {
                                                    m2.sx[jj], m2.sy[jj],
                                                    m2.sz[jj]};
                                                Float splus1 = 0, scross1 = 0;
                                                Float splus2 = 0, scross2 = 0;
                                                if (sp1) {
                                                    pairmath::compute_spin_projection_cartesian(
                                                        r1v, r2v, e1v,
                                                        a.spin_order1,
                                                        &splus1, &scross1);
                                                }
                                                if (sp2) {
                                                    const Float e2v[2] = {
                                                        m2.e1[jj], m2.e2[jj]};
                                                    pairmath::compute_spin_projection_cartesian(
                                                        r1v, r2v, e2v,
                                                        a.spin_order2,
                                                        &splus2, &scross2);
                                                }
                                                if (sp1 && sp2) {
                                                    chan[0] = w * splus1 * splus2;
                                                    chan[1] = w * scross1 * splus2;
                                                    chan[2] = w * scross1 * scross2;
                                                    nchan = 3;
                                                } else if (sp1) {
                                                    chan[0] = w * splus1;
                                                    chan[1] = w * scross1;
                                                    nchan = 2;
                                                } else {
                                                    chan[0] = w * splus2;
                                                    chan[1] = w * scross2;
                                                    nchan = 2;
                                                }
                                            }

                                            if constexpr (!Poles) {
                                                for (size_t c = 0; c < nchan; ++c)
                                                    local[c * nbins_total + b] += chan[c];
                                            } else {
                                                // (2 ell + 1) P_ell(mu) into
                                                // the nells bins that follow b.
                                                const double muk = mubuf[k];
                                                double leg[MAX_POLE + 1];
                                                pairmath::set_legendre(
                                                    leg, ellmin, ellmax,
                                                    ellstep_legendre, muk,
                                                    muk * muk);
                                                for (size_t ill = 0; ill < nells; ++ill) {
                                                    const int ell = ells[ill];
                                                    const double f =
                                                        (2 * ell + 1) * leg[ell];
                                                    const size_t bl = b * nells + ill;
                                                    for (size_t c = 0; c < nchan; ++c)
                                                        local[c * nbins_total + bl] +=
                                                            chan[c] * f;
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    const auto wv = hn::IfThenElseZero(ok, wpair);
                                    for (size_t b = 0; b < nbins_geom; ++b) {
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
                for (size_t b = 0; b < nbins_geom; ++b) {
                    local[b] += static_cast<double>(
                        hn::ReduceSum(d, hn::Load(d, &acc[b * L])));
                    hn::Store(hn::Zero(d), d, &acc[b * L]);
                }
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        for (size_t b = 0; b < nw * nbins_total; ++b) out[b] += local[b];
    }
}

// Dispatch helpers for the runtime config to compile-time template parameters.
// Highway does the SIMD dispatch at the outermost level; here, we're just dispatching
// on our own config parameters.

template <class Float, int NDim, bool Poles, class SBin, LosKind LOS,
          bool Periodic>
static void DispatchScatter(const Count2Args& a, const Mesh<Float>& m1,
                            const Mesh<Float>& m2, const BinSpec<Float>& sb,
                            const BinSpec<Float>& mb) {
    // Spin/bitwise/negative/angular accumulation is scalar per surviving lane,
    // so BinMajor's replicated histogram has nothing to offer; those requests
    // always take Scalar, with the per-lane tail compiled in (ScalarTail).
    // Multipoles are per-lane by construction, and `if constexpr` keeps the
    // other three combinations out of the Poles build entirely.
    if constexpr (Poles) {
        Count2Impl<Float, NDim, true, SBin, LOS, Periodic, ScatterKind::Scalar,
                   true>(a, m1, m2, sb, mb);
    } else {
        const bool tail = a.spin1 || a.spin2 || (a.bw1 && a.bw2) ||
                          (a.nw1 && a.nw2) || a.angular_weight ||
                          a.sel_s || a.sel_theta;
        if (tail) {
            Count2Impl<Float, NDim, false, SBin, LOS, Periodic,
                       ScatterKind::Scalar, true>(a, m1, m2, sb, mb);
        } else if (a.cfg.scatter == ScatterKind::Scalar) {
            Count2Impl<Float, NDim, false, SBin, LOS, Periodic,
                       ScatterKind::Scalar, false>(a, m1, m2, sb, mb);
        } else {
            Count2Impl<Float, NDim, false, SBin, LOS, Periodic,
                       ScatterKind::BinMajor, false>(a, m1, m2, sb, mb);
        }
    }
}

template <class Float, int NDim, bool Poles, class SBin, LosKind LOS>
static void DispatchPeriodic(const Count2Args& a, const Mesh<Float>& m1,
                             const Mesh<Float>& m2, const BinSpec<Float>& sb,
                             const BinSpec<Float>& mb) {
    if (a.cfg.periodic) {
        DispatchScatter<Float, NDim, Poles, SBin, LOS, true>(a, m1, m2, sb, mb);
    } else {
        DispatchScatter<Float, NDim, Poles, SBin, LOS, false>(a, m1, m2, sb, mb);
    }
}

template <class Float, int NDim, bool Poles, class SBin>
static void DispatchLos(const Count2Args& a, const Mesh<Float>& m1,
                        const Mesh<Float>& m2, const BinSpec<Float>& sb,
                        const BinSpec<Float>& mb) {
    // Without a mu axis and without multipoles there is no mu at all, so
    // AxisZ stands in for every LOS.
    constexpr bool need_mu = (NDim == 2) || Poles;
    if (!need_mu || a.cfg.los == LosKind::AxisZ) {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::AxisZ>(a, m1, m2, sb, mb);
    } else if (a.cfg.los == LosKind::AxisX) {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::AxisX>(a, m1, m2, sb, mb);
    } else if (a.cfg.los == LosKind::AxisY) {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::AxisY>(a, m1, m2, sb, mb);
    } else if (a.cfg.los == LosKind::FirstPoint) {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::FirstPoint>(a, m1, m2, sb, mb);
    } else if (a.cfg.los == LosKind::EndPoint) {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::EndPoint>(a, m1, m2, sb, mb);
    } else {
        DispatchPeriodic<Float, NDim, Poles, SBin, LosKind::Midpoint>(a, m1, m2, sb, mb);
    }
}

template <class Float, int NDim, bool Poles>
static void DispatchSBin(const Count2Args& a, const Mesh<Float>& m1,
                         const Mesh<Float>& m2, const BinSpec<Float>& sb,
                         const BinSpec<Float>& mb) {
    switch (a.cfg.sbin) {
        case BinKind::Linear:
            DispatchLos<Float, NDim, Poles, BinLinear>(a, m1, m2, sb, mb);
            break;
        case BinKind::Log:
            DispatchLos<Float, NDim, Poles, BinLog>(a, m1, m2, sb, mb);
            break;
        case BinKind::Edges:
            DispatchLos<Float, NDim, Poles, BinEdges>(a, m1, m2, sb, mb);
            break;
    }
}


// Multipoles ride the s axis only ((s, pole)); the binding declines any other
// combination, so Poles=true is never instantiated for NDim == 2.
template <class Float, int NDim>
static void DispatchPoles(const Count2Args& a, const Mesh<Float>& m1,
                          const Mesh<Float>& m2, const BinSpec<Float>& sb,
                          const BinSpec<Float>& mb) {
    if constexpr (NDim == 1) {
        if (a.nells) {
            DispatchSBin<Float, NDim, true>(a, m1, m2, sb, mb);
            return;
        }
    }
    DispatchSBin<Float, NDim, false>(a, m1, m2, sb, mb);
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

    // One dims for both meshes: the kernel walks m1's cell coordinates on
    // m2's grid. The mesh is the one MeshAttrs chose, the same object the CUDA
    // backend is given; mesh_dims is only the fallback for the raw-array entry
    // point, which has no MeshAttrs to take one from.
    int dims[3];
    if (a.meshsize[0]) {
        for (int axis = 0; axis < 3; ++axis)
            dims[axis] = static_cast<int>(a.meshsize[axis]);
    }
    else {
        mesh_dims(a.boxsize, smax, (a.n1 + a.n2) / 2, dims);
    }
    // Every mesh gives the same counts, so which one ran is invisible in the
    // result and only this says so.
    log_message(LOG_LEVEL_DEBUG, "cpu kernel: mesh %d x %d x %d (%s).\n",
                dims[0], dims[1], dims[2],
                a.meshsize[0] ? "from MeshAttrs" : "no MeshAttrs, own heuristic");

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const bool needs_mu = (a.cfg.ndim == 2) || a.nells;
    const bool with_spos =
        (a.spin1 != nullptr) || (a.spin2 != nullptr) ||
        (a.angular_weight != nullptr) || a.sel_theta ||
        (needs_mu && (a.cfg.los == LosKind::FirstPoint ||
                      a.cfg.los == LosKind::EndPoint));
    const Mesh<Float> m1 = build_mesh<Float>(a.pos1, a.w1, a.n1, a.boxsize,
                                             a.origin, dims, a.spin1, with_spos,
                                             a.bw1, a.nbitwise, a.nw1);
    const Mesh<Float> m2 = build_mesh<Float>(a.pos2, a.w2, a.n2, a.boxsize,
                                             a.origin, dims, a.spin2, with_spos,
                                             a.bw2, a.nbitwise, a.nw2);
    const auto t1 = Clock::now();

    if (a.cfg.ndim == 1) {
        DispatchPoles<Float, 1>(a, m1, m2, sb, mb);
    } else {
        DispatchPoles<Float, 2>(a, m1, m2, sb, mb);
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
