// Close triplet counts: see cpu/include/count3.h.
//
// A port of add_weight3 and of the traversals in cuda/src/count3close.cu:
// from each particle 1 for close_pair (1, 2) and (1, 3), from each particle 2
// for (2, 3). All enumerate the same triplets; the meshes must be sized for
// the strategy (see count3close.h).

#include "count3close.h"
#include "mesh.h"
#include "layout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cucount {
namespace cpu {
namespace {

using pairmath::build_los_frame;
using pairmath::clamp_unit;
using pairmath::compute_pbar_all_lmax5;
using pairmath::compute_trig_up_to_m;
using pairmath::dot3;
using pairmath::get_count3_los;
using pairmath::get_sep_bin_index;
using pairmath::lookup_angular_weight;

// The weight of a triplet: its individual weights, the 3-dimensional angular
// upweight, indexed by the triangle's three cos(theta) -- the only place an
// N-dimensional angular table is used on either backend -- and the negative
// weights. Shared by add_weight3 and the fast path of Count3Close.
inline double triplet_weight(const double* value1, const double* value2, const double* value3,
                             const IndexValue& index_value1, const IndexValue& index_value2,
                             const IndexValue& index_value3, const double (&costheta)[3],
                             const WeightAttrs& wattrs) {
    double weight = 1.;
    if (index_value1.size_individual_weight)
        weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight)
        weight *= value2[index_value2.start_individual_weight];
    if (index_value3.size_individual_weight)
        weight *= value3[index_value3.start_individual_weight];
    if (wattrs.angular.size)
        weight *= lookup_angular_weight<3>(costheta, wattrs.angular);
    if (index_value1.size_negative_weight && index_value2.size_negative_weight &&
        index_value3.size_negative_weight) {
        weight -= value1[index_value1.start_negative_weight] *
                  value2[index_value2.start_negative_weight] *
                  value3[index_value3.start_negative_weight];
    }
    return weight;
}

// One leg of a triplet: its cos(theta), which the angular weight needs on
// every leg, and its bin, -1 outside the binning (or for a variable
// add_weight3 does not bin), 0 for a leg with no axis. As add_weight3.
struct Leg {
    int bin = 0;
    double costheta = 1.;
};

inline Leg make_leg(const BinAttrs& battrs, bool binned, const MeshAttrs& mattrs,
                    const double* sposition1, const double* sposition2,
                    const double* position1, const double* position2) {
    Leg leg;
    leg.costheta = clamp_unit(dot3(sposition1, sposition2));
    if (!binned) return leg;
    const VAR_TYPE var = battrs.var[0];
    double value;
    if (var == VAR_S || var == VAR_POLE) {
        double diff[NDIM];
        difference(diff, position2, position1, mattrs);
        value = std::sqrt(dot3(diff, diff));
    }
    else if (var == VAR_THETA) {
        value = std::acos(leg.costheta) / DTORAD;
    }
    else {
        leg.bin = -1;
        return leg;
    }
    leg.bin = get_sep_bin_index(value, battrs.array[0], static_cast<int>(battrs.shape[0]),
                                battrs.bin[0], true);
    return leg;
}

// Port of add_weight3. The three legs are addressed uniformly as
// (1, 2), (1, 3), (2, 3), which is what lets one loop bin all of them and
// what the 3-dimensional angular table is indexed by.
void add_weight3(double* counts, const double local_frame[3][3],
                 const double* sposition1, const double* sposition2,
                 const double* sposition3,
                 const double* position1, const double* position2,
                 const double* position3,
                 const double* value1, const double* value2, const double* value3,
                 const IndexValue& index_value1, const IndexValue& index_value2,
                 const IndexValue& index_value3,
                 const MeshAttrs& mattrs2, const MeshAttrs& mattrs3,
                 const BinAttrs& battrs12, const BinAttrs& battrs13,
                 const BinAttrs& battrs23, const WeightAttrs& wattrs,
                 const Count3PoleLayout& layout) {
    if (battrs12.ndim == 0 || battrs13.ndim == 0) return;
    const bool has_third = (battrs23.ndim > 0);
    const int ncoords = has_third ? 3 : 2;
    const bool need_pole = (layout.nprojs > 0);

    const double* spos1[3] = {sposition1, sposition1, sposition2};
    const double* spos2[3] = {sposition2, sposition3, sposition3};
    const double* pos1[3] = {position1, position1, position2};
    const double* pos2[3] = {position2, position3, position3};

    const BinAttrs* battrs[3] = {&battrs12, &battrs13, &battrs23};
    const MeshAttrs* mattrs[3] = {&mattrs2, &mattrs3, &mattrs3};

    double costheta[3];
    double diff[3][NDIM];
    double r[3] = {0., 0., 0.};

    for (int icoord = 0; icoord < 3; icoord++)
        costheta[icoord] = clamp_unit(dot3(spos1[icoord], spos2[icoord]));

    size_t ibin = 0;

    for (int icoord = 0; icoord < 3; icoord++) {
        if (icoord >= ncoords) continue;

        const BinAttrs* battr = battrs[icoord];
        const VAR_TYPE var = battr->var[0];
        double value;

        if (var == VAR_S || var == VAR_POLE) {
            difference(diff[icoord], pos2[icoord], pos1[icoord], *mattrs[icoord]);
            r[icoord] = std::sqrt(dot3(diff[icoord], diff[icoord]));
            value = r[icoord];
        }
        else if (var == VAR_THETA) {
            value = std::acos(costheta[icoord]) / DTORAD;
        }
        else return;

        const int ib = get_sep_bin_index(value, battr->array[0],
                                         static_cast<int>(battr->shape[0]),
                                         battr->bin[0], true);
        if (ib < 0) return;

        ibin = ibin * battr->shape[0] + static_cast<size_t>(ib);
    }

    const double weight = triplet_weight(value1, value2, value3, index_value1, index_value2,
                                         index_value3, costheta, wattrs);

    if (has_third || !need_pole) {
        counts[ibin] += weight;
        return;
    }

    double rhat[2][NDIM];
    for (int ivec = 0; ivec < 2; ivec++) {
        if (r[ivec] == 0.) {
            for (int icoord = 0; icoord < NDIM; icoord++)
                rhat[ivec][icoord] = local_frame[0][icoord];
        }
        else {
            for (int icoord = 0; icoord < NDIM; icoord++)
                rhat[ivec][icoord] = diff[ivec][icoord] / r[ivec];
        }
    }

    const double* ez = local_frame[0];
    const double* ex = local_frame[1];
    const double* ey = local_frame[2];

    double mu[2] = {0., 0.};
    for (int ivec = 0; ivec < 2; ivec++) {
        for (int icoord = 0; icoord < NDIM; icoord++)
            mu[ivec] += rhat[ivec][icoord] * ez[icoord];
        mu[ivec] = clamp_unit(mu[ivec]);
    }

    double xy[2][2] = {{0., 0.}, {0., 0.}};
    for (int ivec = 0; ivec < 2; ivec++) {
        for (int icoord = 0; icoord < NDIM; icoord++) {
            xy[ivec][0] += rhat[ivec][icoord] * ex[icoord];
            xy[ivec][1] += rhat[ivec][icoord] * ey[icoord];
        }
    }

    // The azimuthal separation enters only as cos and sin of m (phi1 - phi2),
    // so it is taken from the dot and cross of the two transverse parts rather
    // than from two arctangents.
    double rho[2];
    for (int ivec = 0; ivec < 2; ivec++)
        rho[ivec] = std::sqrt(std::max(0., 1. - mu[ivec] * mu[ivec]));

    double cdphi = 1., sdphi = 0.;
    if (rho[0] > 1e-12 && rho[1] > 1e-12) {
        const double inv = 1. / (rho[0] * rho[1]);
        cdphi = clamp_unit((xy[0][0] * xy[1][0] + xy[0][1] * xy[1][1]) * inv);
        // The cross product is oriented (1, 2) x (1, 3), so that sdphi is
        // sin(phi12 - phi13) and compute_trig_up_to_m raises exp(+i dphi).
        // Taken the other way round it raises the conjugate, which leaves the
        // real coefficients untouched -- cos is even -- and negates every
        // imaginary one against count3.
        sdphi = clamp_unit((xy[0][1] * xy[1][0] - xy[0][0] * xy[1][1]) * inv);
    }

    const int global_mmax = std::min(layout.ellmax1, layout.ellmax2);

    double cm[MMAX_SIZE], sm[MMAX_SIZE];
    compute_trig_up_to_m(global_mmax, cdphi, sdphi, cm, sm);

    double P1[MMAX_SIZE][MMAX_SIZE];
    double P2[MMAX_SIZE][MMAX_SIZE];
    compute_pbar_all_lmax5(layout.ellmax1, mu[0], P1);
    compute_pbar_all_lmax5(layout.ellmax2, mu[1], P2);

    double* counts_bin = counts + ibin * layout.nprojs;

    size_t iproj = 0;
    for (size_t i1 = 0; i1 < layout.nells1; i1++) {
        const int ell1 = static_cast<int>(layout.ells1[i1]);

        for (size_t i2 = 0; i2 < layout.nells2; i2++) {
            const int ell2 = static_cast<int>(layout.ells2[i2]);
            const int mmax = std::min(ell1, ell2);

            const double ell_norm =
                std::sqrt(static_cast<double>((2 * ell1 + 1) * (2 * ell2 + 1)));

            for (int m = 0; m <= mmax; m++) {
                const double amp = weight * ell_norm * P1[ell1][m] * P2[ell2][m];
                counts_bin[iproj + static_cast<size_t>(m)] += amp * cm[m];
                if (m > 0)
                    counts_bin[iproj + static_cast<size_t>(mmax + m)] += amp * sm[m];
            }

            iproj += static_cast<size_t>(2 * mmax + 1);
        }
    }
}

}  // namespace

void Count3Close(const Count3Args& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const ScalarMesh m1 = build_mesh(args.p1, args.attrs.mattrs1);
    const ScalarMesh m2 = build_mesh(args.p2, args.attrs.mattrs2);
    const ScalarMesh m3 = build_mesh(args.p3, args.attrs.mattrs3);

    const auto t1 = Clock::now();

    const Count3PoleLayout layout =
        make_count3_pole_layout(args.attrs.battrs12, args.attrs.battrs13, args.attrs.battrs23);
    const size_t csize = layout.csize;

    const LOS_TYPE los = get_count3_los(args.attrs.battrs12, args.attrs.battrs13);
    const Count3Attrs& at = args.attrs;
    // add_weight3 counts nothing without both of these axes.
    if (at.battrs12.ndim == 0 || at.battrs13.ndim == 0) return;

    // The (1, 2) and (1, 3) tests, and the (2, 3) ones, shared by every strategy.
    auto keep12 = [&](const double* s1, const double* s2, const double* p1, const double* p2) {
        if (!is_selected_pair(s1, s2, p1, p2, at.sattrs12, at.mattrs2)) return false;
        return !(at.veto12.ndim && is_selected_pair(s1, s2, p1, p2, at.veto12, at.mattrs2));
    };
    auto keep13 = [&](const double* s1, const double* s3, const double* p1, const double* p3) {
        if (!is_selected_pair(s1, s3, p1, p3, at.sattrs13, at.mattrs3)) return false;
        return !(at.veto13.ndim && is_selected_pair(s1, s3, p1, p3, at.veto13, at.mattrs3));
    };
    auto keep23 = [&](const double* s2, const double* s3, const double* p2, const double* p3) {
        if (!is_selected_pair(s2, s3, p2, p3, at.sattrs23, at.mattrs3)) return false;
        return !(at.veto23.ndim && is_selected_pair(s2, s3, p2, p3, at.veto23, at.mattrs3));
    };

    // Without multipoles (no pole axis, or a (2, 3) axis, which add_weight3
    // bins without them), a triplet is its three legs' bins and cos(theta),
    // and each leg only depends on its two ends: it is computed once, where
    // the walk fixes them, rather than once per triplet, and a pair outside
    // the binning is dropped there. Otherwise every triplet goes through
    // add_weight3, which needs the full geometry for the poles.
    const bool has_third = (at.battrs23.ndim > 0);
    const bool fast = has_third || (layout.nprojs == 0);
    const size_t n13 = at.battrs13.shape[0], n23 = has_third ? at.battrs23.shape[0] : 1;
    auto leg12 = [&](size_t i1, size_t i2) { return make_leg(at.battrs12, true, at.mattrs2, m1.sposition(i1), m2.sposition(i2), m1.position(i1), m2.position(i2)); };
    auto leg13 = [&](size_t i1, size_t i3) { return make_leg(at.battrs13, true, at.mattrs3, m1.sposition(i1), m3.sposition(i3), m1.position(i1), m3.position(i3)); };
    auto leg23 = [&](size_t i2, size_t i3) { return make_leg(at.battrs23, has_third, at.mattrs3, m2.sposition(i2), m3.sposition(i3), m2.position(i2), m3.position(i3)); };
    auto emit_legs = [&](double* local, const Leg& l12, const Leg& l13, const Leg& l23, size_t i1, size_t i2, size_t i3) {
        if (l23.bin < 0) return;
        const size_t ibin = ((static_cast<size_t>(l12.bin) * n13) + static_cast<size_t>(l13.bin)) * n23 + static_cast<size_t>(l23.bin);
        const double costheta[3] = {l12.costheta, l13.costheta, l23.costheta};
        local[ibin] += triplet_weight(m1.value(i1), m2.value(i2), m3.value(i3), m1.iv, m2.iv, m3.iv, costheta, at.wattrs);
    };
    auto emit = [&](double* local, const double local_frame[3][3], size_t i1, size_t i2, size_t i3) {
        add_weight3(local, local_frame, m1.sposition(i1), m2.sposition(i2), m3.sposition(i3),
                    m1.position(i1), m2.position(i2), m3.position(i3),
                    m1.value(i1), m2.value(i2), m3.value(i3), m1.iv, m2.iv, m3.iv,
                    at.mattrs2, at.mattrs3, at.battrs12, at.battrs13, at.battrs23, at.wattrs, layout);
    };

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads)
#endif
    {
        std::vector<double> local(csize, 0.);
        // Each candidate with its leg to the particle the walk is centred on
        // (unused, and left default, outside the fast path).
        std::vector<std::pair<size_t, Leg>> close, cand3;

        if (args.close_pair == 23) {
            // Walk from each particle 2, as CUDA does: its close particles 3
            // first, in the (2, 3) window, then particle 1 in the (1, 2) one.
            // Centred on particle 1 instead, the (1, 3) leg -- often unbounded
            // here -- made every primary sweep the whole of catalogue 3.
            // The frame is particle 1's, so it is built per particle 1, and
            // only when a pole needs it.
            const bool need_frame = !fast;
            const long total2 = static_cast<long>(m2.total);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
            for (long i2 = 0; i2 < total2; i2++) {
                const double* position2 = m2.position(i2);
                const double* sposition2 = m2.sposition(i2);

                close.clear();
                for_each_candidate(at.mattrs3, m3, position2, sposition2, [&](size_t i3) {
                    if (!keep23(sposition2, m3.sposition(i3), position2, m3.position(i3))) return;
                    Leg l23;
                    if (fast && (l23 = leg23(i2, i3)).bin < 0) return;
                    close.emplace_back(i3, l23);
                });
                if (close.empty()) continue;

                for_each_candidate(at.mattrs1, m1, position2, sposition2, [&](size_t i1) {
                    const double* position1 = m1.position(i1);
                    const double* sposition1 = m1.sposition(i1);
                    if (!keep12(sposition1, sposition2, position1, position2)) return;
                    if (fast) {
                        const Leg l12 = leg12(i1, i2);
                        if (l12.bin < 0) return;
                        for (const auto& [i3, l23] : close) {
                            if (!keep13(sposition1, m3.sposition(i3), position1, m3.position(i3))) continue;
                            const Leg l13 = leg13(i1, i3);
                            if (l13.bin < 0) continue;
                            emit_legs(local.data(), l12, l13, l23, i1, i2, i3);
                        }
                        return;
                    }
                    double local_frame[3][3] = {};
                    if (need_frame) build_los_frame(sposition1, los, local_frame);
                    for (const auto& [i3, l23] : close) {
                        if (!keep13(sposition1, m3.sposition(i3), position1, m3.position(i3))) continue;
                        emit(local.data(), local_frame, i1, i2, i3);
                    }
                });
            }
        }
        else {
            const long total1 = static_cast<long>(m1.total);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
            for (long i1 = 0; i1 < total1; i1++) {
                const double* position1 = m1.position(i1);
                const double* sposition1 = m1.sposition(i1);

                double local_frame[3][3] = {};
                if (!fast) build_los_frame(sposition1, los, local_frame);

                // Leg 3's candidates, and the (1, 3) tests, depend only on the
                // primary, so they are gathered once here rather than re-walked
                // for every candidate of leg 2. Centred on particle 1, like the
                // CUDA traversal: the (1, 3) window bounds this leg, not (2, 3).
                auto gather3 = [&]() {
                    cand3.clear();
                    for_each_candidate(at.mattrs3, m3, position1, sposition1, [&](size_t i3) {
                        if (!keep13(sposition1, m3.sposition(i3), position1, m3.position(i3))) return;
                        Leg l13;
                        if (fast && (l13 = leg13(i1, i3)).bin < 0) return;
                        cand3.emplace_back(i3, l13);
                    });
                };
                // The triplets of particle 1 with particle 2 and its leg.
                auto triplets = [&](size_t i2, const Leg& l12) {
                    for (const auto& [i3, l13] : cand3) {
                        if (!keep23(m2.sposition(i2), m3.sposition(i3), m2.position(i2), m3.position(i3))) continue;
                        if (fast) emit_legs(local.data(), l12, l13, leg23(i2, i3), i1, i2, i3);
                        else emit(local.data(), local_frame, i1, i2, i3);
                    }
                };

                if (args.close_pair == 12) {
                    // The close particles 2 first: most primaries have none,
                    // and then the wide (1, 3) window is never swept. The
                    // visit order is that of the walk below, so the
                    // accumulation order is unchanged.
                    close.clear();
                    for_each_candidate(at.mattrs2, m2, position1, sposition1, [&](size_t i2) {
                        if (!keep12(sposition1, m2.sposition(i2), position1, m2.position(i2))) return;
                        Leg l12;
                        if (fast && (l12 = leg12(i1, i2)).bin < 0) return;
                        close.emplace_back(i2, l12);
                    });
                    if (close.empty()) continue;
                    gather3();
                    for (const auto& [i2, l12] : close) triplets(i2, l12);
                    continue;
                }

                gather3();
                if (cand3.empty()) continue;
                for_each_candidate(at.mattrs2, m2, position1, sposition1, [&](size_t i2) {
                    if (!keep12(sposition1, m2.sposition(i2), position1, m2.position(i2))) return;
                    Leg l12;
                    if (fast && (l12 = leg12(i1, i2)).bin < 0) return;
                    triplets(i2, l12);
                });
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        for (size_t b = 0; b < csize; b++) args.out[b] += local[b];
    }

    if (args.timings) {
        using Sec = std::chrono::duration<double>;
        args.timings[0] = Sec(t1 - t0).count();
        args.timings[1] = Sec(Clock::now() - t1).count();
    }
}

}  // namespace cpu
}  // namespace cucount
