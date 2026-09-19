// Close triplet counts: see cpu/include/cucount/cpu/triplet.h.
//
// A port of add_weight3 and the close_pair == (1, 2) traversal in
// cuda/src/count3close.cu. The CUDA backend's other two strategies enumerate
// the same triplets in a different loop order, so this one implementation
// serves every close_pair.

#include "cucount/cpu/triplet.h"
#include "cucount/cpu/walk.h"
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
                 const Count3ProjLayout& layout) {
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

    double triplet_weight = 1.;

    if (index_value1.size_individual_weight)
        triplet_weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight)
        triplet_weight *= value2[index_value2.start_individual_weight];
    if (index_value3.size_individual_weight)
        triplet_weight *= value3[index_value3.start_individual_weight];

    // The 3-dimensional angular upweight, indexed by the triangle's three
    // cos(theta). This is the only place an N-dimensional angular table is
    // used on either backend.
    if (wattrs.angular.size)
        triplet_weight *= lookup_angular_weight<3>(costheta, wattrs.angular);

    if (index_value1.size_negative_weight && index_value2.size_negative_weight &&
        index_value3.size_negative_weight) {
        triplet_weight -= value1[index_value1.start_negative_weight] *
                          value2[index_value2.start_negative_weight] *
                          value3[index_value3.start_negative_weight];
    }

    if (has_third || !need_pole) {
        counts[ibin] += triplet_weight;
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
        sdphi = clamp_unit((xy[0][0] * xy[1][1] - xy[0][1] * xy[1][0]) * inv);
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
                const double amp = triplet_weight * ell_norm * P1[ell1][m] * P2[ell2][m];
                counts_bin[iproj + static_cast<size_t>(m)] += amp * cm[m];
                if (m > 0)
                    counts_bin[iproj + static_cast<size_t>(mmax + m)] += amp * sm[m];
            }

            iproj += static_cast<size_t>(2 * mmax + 1);
        }
    }
}

}  // namespace

void Count3Close(const Count3CloseArgs& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const ScalarMesh m1 = build_mesh(args.p1, args.mattrs1);
    const ScalarMesh m2 = build_mesh(args.p2, args.mattrs2);
    const ScalarMesh m3 = build_mesh(args.p3, args.mattrs3);

    const auto t1 = Clock::now();

    const Count3ProjLayout layout =
        make_count3_proj_layout(args.battrs12, args.battrs13, args.battrs23);
    const size_t csize = layout.csize;

    const LOS_TYPE los = get_count3_los(args.battrs12, args.battrs13);
    const long total1 = static_cast<long>(m1.total);

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads)
#endif
    {
        std::vector<double> local(csize, 0.);
        std::vector<size_t> cand3;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (long i1 = 0; i1 < total1; i1++) {
            const double* position1 = m1.position(i1);
            const double* sposition1 = m1.sposition(i1);
            const double* value1 = m1.value(i1);

            double local_frame[3][3];
            build_los_frame(sposition1, los, local_frame);

            // Leg 3's candidates, and the (1, 3) tests, depend only on the
            // primary, so they are gathered once here rather than re-walked
            // for every candidate of leg 2. The visit order is unchanged, so
            // the accumulation order is too. Centred on particle 1, like the
            // CUDA traversal: the (1, 3) window bounds this leg, not (2, 3).
            cand3.clear();
            for_each_candidate(args.mattrs3, m3, position1, sposition1, [&](size_t i3) {
                const double* position3 = m3.position(i3);
                const double* sposition3 = m3.sposition(i3);

                if (!is_selected_pair(sposition1, sposition3, position1, position3,
                                      args.sattrs13, args.mattrs3)) return;
                if (args.veto13.ndim &&
                    is_selected_pair(sposition1, sposition3, position1, position3,
                                     args.veto13, args.mattrs3)) return;
                cand3.push_back(i3);
            });
            if (cand3.empty()) continue;

            for_each_candidate(args.mattrs2, m2, position1, sposition1, [&](size_t i2) {
                const double* position2 = m2.position(i2);
                const double* sposition2 = m2.sposition(i2);
                const double* value2 = m2.value(i2);

                if (!is_selected_pair(sposition1, sposition2, position1, position2,
                                      args.sattrs12, args.mattrs2)) return;
                if (args.veto12.ndim &&
                    is_selected_pair(sposition1, sposition2, position1, position2,
                                     args.veto12, args.mattrs2)) return;

                for (size_t i3 : cand3) {
                    const double* position3 = m3.position(i3);
                    const double* sposition3 = m3.sposition(i3);
                    const double* value3 = m3.value(i3);

                    if (!is_selected_pair(sposition2, sposition3, position2, position3,
                                          args.sattrs23, args.mattrs3)) continue;
                    if (args.veto23.ndim &&
                        is_selected_pair(sposition2, sposition3, position2, position3,
                                         args.veto23, args.mattrs3)) continue;

                    add_weight3(local.data(), local_frame, sposition1, sposition2,
                                sposition3, position1, position2, position3,
                                value1, value2, value3, m1.iv, m2.iv, m3.iv,
                                args.mattrs2, args.mattrs3, args.battrs12,
                                args.battrs13, args.battrs23, args.wattrs, layout);
                }
            });
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
