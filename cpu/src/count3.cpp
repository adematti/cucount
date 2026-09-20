// Factorized triplet counts: see cpu/include/count3.h.
//
// A port of cuda/src/count3.cu. The per-primary histograms, the projection
// onto the local frame's real spherical harmonics and the contraction over m
// all follow the CUDA original line for line; the shared math comes from
// include/cmath.h and the shared output layout from include/layout.h.

#include "count3.h"
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

// One leg's per-primary histogram: which separation bin this secondary falls
// in, and either its plain weight or its projection onto the harmonics of the
// primary's local frame. `seen` marks the bins that were touched, so the outer
// product below can skip the empty ones -- the histograms are sparse in
// practice and this is where the CUDA kernel spends its time.
void add_pair_weight(double* hist, unsigned char* seen,
                     const double local_frame[3][3],
                     const double* sposition1, const double* sposition2,
                     const double* position1, const double* position2,
                     const double* value2, const IndexValue& index_value2,
                     const BinAttrs& battrs, size_t nprojs, size_t nells,
                     const size_t* ells, int ellmax, const MeshAttrs& mattrs) {
    if (battrs.ndim == 0) return;

    const bool need_pole = (nprojs > 0);

    const double costheta = clamp_unit(dot3(sposition1, sposition2));

    double diff[NDIM];
    double r = 0.;
    double value;

    const VAR_TYPE var = battrs.var[0];

    if (var == VAR_S || var == VAR_POLE) {
        difference(diff, position2, position1, mattrs);
        r = std::sqrt(dot3(diff, diff));
        value = r;
    }
    else if (var == VAR_THETA) {
        value = std::acos(costheta) / DTORAD;
    }
    else return;

    const int ibin = get_sep_bin_index(value, battrs.array[0],
                                       static_cast<int>(battrs.shape[0]),
                                       battrs.bin[0], true);
    if (ibin < 0) return;

    double weight = 1.;
    if (index_value2.size_individual_weight)
        weight *= value2[index_value2.start_individual_weight];
    if (index_value2.size_negative_weight)
        weight -= value2[index_value2.start_negative_weight];
    if (weight == 0.) return;

    seen[ibin] = 1;

    if (!need_pole) {
        hist[ibin] += weight;
        return;
    }

    // A coincident secondary has no direction of its own; the CUDA kernel
    // points it along the frame's own axis rather than dividing by zero.
    double rhat[NDIM];
    if (r == 0.) {
        for (int icoord = 0; icoord < NDIM; icoord++) rhat[icoord] = local_frame[0][icoord];
    }
    else {
        for (int icoord = 0; icoord < NDIM; icoord++) rhat[icoord] = diff[icoord] / r;
    }

    const double* ez = local_frame[0];
    const double* ex = local_frame[1];
    const double* ey = local_frame[2];

    double mu = 0., x = 0., y = 0.;
    for (int icoord = 0; icoord < NDIM; icoord++) {
        mu += rhat[icoord] * ez[icoord];
        x += rhat[icoord] * ex[icoord];
        y += rhat[icoord] * ey[icoord];
    }
    mu = clamp_unit(mu);

    const double rho = std::sqrt(std::max(0., x * x + y * y));
    double cphi = 1., sphi = 0.;
    if (rho > 1e-12) {
        cphi = x / rho;
        sphi = y / rho;
    }

    double cm[MMAX_SIZE], sm[MMAX_SIZE];
    compute_trig_up_to_m(ellmax, cphi, sphi, cm, sm);

    double P[MMAX_SIZE][MMAX_SIZE];
    compute_pbar_all_lmax5(ellmax, mu, P);

    double* hist_bin = hist + static_cast<size_t>(ibin) * nprojs;

    size_t iproj = 0;
    for (size_t iell = 0; iell < nells; iell++) {
        const int ell = static_cast<int>(ells[iell]);
        const int mmax = ell;

        for (int m = 0; m <= mmax; m++) {
            hist_bin[iproj + static_cast<size_t>(m)] += weight * P[ell][m] * cm[m];
            if (m > 0)
                hist_bin[iproj + static_cast<size_t>(mmax + m)] += weight * P[ell][m] * sm[m];
        }

        iproj += static_cast<size_t>(2 * mmax + 1);
    }
}

// One leg's candidate sweep, with its selection and its veto.
template <class Op>
void sweep_leg(const MeshAttrs& mattrs, const ScalarMesh& mesh,
               const double* position1, const double* sposition1,
               const SelectionAttrs& sattrs, const SelectionAttrs& veto, Op&& op) {
    for_each_candidate(mattrs, mesh, position1, sposition1, [&](size_t j) {
        const double* position = mesh.position(j);
        const double* sposition = mesh.sposition(j);

        if (!is_selected_pair(sposition1, sposition, position1, position, sattrs, mattrs))
            return;
        if (veto.ndim && is_selected_pair(sposition1, sposition, position1, position,
                                          veto, mattrs))
            return;

        op(j, position, sposition, mesh.value(j));
    });
}

}  // namespace

void Count3(const Count3Args& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const ScalarMesh m1 = build_mesh(args.p1, args.attrs.mattrs1);
    const ScalarMesh m2 = build_mesh(args.p2, args.attrs.mattrs2);
    const ScalarMesh m3 = build_mesh(args.p3, args.attrs.mattrs3);

    const auto t1 = Clock::now();

    BinAttrs battrs23{};
    const Count3PoleLayout layout =
        make_count3_pole_layout(args.attrs.battrs12, args.attrs.battrs13, battrs23);

    const size_t nbin12 = args.attrs.battrs12.shape[0];
    const size_t nbin13 = args.attrs.battrs13.shape[0];
    const size_t hsize2 = nbin12 * (layout.nprojs1 ? layout.nprojs1 : 1);
    const size_t hsize3 = nbin13 * (layout.nprojs2 ? layout.nprojs2 : 1);
    const size_t csize = layout.csize;

    const LOS_TYPE los = get_count3_los(args.attrs.battrs12, args.attrs.battrs13);
    const long total1 = static_cast<long>(m1.total);

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads)
#endif
    {
        std::vector<double> local(csize, 0.);
        std::vector<double> hist2(hsize2), hist3(hsize3);
        std::vector<unsigned char> seen2(nbin12), seen3(nbin13);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
        for (long i1 = 0; i1 < total1; i1++) {
            const double* position1 = m1.position(i1);
            const double* sposition1 = m1.sposition(i1);
            const double* value1 = m1.value(i1);

            double w1 = 1.;
            if (m1.iv.size_individual_weight)
                w1 *= value1[m1.iv.start_individual_weight];
            if (m1.iv.size_negative_weight)
                w1 -= value1[m1.iv.start_negative_weight];
            if (w1 == 0.) continue;

            std::fill(hist2.begin(), hist2.end(), 0.);
            std::fill(hist3.begin(), hist3.end(), 0.);
            std::fill(seen2.begin(), seen2.end(), 0);
            std::fill(seen3.begin(), seen3.end(), 0);

            double local_frame[3][3];
            build_los_frame(sposition1, los, local_frame);

            sweep_leg(args.attrs.mattrs2, m2, position1, sposition1, args.attrs.sattrs12, args.attrs.veto12,
                      [&](size_t, const double* position, const double* sposition,
                          const double* value) {
                          add_pair_weight(hist2.data(), seen2.data(), local_frame,
                                          sposition1, sposition, position1, position,
                                          value, m2.iv, args.attrs.battrs12, layout.nprojs1,
                                          layout.nells1, layout.ells1, layout.ellmax1,
                                          args.attrs.mattrs2);
                      });

            sweep_leg(args.attrs.mattrs3, m3, position1, sposition1, args.attrs.sattrs13, args.attrs.veto13,
                      [&](size_t, const double* position, const double* sposition,
                          const double* value) {
                          add_pair_weight(hist3.data(), seen3.data(), local_frame,
                                          sposition1, sposition, position1, position,
                                          value, m3.iv, args.attrs.battrs13, layout.nprojs2,
                                          layout.nells2, layout.ells2, layout.ellmax2,
                                          args.attrs.mattrs3);
                      });

            if (layout.nprojs == 0) {
                for (size_t ibin12 = 0; ibin12 < nbin12; ibin12++) {
                    if (!seen2[ibin12]) continue;
                    const double w2 = hist2[ibin12];

                    for (size_t ibin13 = 0; ibin13 < nbin13; ibin13++) {
                        if (!seen3[ibin13]) continue;
                        local[ibin12 * nbin13 + ibin13] += w1 * w2 * hist3[ibin13];
                    }
                }
                continue;
            }

            // Contract the two projections over m: the real and imaginary
            // parts recombine as c2 c3 + s2 s3 and s2 c3 - c2 s3, and only
            // m <= min(ell1, ell2) survives.
            size_t iproj = 0;
            size_t iproj1 = 0;

            for (size_t iell1 = 0; iell1 < layout.nells1; iell1++) {
                const int ell1 = static_cast<int>(layout.ells1[iell1]);
                size_t iproj2 = 0;

                for (size_t iell2 = 0; iell2 < layout.nells2; iell2++) {
                    const int ell2 = static_cast<int>(layout.ells2[iell2]);
                    const int mmax = std::min(ell1, ell2);

                    const double ell_norm_w1 =
                        std::sqrt(static_cast<double>((2 * ell1 + 1) * (2 * ell2 + 1))) * w1;

                    for (size_t ibin12 = 0; ibin12 < nbin12; ibin12++) {
                        if (!seen2[ibin12]) continue;
                        const double* hist2_bin = hist2.data() + ibin12 * layout.nprojs1;

                        for (size_t ibin13 = 0; ibin13 < nbin13; ibin13++) {
                            if (!seen3[ibin13]) continue;
                            const double* hist3_bin = hist3.data() + ibin13 * layout.nprojs2;

                            double* counts_bin =
                                local.data() + (ibin12 * nbin13 + ibin13) * layout.nprojs;

                            counts_bin[iproj] +=
                                ell_norm_w1 * hist2_bin[iproj1] * hist3_bin[iproj2];

                            for (int m = 1; m <= mmax; m++) {
                                const double c2 = hist2_bin[iproj1 + m];
                                const double s2 = hist2_bin[iproj1 + ell1 + m];
                                const double c3 = hist3_bin[iproj2 + m];
                                const double s3 = hist3_bin[iproj2 + ell2 + m];

                                counts_bin[iproj + m] += ell_norm_w1 * (c2 * c3 + s2 * s3);
                                counts_bin[iproj + mmax + m] +=
                                    ell_norm_w1 * (s2 * c3 - c2 * s3);
                            }
                        }
                    }

                    iproj += static_cast<size_t>(2 * mmax + 1);
                    iproj2 += static_cast<size_t>(2 * ell2 + 1);
                }

                iproj1 += static_cast<size_t>(2 * ell1 + 1);
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
