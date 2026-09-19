// Generic scalar count2: see cpu/include/cucount/cpu/generic.h for why this
// lives beside the Highway kernel rather than inside it.
//
// Every routine here is a port of its CUDA counterpart in cuda/src/count2.cu
// and cuda/src/mesh.cu, kept line-for-line close so the two backends bucket
// particles, walk candidates and weight pairs identically. The per-pair math
// itself is not duplicated: it comes from the shared include/pair_math.h.

#include "cucount/cpu/generic.h"
// The scalar mesh, candidate walk and per-pair geometry, shared with the
// triplet counts.
#include "cucount/cpu/walk.h"
// The shared output layout: channel names and ordering come from the same
// place as the CUDA binding's, so the two cannot disagree.
#include "layout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cucount {
namespace cpu {
namespace {

using pairmath::add3;
using pairmath::compute_pair_mu;
using pairmath::compute_spin_projection_cartesian;
using pairmath::dot3;
using pairmath::get_bessel;
using pairmath::get_sep_bin_index;
using pairmath::lookup_angular_weight;
using pairmath::pair_bitwise_weight;
using pairmath::set_legendre;


// ---------------------------------------------------------------------------
// Per-pair accumulation
// ---------------------------------------------------------------------------

// The ell values the multipole axis asks for, and whether they are all even
// (which picks set_legendre's closed forms). Ported from fill_ells and
// make_device_count2_layout so the two backends agree on the ell ordering.
struct Layout {
    size_t nbins = 1;
    size_t split_size = 1;
    size_t nells = 0;
    size_t ells[MAX_POLE + 2] = {0};
    bool ells_even = false;
};

size_t fill_ells(const BinAttrs& battrs, int index, size_t* ells) {
    const size_t ellmin = static_cast<size_t>(battrs.min[index]);
    const size_t ellmax = static_cast<size_t>(battrs.max[index]);
    const size_t ellstep = (battrs.bin[index] == BIN_LIN)
        ? static_cast<size_t>(battrs.step[index]) : size_t{1};

    if (ellstep == 0) return 0;
    size_t nells = 0;
    for (size_t ell = ellmin; ell <= ellmax; ell += ellstep) ells[nells++] = ell;
    return nells;
}

Layout make_layout(const BinAttrs& battrs, const SplitAttrs& spattrs) {
    Layout layout;
    layout.nbins = battrs.size;
    layout.split_size = spattrs.size;

    if (battrs.ndim > 0 && battrs.var[battrs.ndim - 1] == VAR_POLE) {
        const int ipole = static_cast<int>(battrs.ndim) - 1;
        layout.nells = fill_ells(battrs, ipole, layout.ells);

        if (layout.nells > 0) {
            if (layout.nells == 1) {
                layout.ells_even = (layout.ells[0] % 2 == 0);
            }
            else {
                const size_t ellstep = (battrs.asize[ipole] == 0)
                    ? static_cast<size_t>(battrs.step[ipole]) : size_t{1};
                layout.ells_even = ((layout.ells[0] % 2 == 0) && (ellstep % 2 == 0));
            }
        }
    }
    return layout;
}

inline void accumulate_weight2(double* counts, const double* weight, size_t wsize,
                               size_t bin_loc, int nsplit_targets,
                               const size_t* split_targets, const Layout& layout,
                               double factor) {
    if (nsplit_targets == 0) {
        for (size_t iweight = 0; iweight < wsize; iweight++)
            counts[bin_loc + iweight * layout.nbins] += weight[iweight] * factor;
        return;
    }
    const size_t weight_stride = layout.split_size * layout.nbins;
    for (int itarget = 0; itarget < nsplit_targets; itarget++) {
        const size_t offset = split_targets[itarget] * layout.nbins + bin_loc;
        for (size_t iweight = 0; iweight < wsize; iweight++)
            counts[offset + iweight * weight_stride] += weight[iweight] * factor;
    }
}

// Port of add_weight2. Structure, ordering and every early return follow the
// CUDA original; only the atomic add becomes a plain add into the caller's
// thread-local buffer.
void add_weight2(double* counts,
                 const double* sposition1, const double* sposition2,
                 const double* position1, const double* position2,
                 const double* value1, const double* value2,
                 const IndexValue& index_value1, const IndexValue& index_value2,
                 const BinAttrs& battrs, const WeightAttrs& wattrs,
                 const MeshAttrs& mattrs, const SplitAttrs& spattrs,
                 const Layout& layout) {
    int nsplit_targets = 0;
    size_t split_targets[2] = {0, 0};

    if (index_value1.size_split && index_value2.size_split) {
        if (spattrs.mode == SPLIT_JACKKNIFE) {
            // The split label rides the double-valued wire format as a
            // reinterpreted 64-bit integer, like every other packed column.
            long split1, split2;
            std::memcpy(&split1, &value1[index_value1.start_split], sizeof(long));
            std::memcpy(&split2, &value2[index_value2.start_split], sizeof(long));

            if (split2 == split1) {
                nsplit_targets = 1;
                split_targets[0] = static_cast<size_t>(split1);
            }
            else {
                nsplit_targets = 2;
                split_targets[0] = spattrs.nsplits + static_cast<size_t>(split1);
                split_targets[1] = spattrs.nsplits * 2 + static_cast<size_t>(split2);
            }
        }
    }

    double diff[NDIM];
    difference(diff, position2, position1, mattrs);

    const double s2 = dot3(diff, diff);
    const double kDefault = -1000.;

    double s = kDefault, mu = kDefault, mu2 = kDefault;

    LOS_TYPE los = LOS_NONE;
    VAR_TYPE var = VAR_NONE;

    bool required_s = false, required_mu = false, required_mu2 = false;
    size_t i = 0;

    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        if (var == VAR_S || var == VAR_K) required_s = true;
        if (var == VAR_MU) { los = battrs.los[i]; required_mu = true; }
        if (var == VAR_RP) { los = battrs.los[i]; required_mu2 = true; }
        if (var == VAR_PI) { los = battrs.los[i]; required_mu = true; }
        if (var == VAR_POLE) {
            los = battrs.los[i];
            required_mu2 = true;
            if (!layout.ells_even) required_mu = true;
        }
    }

    required_s |= required_mu;
    if (required_s) s = std::sqrt(s2);

    if (required_mu2 || required_mu)
        compute_pair_mu(diff, sposition1, sposition2, position1, position2,
                        los, s, s2, required_mu, &mu, &mu2);

    size_t ibin = 0;

    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        double value = 0.;

        if (var == VAR_S) value = s;
        else if (var == VAR_MU) value = mu;
        else if (var == VAR_THETA) value = std::acos(dot3(sposition1, sposition2)) / DTORAD;
        else if (var == VAR_PI) value = mu * s;
        else if (var == VAR_RP) value = (s2 <= 0.) ? 0. : std::sqrt(s2 - s2 * mu2);

        if (var != VAR_POLE && var != VAR_K) {
            const int ibin_loc = get_sep_bin_index(
                value, battrs.array[i], static_cast<int>(battrs.shape[i]),
                battrs.bin[i], true);
            if (ibin_loc < 0) return;
            ibin = ibin * battrs.shape[i] + static_cast<size_t>(ibin_loc);
        }
        else break;
    }

    double pair_weight = 1.;

    if (index_value1.size_individual_weight)
        pair_weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight)
        pair_weight *= value2[index_value2.start_individual_weight];

    if (index_value1.size_bitwise_weight && index_value2.size_bitwise_weight)
        pair_weight *= pair_bitwise_weight(value1, value2, index_value1,
                                           index_value2, wattrs.bitwise);

    if (wattrs.angular.size) {
        const double ct[1] = {dot3(sposition1, sposition2)};
        pair_weight *= lookup_angular_weight<1>(ct, wattrs.angular);
    }

    if (index_value1.size_negative_weight && index_value2.size_negative_weight)
        pair_weight -= value1[index_value1.start_negative_weight] *
                       value2[index_value2.start_negative_weight];

    double weight[MAX_NWEIGHT];
    size_t wsize = 1;

    double splus1 = 0, scross1 = 0, splus2 = 0, scross2 = 0;

    if (index_value1.size_spin)
        compute_spin_projection_cartesian(sposition1, sposition2,
                                          &value1[index_value1.start_spin],
                                          static_cast<int>(wattrs.spin[0]),
                                          &splus1, &scross1);
    if (index_value2.size_spin)
        compute_spin_projection_cartesian(sposition1, sposition2,
                                          &value2[index_value2.start_spin],
                                          static_cast<int>(wattrs.spin[1]),
                                          &splus2, &scross2);

    if (index_value1.size_spin && index_value2.size_spin) {
        wsize = 3;
        weight[0] = pair_weight * splus1 * splus2;
        weight[1] = pair_weight * scross1 * splus2;
        weight[2] = pair_weight * scross1 * scross2;
    }
    else if (index_value1.size_spin) {
        wsize = 2;
        weight[0] = pair_weight * splus1;
        weight[1] = pair_weight * scross1;
    }
    else if (index_value2.size_spin) {
        wsize = 2;
        weight[0] = pair_weight * splus2;
        weight[1] = pair_weight * scross2;
    }
    else {
        wsize = 1;
        weight[0] = pair_weight;
    }

    const int ellstep_legendre = layout.ells_even ? 2 : 1;

    if (i == battrs.ndim) {
        accumulate_weight2(counts, weight, wsize, ibin, nsplit_targets,
                           split_targets, layout, 1.);
    }
    else if (i == battrs.ndim - 1 && var == VAR_POLE) {
        double legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, static_cast<int>(layout.ells[0]),
                     static_cast<int>(layout.ells[layout.nells - 1]),
                     ellstep_legendre, mu, mu2);

        for (size_t ill = 0; ill < layout.nells; ill++) {
            const size_t ell = layout.ells[ill];
            const size_t bin_loc = ibin * layout.nells + ill;
            const double leg = (2 * ell + 1) * legendre_cache[ell];
            accumulate_weight2(counts, weight, wsize, bin_loc, nsplit_targets,
                               split_targets, layout, leg);
        }
    }
    else if (i == battrs.ndim - 2 && battrs.var[i] == VAR_K &&
             battrs.var[i + 1] == VAR_POLE) {
        const size_t ik_dim = i;
        double legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, static_cast<int>(layout.ells[0]),
                     static_cast<int>(layout.ells[layout.nells - 1]),
                     ellstep_legendre, mu, mu2);

        const size_t nk = battrs.shape[ik_dim];
        const size_t npole = layout.nells;

        for (size_t ill = 0; ill < npole; ill++) {
            const int ell = static_cast<int>(layout.ells[ill]);
            const double leg = (((ell / 2) & 1) ? -1.0 : 1.0) * (2 * ell + 1) *
                               legendre_cache[ell];

            for (size_t ik = 0; ik < nk; ik++) {
                const double k = (battrs.asize[ik_dim] > 0)
                    ? battrs.array[ik_dim][ik]
                    : ik * battrs.step[ik_dim] + battrs.min[ik_dim];
                const size_t bin_loc = (ibin * nk + ik) * npole + ill;
                accumulate_weight2(counts, weight, wsize, bin_loc, nsplit_targets,
                                   split_targets, layout, leg * get_bessel(ell, k * s));
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void Count2Generic(const Count2GenericArgs& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const MeshAttrs& mattrs = args.mattrs;
    const ScalarMesh m1 = build_mesh(args.p1, mattrs);
    const ScalarMesh m2 = build_mesh(args.p2, mattrs);

    const auto t1 = Clock::now();

    const Layout layout = make_layout(args.battrs, args.spattrs);
    char raw_names[MAX_NWEIGHT][SIZE_NAME];
    const size_t nweights = get_count2_weight_names(m1.iv, m2.iv, raw_names);
    const size_t csize = nweights * layout.split_size * layout.nbins;

    const long total1 = static_cast<long>(m1.total);

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads)
#endif
    {
        std::vector<double> local(csize, 0.);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (long ii = 0; ii < total1; ii++) {
            const double* position1 = m1.position(ii);
            const double* sposition1 = m1.sposition(ii);
            const double* value1 = m1.value(ii);

            for_each_candidate(mattrs, m2, position1, sposition1, [&](size_t jj) {
                const double* position2 = m2.position(jj);
                const double* sposition2 = m2.sposition(jj);
                const double* value2 = m2.value(jj);

                if (!is_selected_pair(sposition1, sposition2, position1, position2,
                                      args.sattrs, mattrs))
                    return;

                add_weight2(local.data(), sposition1, sposition2, position1, position2,
                            value1, value2, m1.iv, m2.iv, args.battrs, args.wattrs,
                            mattrs, args.spattrs, layout);
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
