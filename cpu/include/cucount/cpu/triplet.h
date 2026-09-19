// Factorized triplet counts on the CPU.
//
// For each primary in catalogue 1, catalogue 2 is histogrammed against the
// (1, 2) separation and catalogue 3 against the (1, 3) separation, and the
// outer product of the two histograms is accumulated. There is no binning or
// selection on the (2, 3) separation, which is what makes the cost O(n * k)
// rather than O(n * k^2). When both legs carry a multipole axis the
// histograms are projections onto the real spherical harmonics of the local
// line-of-sight frame, and the outer product contracts over m.
//
// A port of cuda/src/count3.cu, over the same attrs structs and the same
// mesh; the per-pair math comes from the shared include/pair_math.h.
#pragma once

// Before attrs.h in any translation unit that needs both: pair_math.h
// undefines common.h's convenience macros on its way out.
#include "pair_math.h"

namespace cucount {
namespace cpu {

struct Count3Args {
    Particles p1;
    Particles p2;
    Particles p3;

    MeshAttrs mattrs1;
    MeshAttrs mattrs2;
    MeshAttrs mattrs3;

    BinAttrs battrs12;
    BinAttrs battrs13;

    WeightAttrs wattrs;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs veto12;
    SelectionAttrs veto13;

    int nthreads = 1;

    // nbins * nprojs accumulators, the layout get_count3_out_layout
    // describes. Zeroed by the caller.
    double* out = nullptr;

    // Optional [mesh_seconds, triplet_seconds].
    double* timings = nullptr;
};

void Count3(const Count3Args& args);


// Close triplet counts: every (1, 2, 3) triplet is formed and binned, with an
// optional (2, 3) axis, so unlike Count3 there is no factorization. This is
// where the 3-dimensional angular upweight applies, indexed by the three
// cos(theta) of the triangle.
//
// The CUDA backend picks among several search strategies with `close_pair`;
// all of them enumerate the same triplets, and the choice is a performance
// hint. This backend has one strategy -- walk catalogue 2 and then catalogue 3
// from each primary -- so `close_pair` is accepted and ignored. A request that
// only bounds the (2, 3) separation will therefore be slow here, not wrong.
struct Count3CloseArgs {
    Particles p1;
    Particles p2;
    Particles p3;

    MeshAttrs mattrs1;
    MeshAttrs mattrs2;
    MeshAttrs mattrs3;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;  // ndim == 0 when there is no (2, 3) axis

    WeightAttrs wattrs;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;
    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    int nthreads = 1;

    double* out = nullptr;
    double* timings = nullptr;
};

void Count3Close(const Count3CloseArgs& args);

}  // namespace cpu
}  // namespace cucount
