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
// The shared request bundle: what to count, identical on both backends.
#include "args.h"

namespace cucount {
namespace cpu {

// What both triplet entry points need. One struct for the two, because
// count3 is count3close without a (2, 3) axis: it simply leaves the (2, 3)
// members of Count3Attrs default.
struct Count3Args {
    Particles p1;
    Particles p2;
    Particles p3;

    Count3Attrs attrs;

    int nthreads = 1;

    // The accumulators get_count3_layout describes. Zeroed by the caller.
    double* out = nullptr;

    // Optional [mesh_seconds, count_seconds], to separate setup from the counting.
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
void Count3Close(const Count3Args& args);

}  // namespace cpu
}  // namespace cucount
