// Generic scalar count2 for the CPU backend.
//
// The Highway kernel in kernel-inl.h serves the shapes worth vectorising --
// s, (s, mu) and (s, pole) binning on a cartesian mesh. Everything else the
// CUDA backend accepts (theta / rp / pi / k axes, any combination and order of
// them, the angular mesh, jackknife splits) comes through here instead: one
// pair per iteration, over the same attrs structs the CUDA backend consumes,
// with the per-pair math taken from the shared pair_math.h.
//
// The two paths are deliberately separate translation units. Folding the
// generic cases into the kernel would put runtime branches in its inner loop,
// which the A/B benchmark has twice shown costs the plain path 2-8% even when
// the branch is never taken.
#pragma once

// Brings the shared descriptors (Particles, MeshAttrs, BinAttrs, WeightAttrs,
// SelectionAttrs, SplitAttrs) via common.h, and defines CUCOUNT_NO_CUDA on
// the way in so the CUDA include is skipped. Include this header before
// attrs.h in a translation unit that needs both: pair_math.h undefines
// common.h's convenience macros on its way out, and attrs.h's own include of
// common.h then restores them.
#include "pair_math.h"
// The shared request bundle: what to count, identical on both backends.
#include "args.h"

namespace cucount {
namespace cpu {

// What the scalar entry point needs: the shared request bundle, plus what
// this backend runs it with. Plain descriptors holding borrowed pointers into
// buffers the caller keeps alive for the duration of the call.
struct Count2Args {
    Particles p1;
    Particles p2;
    Count2Attrs attrs;

    int nthreads = 1;

    // nweights * spattrs.size * battrs.size accumulators, channel-major, the
    // layout get_count2_layout describes. Zeroed by the caller.
    double* out = nullptr;

    // Optional [mesh_seconds, count_seconds], to separate setup from the counting.
    double* timings = nullptr;
};

void Count2(const Count2Args& args);

}  // namespace cpu
}  // namespace cucount
