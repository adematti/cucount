// Close triplet counts on the CPU: see count3.h for the shared request
// struct, which both triplet entry points take.
#pragma once

// Count3Args, and through it cmath.h and the shared args.h.
#include "count3.h"

namespace cucount {
namespace cpu {

// Close triplet counts: every (1, 2, 3) triplet is formed and binned, with an
// optional (2, 3) axis, so unlike Count3 there is no factorization. This is
// where the 3-dimensional angular upweight applies, indexed by the three
// cos(theta) of the triangle.
//
// `close_pair` (Count3Args::close_pair: 12, 13 or 23) picks the search
// strategy, as on CUDA; all of them enumerate the same triplets. (1, 2) and
// (1, 3) walk from each particle 1: the close leg first, then the other one,
// so mattrs2 has to bound the (1, 2) separation and mattrs3 the (1, 3) one.
// (2, 3) walks from each particle 2: its close particles 3 in mattrs3's
// window, which bounds the (2, 3) separation, then particle 1 in mattrs1's,
// which bounds the (1, 2) one.
//
// That makes the meshes part of the contract: one too narrow for the leg it
// is walked over drops triplets rather than rejecting them, which is quietly
// wrong rather than merely slow. The frontends size the default meshes for
// close_pair, as for CUDA, when the module has `count3close_strategies` (an
// older build walked every close pair from particle 1, and needed the (1, 2)
// sizing whatever close_pair said), and check any mesh passed in by hand.
void Count3Close(const Count3Args& args);

}  // namespace cpu
}  // namespace cucount
