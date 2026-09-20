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
// The CUDA backend picks among several search strategies with `close_pair`;
// all of them enumerate the same triplets, and the choice is a performance
// hint. This backend has one strategy -- walk catalogue 2 and then catalogue 3
// from each primary -- so `close_pair` is accepted and ignored.
//
// That makes the meshes it is handed part of the contract. Centring on
// particle 1 means mattrs2 has to bound the (1, 2) separation and mattrs3 the
// (1, 3) one, whatever close_pair says; a mesh sized for the (2, 3) leg
// sweeps too narrow a window and drops triplets rather than rejecting them,
// which is quietly wrong rather than merely slow. The frontend sizes the
// default meshes for this traversal and checks any mesh passed in by hand.
void Count3Close(const Count3Args& args);

}  // namespace cpu
}  // namespace cucount
