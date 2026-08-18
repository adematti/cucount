// Runtime configuration and data layout for the CPU count2.
// Nothing here depends on Highway, so this header is safe to include once
// per program rather than once per SIMD target.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cucount {
namespace cpu {

// Mirrors BIN_LIN / BIN_LOG / BIN_CUSTOM in include/common.h.
enum class BinKind { Linear, Log, Edges };

enum class LosKind { AxisZ, Midpoint };

enum class ScatterKind { Scalar, BinMajor };

struct Config {
    int ndim = 1;  // 1 -> bin in s; 2 -> bin in (s, mu)
    BinKind sbin = BinKind::Linear;
    LosKind los = LosKind::AxisZ;  // ignored when ndim == 1
    bool periodic = false;
    bool float32 = false;
    ScatterKind scatter = ScatterKind::Scalar;
};

// Bin edges held in the working precision.
//
// Only the Linear policy needs s itself; Log and Edges are evaluated against
// s^2 so the kernel can skip the square root entirely when ndim == 1. sq holds
// the squared edges for those two, and is left empty for the mu axis.
template <class Float>
struct BinSpec {
    std::vector<Float> edges;
    std::vector<Float> sq;
    Float lo = 0, hi = 0;
    Float sq_lo = 0, sq_hi = 0;
    Float inv_step = 0;  // Linear: 1/step. Log: 1/log(step), halved so it applies to s^2.
    Float log_sq_lo = 0;  // Log only: log of the first squared edge.
    size_t nbins = 0;

    void set(const double* e, size_t n, bool squared) {
        nbins = n;
        edges.assign(e, e + n + 1);
        lo = edges.front();
        hi = edges.back();
        if (squared) {
            sq.resize(n + 1);
            for (size_t i = 0; i <= n; ++i) sq[i] = edges[i] * edges[i];
            sq_lo = sq.front();
            sq_hi = sq.back();
        }
    }
};

// Particles bucketed into cells and stored SoA, so the candidate loop can
// load whole vectors of x (then y, then z) without a gather. This is the one
// deliberate layout divergence from the CUDA mesh, which interleaves xyz.
template <class Float>
struct Mesh {
    std::vector<Float> x, y, z, w;
    std::vector<size_t> start;  // ncells + 1 offsets into x/y/z/w
    int dims[3] = {1, 1, 1};
    Float cell[3] = {0, 0, 0};
    Float origin[3] = {0, 0, 0};  // low corner of the box

    size_t ncells() const {
        return static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    }
};

// Everything the entry point needs, in double regardless of the working
// precision; conversion happens once the Float type has been chosen.
struct Count2Args {
    const double* pos1 = nullptr;  // interleaved xyz, n1 * 3
    const double* w1 = nullptr;
    size_t n1 = 0;
    const double* pos2 = nullptr;
    const double* w2 = nullptr;
    size_t n2 = 0;

    const double* sedges = nullptr;
    size_t nsbins = 0;
    const double* muedges = nullptr;
    size_t nmubins = 0;

    double boxsize[3] = {0, 0, 0};
    double origin[3] = {0, 0, 0};

    Config cfg;
    int nthreads = 1;

    // nsbins * nmubins accumulators, always double: the point of float32 is
    // twice the lanes through the geometry, not a narrower accumulator.
    double* out = nullptr;

    // Optional [mesh_seconds, pair_seconds], to separate setup from pair work.
    double* timings = nullptr;
};

void Count2(const Count2Args& args);

// Force Highway to a specific ISA, for the cross-target determinism check and
// the SIMD-width scaling measurement. Empty string restores the default.
// Returns the name of the target actually selected afterwards.
const char* SetTarget(const char* name);

const char* CurrentTarget();

std::vector<const char*> AvailableTargets();

}  // namespace cpu
}  // namespace cucount
