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

// Full parity with the CUDA LOS_TYPE choices (minus LOS_NONE).
// FirstPoint/EndPoint project on the unit-sphere position of particle 1 / 2,
// so those two need spositions carried through the mesh.
enum class LosKind { AxisZ, AxisX, AxisY, Midpoint, FirstPoint, EndPoint };

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
    // Filled only when the pair count involves spin: unit-sphere positions
    // (both meshes; the projection frame needs both endpoints) and this
    // catalogue's two spin components (only where they exist).
    std::vector<Float> sx, sy, sz;
    std::vector<Float> e1, e2;
    // Filled only for PIP weighting: nbitwise 64-bit realization masks per
    // particle, interleaved, kept as doubles (the wire format's bit patterns)
    // whatever the working precision; and the optional negative-weight column.
    std::vector<double> bw;
    std::vector<Float> nw;
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

    // Optional spin-2 components, interleaved (e1, e2) per particle, nN * 2;
    // null when that catalogue carries none. The order is the spin the
    // components transform with (2 for shear); meaningful only with a non-null
    // pointer on the same side.
    const double* spin1 = nullptr;
    const double* spin2 = nullptr;
    int spin_order1 = 0;
    int spin_order2 = 0;

    // Optional PIP (bitwise) weighting: nbitwise 64-bit realization masks per
    // particle stored as doubles, interleaved; applied only when both sides
    // carry them, like the CUDA kernel. The correction table, when present,
    // is (p_nbits x p_nbits) doubles indexed by the per-particle popcounts.
    const double* bw1 = nullptr;
    const double* bw2 = nullptr;
    size_t nbitwise = 0;
    double bitwise_default = 0.;
    double bitwise_nrealizations = 0.;
    int bitwise_noffset = 0;
    size_t bitwise_p_nbits = 0;
    const double* bitwise_p_correction = nullptr;

    // Optional negative weights (one column per side; subtracted as
    // nw1 * nw2 when both sides carry one, after the bitwise factor).
    const double* nw1 = nullptr;
    const double* nw2 = nullptr;

    // Optional pair selections (VAR_S / VAR_THETA), INCLUSIVE on both ends
    // like is_selected_pair. A selection routes the request through the
    // scalar tail, so the plain vector path stays untouched.
    bool sel_s = false;
    double sel_s_min = 0., sel_s_max = 0.;
    bool sel_theta = false;
    double sel_ct_min = 0., sel_ct_max = 0.;

    // Optional multipole axis (VAR_POLE). It is the FASTEST bin axis: for
    // each pair the kernel adds (2 ell + 1) P_ell(mu) into nells consecutive
    // bins, so mu is computed but never binned. ells_even picks the
    // even-only closed forms in set_legendre (else the full recursion).
    const int* ells = nullptr;
    size_t nells = 0;
    bool ells_even = false;

    // Optional 1D angular (PIP) upweight, tabulated against ascending
    // cos(theta) (the Python layer converts from degrees): interpolation
    // points or bin edges per angular_sep_is_edges, policy angular_bin
    // (a BIN_TYPE value), applied as a factor between bitwise and negative.
    const double* angular_sep = nullptr;
    const double* angular_weight = nullptr;
    size_t angular_shape = 0;
    int angular_bin = 0;
    bool angular_sep_is_edges = false;

    double boxsize[3] = {0, 0, 0};
    double origin[3] = {0, 0, 0};

    Config cfg;
    int nthreads = 1;

    // nweights * nsbins * nmubins accumulators (channel-major, matching the
    // CUDA layout; nweights = 1 + one per side with spin), always double: the
    // point of float32 is twice the lanes through the geometry, not a narrower
    // accumulator.
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
