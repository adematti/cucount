#ifndef _CUCOUNT_COUNT2_SOFT_
#define _CUCOUNT_COUNT2_SOFT_

// Soft (Gaussian-softmax) pair counts on the separation axis.
//
// This is an extension, not a mode of count2: its own header, its own
// translation unit, its own binding. The hard kernel's inner loop is the
// reference for both correctness and speed, and it must not grow a branch or a
// register for soft binning, so nothing here is included by it.
//
// v1 serves one ``s`` axis -- isotropic clustering. A request for anything else
// (a second axis, multipoles, spin, angular upweights, splits) is refused by the
// binding with a Python exception; these kernels assume the request has already
// been narrowed.
//
// Semantics (the same as pyfcfc's soft_hist.c, which is the oracle):
//
//   c_i = (edges[i] + edges[i+1]) / 2      bin centres
//   x_i = (s - c_i)^2 / (2 sigma_i^2)      Gaussian logits
//   W_i = exp(-x_i) / sum_j exp(-x_j)      softmax over the bins
//   counts[i] += w_pair * W_i              per pair, w_pair = product of weights
//
// The sum runs over the taps within ``cutoff`` of the peak, then everything is
// renormalized, so a pair's weight is exactly conserved and sigma -> 0
// degenerates to hard binning (nearest centre). Pairs outside [smin, smax) are
// dropped whole, exactly as the hard kernel drops them, which is what makes the
// soft and hard totals agree at any sigma.
//
// Two grids, two code paths:
//
//  * uniform edges keep the multiplicative-recurrence kernel: two exp() calls
//    per pair and one multiply per tap;
//  * arbitrary monotone edges use the general kernel: a binary search for the
//    peak, a window expanded until the logit distance passes ln(1/cutoff), and
//    one exp() per tap. Because the general path has no recurrence, the
//    sigma_i^2 and centre arrays are uploaded once per call (a few kB).
//
// sigma follows the width of the bin unless asked otherwise:
//
//   sigma <= 0 (None) + non-uniform edges -> sigma_i = 0.5 * (edges[i+1] - edges[i])
//   sigma <= 0 (None) + uniform edges     -> sigma = 0.5 * ds   (the same thing)
//   sigma > 0                             -> one absolute sigma for every bin
//
// The width-scaled default is what makes a log-spaced grid mean "constant
// relative resolution" instead of "wide bins are under-smoothed and narrow ones
// are over-smoothed", and because sigma_i is then a function of the edges it is
// a constant of the binning: there is no dL/dsigma to compute for it.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common.h"
#include "args.h"


// Everything the kernel needs per pair, resolved on the host. Small enough to
// ride a kernel argument in the constant bank.
struct SoftBinSpec {
    int nbins;        // number of s bins
    FLOAT c0;         // centre of bin 0, (edges[0] + edges[1]) / 2
    FLOAT ds;         // bin width
    FLOAT inv_ds;     // 1 / ds, for the nearest-centre search
    FLOAT lo, hi;     // [lo, hi) of s, as hard binning has it
    FLOAT inv2sig2;   // 1 / (2 sigma^2)
    FLOAT inv_sigma;  // 1 / sigma, for the sigma cotangent
    FLOAT gg;         // ds^2 / sigma^2: how much the log-ratio steps per tap
    FLOAT iq;         // exp(-gg): the multiplicative step of the tap ratios
    FLOAT cutoff;     // tap truncation threshold, e.g. 1e-13
    int kmax;         // taps tried on each side (window half-width)
    // Scratch paths are a performance detail, decided inside count2_soft from
    // the shared-memory budget: how many private copies of the histogram the
    // block keeps. Written back so the caller can report it.
    int path;         // 0 = global atomics, else copies per warp (32, 8 or 1)

    // --- non-uniform extension -------------------------------------------
    int uniform;         // 1 when edges[i] == edges[0] + i * ds: recurrence path
    int per_bin_sigma;   // 1 when sigma_i = alpha * (edges[i+1] - edges[i])
    int fast_taps;       // 1: the general path may trade accuracy for __expf
    FLOAT sigma;         // the absolute sigma when per_bin_sigma == 0
    FLOAT alpha;         // sigma_i / local width when per_bin_sigma == 1
    FLOAT thresh;        // ln(1 / cutoff): the logit distance that truncates
    const FLOAT *host_edges;  // host copy of the edges, nbins + 1 (general path)
    FLOAT *centres;           // device, nbins (general path only)
    FLOAT *inv2s2;            // device, nbins (general path only)
    // The window search needs the *edges*, not the centres: "the last centre at
    // or below s" is the previous bin whenever s sits in the lower half of its
    // bin, which only shows up when the kernel is narrow enough for the error
    // not to be covered by the window.
    FLOAT *edges;             // device, nbins + 1 (general path only)
};


#ifdef __CUDACC__

// Per-tap weight of the general path. ``fast`` compiles to a single SFU
// instruction (~1e-6 relative), the default is the exact double exp.
__device__ __forceinline__ FLOAT soft_tap_weight(FLOAT delta, bool fast)
{
    return fast ? (FLOAT) __expf(-(float) delta) : exp(-delta);
}

#endif  // __CUDACC__


// Host-side resolution of a soft request into the kernel argument. Lives here,
// not in a binding, because two bindings need it: the python extension (which
// raises to the caller) and the jax FFI (which turns the same error into a
// failed computation). Throws std::invalid_argument for anything v1 does not
// serve, so a caller can never silently get hard binning instead.
inline SoftBinSpec make_soft_bin_spec(const BinAttrs &battrs, const WeightAttrs &wattrs,
                                      const SplitAttrs &spattrs, double sigma,
                                      double cutoff, size_t *csize)
{
    const char *prefix = "cuda backend, soft binning: ";

    if (battrs.ndim != 1 || battrs.var[0] != VAR_S) {
        throw std::invalid_argument(
            std::string(prefix) + "v1 softens a single 's' axis (isotropic clustering); "
            "a request with " + std::to_string(battrs.ndim) + " axis/axes is not served yet");
    }

    if (spattrs.mode != SPLIT_NONE) {
        throw std::invalid_argument(
            std::string(prefix) + "splits (jackknife) are not served by the soft kernel yet");
    }

    if (wattrs.angular.size) {
        throw std::invalid_argument(
            std::string(prefix) + "angular (PIP) upweights are not served by the soft kernel yet");
    }

    if (battrs.shape[0] < 2 || battrs.asize[0] < battrs.shape[0] + 1) {
        throw std::invalid_argument(
            std::string(prefix) + "at least 2 s bins with edges are needed");
    }

    const size_t nbins = battrs.shape[0];
    const FLOAT *edges = battrs.array[0];
    const double ds = (double) edges[1] - (double) edges[0];

    if (!(ds > 0)) {
        throw std::invalid_argument(
            std::string(prefix) + "s bin edges must be strictly increasing");
    }

    if (!std::isfinite(cutoff) || !(cutoff > 0) || !(cutoff < 1)) {
        throw std::invalid_argument(
            std::string(prefix) + "soft_cutoff must lie in (0, 1)");
    }

    // Are the edges an arithmetic progression? The check is on the edges, not on
    // battrs.bin, so a BIN_CUSTOM call that happens to be uniform (an arange)
    // gets the fast path too. Non-uniform grids are served by the general path,
    // so this is now only a path selector, not a rejection.
    bool uniform = true;
    const double tol = 1e-9 * std::fabs(ds);

    for (size_t i = 2; i <= nbins; i++) {
        const double want = (double) edges[0] + (double) i * ds;

        if (std::fabs((double) edges[i] - want) > tol) {
            uniform = false;
            break;
        }
    }

    if (!uniform) {
        // Strictly increasing, which the recurrence path gets for free from the
        // uniform check above but the binary search genuinely needs.
        for (size_t i = 0; i < nbins; i++) {
            if (!((double) edges[i + 1] > (double) edges[i])) {
                throw std::invalid_argument(
                    std::string(prefix) + "s bin edges must be strictly increasing");
            }
        }
    }

    if (!std::isfinite(sigma)) {
        throw std::invalid_argument(std::string(prefix) + "soft_sigma must be finite");
    }

    SoftBinSpec spec{};

    spec.nbins = (int) nbins;
    spec.ds = (FLOAT) ds;
    spec.inv_ds = (FLOAT) (1. / ds);
    spec.c0 = (FLOAT) (edges[0] + 0.5 * ds);
    spec.lo = edges[0];
    spec.hi = edges[nbins];
    spec.cutoff = cutoff;
    spec.thresh = (FLOAT) std::log(1. / cutoff);
    spec.uniform = uniform ? 1 : 0;
    spec.host_edges = edges;
    spec.centres = nullptr;
    spec.inv2s2 = nullptr;
    spec.edges = nullptr;
    spec.fast_taps = 0;

    // sigma <= 0 or None means "half a bin", the pyfcfc default: at that width a
    // pair keeps ~68% of its weight on its own bin and the rest on the two
    // neighbours. On a non-uniform grid "half a bin" is per bin, and sigma_i
    // then depends only on the edges, i.e. it is not a differentiable knob.
    const bool width_scaled = !(sigma > 0) && !uniform;

    spec.per_bin_sigma = width_scaled ? 1 : 0;
    spec.alpha = 0.5;
    spec.sigma = (FLOAT) ((sigma > 0) ? sigma : 0.5 * ds);

    if (width_scaled) {
        // One sigma per bin, so no single 1/sigma^2 exists; the kernel reads the
        // per-bin array and the sigma cotangent is identically zero.
        spec.inv2sig2 = 0;
        spec.inv_sigma = 0;
        spec.gg = 0;
        spec.iq = 1;
        spec.kmax = (int) (nbins - 1);
    }
    else {
        const double sg = (double) spec.sigma;

        spec.inv2sig2 = (FLOAT) (1. / (2. * sg * sg));
        spec.inv_sigma = (FLOAT) (1. / sg);
        spec.gg = (FLOAT) (ds * ds / (sg * sg));
        spec.iq = (FLOAT) std::exp(-(double) spec.gg);

        if (uniform) {
            // Taps whose weight is still above `cutoff` at t bins away: the
            // Gaussian logit is (t*ds)^2 / (2 sigma^2), so
            // t_max = sqrt(2 ln(1/cutoff)) sigma/ds. Floor at 1 so a sigma well
            // below the bin width still looks at its neighbours (where the
            // weight is zero, but cheaply), and cap at nbins - 1 because the
            // window cannot be longer than the axis.
            const double kmax =
                std::ceil(std::sqrt(2. * std::log(1. / cutoff)) * sg / ds);

            spec.kmax = (int) std::min(std::max(kmax, 1.), (double) (nbins - 1));
        }
        else {
            // With a single absolute sigma on a non-uniform grid the window's
            // length in bins depends on the local width, which the host does not
            // know per bin: let the kernel expand until the cutoff stops it.
            spec.kmax = (int) (nbins - 1);
        }
    }

    // How the scatter is staged is a kernel decision, not a host one: the kernel
    // picks the number of private histogram copies from the shared-memory budget
    // and writes the choice back into spec.path (0 = direct global atomics).
    spec.path = 0;

    *csize = nbins;

    return spec;
}


void count2_soft(FLOAT *counts, const Mesh *list_mesh, const Count2Attrs &attrs,
                 SoftBinSpec &spec, DeviceMemoryBuffer *buffer,
                 cudaStream_t stream);


// Vector-Jacobian product of the forward above: given the cotangent of the
// histogram (dL/dcounts), accumulate dL/dpositions, dL/dindividual weights and
// dL/dsigma. This is what lets the jax layer declare a custom_vjp and get
// differentiable pair counts without differentiating a scatter by hand.
//
// The gradients are *zeroed* by this call and then accumulated with atomics, so
// grad_position1 holds NDIM * mesh[0].size entries (catalog 1), grad_position2
// NDIM * mesh[1].size (catalog 2), grad_weight* one per particle, and
// grad_sigma a single entry. For an autocorrelation pass the same arrays as
// catalog 1 and 2: the ordered pairs the kernel walks then contribute to both
// endpoints, which is exactly dL/dr_i for the symmetric sum.
//
// Why this is a second kernel rather than reverse-mode AD: the forward is a
// scatter into a histogram, so its input cotangent is a gather from the same
// window, and the per-pair reduction collapses to a handful of sums. Written
// out (c_b = centres, x_b = logits, W_b = softmax, cbar/xbar their W-means):
//
//   dL/dw_pair = sum_b g_b W_b                          (a scalar per pair)
//   dL/ds      = (w_pair / sigma^2) sum_b g_b W_b (c_b - cbar)
//   dL/dsigma  = (2 w_pair / sigma) sum_b g_b W_b (x_b - xbar)
//   dL/dr_i    = dL/ds * (r_i - r_j) / s ,  dL/dr_j = -dL/dr_i
//
// which is pyfcfc's soft_hist_vjp. Window membership and the range check are
// piecewise constant in s, so they are treated as constants (the standard
// convention) and the window's own renormalisation is differentiated exactly.
//
// With a per-bin sigma the same reduction holds tap by tap,
//
//   dL/ds = 2 w_pair [ sum_b g_b W_b (c_b - s) / sigma_b^2
//                      + (sum_b g_b W_b) * sum_j W_j (s - c_j) / sigma_j^2 ]
//
// which reduces to the constant-sigma line above when every sigma_b is equal.
// In that case there is no sigma cotangent (sigma_i comes from the edges, so it
// is a constant of the binning) and grad_sigma is left alone.
void count2_soft_vjp(const FLOAT *cotangent, FLOAT *grad_position1,
                     FLOAT *grad_position2, FLOAT *grad_weight1,
                     FLOAT *grad_weight2, FLOAT *grad_sigma,
                     const Mesh *list_mesh, const Count2Attrs &attrs,
                     const SoftBinSpec &spec, DeviceMemoryBuffer *buffer,
                     cudaStream_t stream);

#endif  // _CUCOUNT_COUNT2_SOFT_
