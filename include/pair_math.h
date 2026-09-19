// Scalar per-pair math shared by the CUDA and CPU backends.
//
// Header-only and templated over the float type: under nvcc every function is
// __host__ __device__; in a plain C++ translation unit (the Highway CPU
// backend) the same source compiles as ordinary inline functions. This is the
// one-source home for the math where correctness bugs hide; each backend keeps
// its own parallelization idiom around it.

#ifndef _CUCOUNT_PAIR_MATH_
#define _CUCOUNT_PAIR_MATH_

#ifdef __CUDACC__
#define CUCOUNT_HOST_DEVICE __host__ __device__
#include <math.h>
#else
#define CUCOUNT_HOST_DEVICE
#include <cmath>
// The shared descriptors (LOS_TYPE, IndexValue, BitwiseWeight) come from
// common.h, whose CUDA include must be skipped in a CUDA-free consumer.
#ifndef CUCOUNT_NO_CUDA
#define CUCOUNT_NO_CUDA
#endif
#endif
#include "common.h"

// 64-bit popcount on either side of the fence.
#if defined(__CUDA_ARCH__)
#define CUCOUNT_POPCOUNT(x) __popcll(x)
#else
#define CUCOUNT_POPCOUNT(x) __builtin_popcountll((unsigned long long)(x))
#endif

// Loop unrolling hint: meaningful under nvcc, silently absent on the host.
#if defined(__CUDACC__)
#define CUCOUNT_UNROLL _Pragma("unroll")
#else
#define CUCOUNT_UNROLL
#endif

namespace cucount {
namespace pairmath {

#ifndef __CUDACC__
using std::sqrt;
using std::sin;
using std::cos;
using std::atan2;
using std::log;
using std::floor;
using std::fabs;
#endif

// Legendre polynomials P_ell(mu) for ellmin <= ell <= ellmax. The even-only
// fast path writes closed forms up to ell = 8; anything else falls back to the
// full Bonnet recursion, which needs every order up to ellmax in the cache.
template <typename Float>
CUCOUNT_HOST_DEVICE inline void set_legendre(Float *legendre_cache, int ellmin, int ellmax, int ellstep, Float mu, Float mu2) {
    if ((ellmin % 2 == 0) && (ellstep % 2 == 0)) {
        for (int ell = ellmin; ell <= ellmax; ell += ellstep) {
            if (ell == 0) {
                legendre_cache[ell] = 1.;
            }
            else if (ell == 2) {
                legendre_cache[ell] = (3.0 * mu2 - 1.0) / 2.0;
            }
            else if (ell == 4) {
                Float mu4 = mu2 * mu2;
                legendre_cache[ell] = (35.0 * mu4 - 30.0 * mu2 + 3.0) / 8.0;
            }
            else if (ell == 6) {
                Float mu4 = mu2 * mu2;
                Float mu6 = mu4 * mu2;
                legendre_cache[ell] = (231.0 * mu6 - 315.0 * mu4 + 105.0 * mu2 - 5.0) / 16.0;
            }
            else if (ell == 8) {
                Float mu4 = mu2 * mu2;
                Float mu6 = mu4 * mu2;
                Float mu8 = mu4 * mu4;
                legendre_cache[ell] = (6435.0 * mu8 - 12012.0 * mu6 + 6930.0 * mu4 - 1260.0 * mu2 + 35.0) / 128.0;
            }
            else {
                legendre_cache[ell] = 0.;
            }
        }
    }
    else {
        legendre_cache[0] = 1.0;
        legendre_cache[1] = mu;

        for (int ell = 2; ell <= ellmax; ell++) {
            legendre_cache[ell] =
                ((2.0 * ell - 1.0) * mu * legendre_cache[ell - 1] -
                 (ell - 1.0) * legendre_cache[ell - 2]) / ell;
        }
    }
}


#ifndef BESSEL_XMIN
#define BESSEL_XMIN 0.1
#endif

// Spherical Bessel j_ell(x), even ell up to 4; series expansion below
// BESSEL_XMIN where the closed forms lose precision to cancellation.
template <typename Float>
CUCOUNT_HOST_DEVICE inline Float get_bessel(int ell, Float x) {
    if (x < BESSEL_XMIN) {
        Float x2 = x * x;

        switch (ell) {
            case 0:
                return 1. - x2 / 6. + x2 * x2 / 120. - x2 * x2 * x2 / 5040.;
            case 2:
                return x2 / 15. - x2 * x2 / 210. + x2 * x2 * x2 / 11340.;
            case 4:
                return x2 * x2 / 945. - x2 * x2 * x2 / 10395.;
            default:
                return 0.0;
        }
    }
    else {
        Float invx  = 1.0 / x;
        Float invx2 = invx * invx;
        Float invx3, invx4;

        switch (ell) {
            case 0:
                return sin(x) * invx;
            case 2:
                return (3.0 * invx2 - 1.0) * sin(x) * invx - 3.0 * cos(x) * invx2;
            case 4:
                invx3 = invx2 * invx;
                invx4 = invx2 * invx2;
                return 5 * (2 * invx2 - 21 * invx4) * cos(x) +
                       (invx - 45 * invx3 + 105 * invx2 * invx3) * sin(x);
            default:
                return 0.0;
        }
    }
}


// Project spin-`spin` components s = (s1, s2) onto the (+, x) frame the pair
// defines: east/north at unit vector r1 (basis fixed by the z pole), rotated by
// `spin` times the position angle of r2 seen from r1. Both backends call this
// with (r1, r2) in the same order for either particle's components.
//
// A particle exactly at a pole has no defined east; like the historical CUDA
// code, the projection then returns non-finite values rather than guessing.
template <typename Float>
CUCOUNT_HOST_DEVICE inline void compute_spin_projection_cartesian(
    const Float *r1,
    const Float *r2,
    const Float *s,
    int spin,
    Float *splus_out,
    Float *scross_out)
{
    if (spin != 0) {
        const Float zhat[3] = {0.0, 0.0, 1.0};

        Float east[3] = {
            zhat[1] * r1[2] - zhat[2] * r1[1],
            zhat[2] * r1[0] - zhat[0] * r1[2],
            zhat[0] * r1[1] - zhat[1] * r1[0]
        };

        // Full-precision normalization (the pre-extraction CUDA code used
        // rsqrtf here, truncating the double path to float precision).
        Float east_norm = Float(1) / sqrt(east[0] * east[0] + east[1] * east[1] + east[2] * east[2]);
        east[0] *= east_norm;
        east[1] *= east_norm;
        east[2] *= east_norm;

        Float north[3] = {
            r1[1] * east[2] - r1[2] * east[1],
            r1[2] * east[0] - r1[0] * east[2],
            r1[0] * east[1] - r1[1] * east[0]
        };

        Float dot12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];

        Float p[3] = {
            r2[0] - dot12 * r1[0],
            r2[1] - dot12 * r1[1],
            r2[2] - dot12 * r1[2]
        };

        Float pe = p[0] * east[0] + p[1] * east[1] + p[2] * east[2];
        Float pn = p[0] * north[0] + p[1] * north[1] + p[2] * north[2];
        Float phi = atan2(pe, pn);

        Float sphi = sin(spin * phi);
        Float cphi = cos(spin * phi);

        *splus_out  = -(s[0] * cphi + s[1] * sphi);
        *scross_out =  (s[0] * sphi - s[1] * cphi);
    }
    else {
        *splus_out = -s[0];
        *scross_out = -s[1];
    }
}

// 3-vector helpers with the same accumulation order as the historical
// macro-generated dot()/addition(), so extractions built on them are
// bit-for-bit refactors on the CUDA side.
template <typename Float>
CUCOUNT_HOST_DEVICE inline Float dot3(const Float *position1, const Float *position2) {
    Float d = (Float)0.;
    for (size_t axis = 0; axis < 3; axis++) {
        d += position1[axis] * position2[axis];
    }
    return d;
}


template <typename Float>
CUCOUNT_HOST_DEVICE inline void add3(Float *add, const Float *position1, const Float *position2) {
    for (size_t axis = 0; axis < 3; axis++) {
        add[axis] = position1[axis] + position2[axis];
    }
}


// The per-pair line-of-sight geometry, lifted verbatim from the CUDA
// add_weight2: given the (already periodic-wrapped) separation diff and the
// unit-sphere / cartesian positions, fill mu (when required_mu) and mu2.
// Coincident points (s2 == 0) take mu = mu2 = 0 by convention.
template <typename Float>
CUCOUNT_HOST_DEVICE inline void compute_pair_mu(
    const Float *diff,
    const Float *sposition1,
    const Float *sposition2,
    const Float *position1,
    const Float *position2,
    LOS_TYPE los,
    Float s,
    Float s2,
    bool required_mu,
    Float *mu,
    Float *mu2)
{
    Float d = 0.;

    if (los == LOS_FIRSTPOINT) {
        d = dot3(diff, sposition1);

        if (required_mu) {
            *mu = d / s;
        }
        else {
            *mu2 = (d * d) / s2;
        }
    }
    else if (los == LOS_ENDPOINT) {
        d = dot3(diff, sposition2);

        if (required_mu) {
            *mu = d / s;
        }
        else {
            *mu2 = (d * d) / s2;
        }
    }
    else if (los == LOS_MIDPOINT) {
        Float vlos[3];
        add3(vlos, position1, position2);

        d = dot3(diff, vlos);

        if (required_mu) {
            *mu = d / sqrt(dot3(vlos, vlos)) / s;
        }
        else {
            *mu2 = d * d / dot3(vlos, vlos) / s2;
        }
    }
    else {
        if (los == LOS_X) {
            d = diff[0];
        }
        else if (los == LOS_Y) {
            d = diff[1];
        }
        else if (los == LOS_Z) {
            d = diff[2];
        }

        if (required_mu) {
            *mu = d / s;
        }
        else {
            *mu2 = (d * d) / s2;
        }
    }

    if (required_mu) {
        *mu2 = (*mu) * (*mu);
    }

    if (s2 == 0) {
        *mu = 0.;
        *mu2 = 0.;
    }
}


// PIP pair weight from bitwise realizations, lifted verbatim from the CUDA
// add_weight2. The values are reinterpreted as 64-bit integers, so this
// expects Float = double (the wire format); a float32 caller must widen its
// bitwise columns first.
template <typename Float>
CUCOUNT_HOST_DEVICE inline Float pair_bitwise_weight(
    const Float *value1,
    const Float *value2,
    const IndexValue index_value1,
    const IndexValue index_value2,
    const BitwiseWeight bitwise)
{
    Float pair_bweight = bitwise.default_value;

    int nbits = bitwise.noffset;
    int nbits1 = 0;
    int nbits2 = 0;

    for (size_t iweight = 0;
         iweight < index_value1.size_bitwise_weight;
         iweight++) {
        long bweight1 =
            *((const long *) &(value1[index_value1.start_bitwise_weight + iweight]));

        long bweight2 =
            *((const long *) &(value2[index_value2.start_bitwise_weight + iweight]));

        nbits += CUCOUNT_POPCOUNT(bweight1 & bweight2);

        if (bitwise.p_nbits) {
            nbits1 += CUCOUNT_POPCOUNT(bweight1);
            nbits2 += CUCOUNT_POPCOUNT(bweight2);
        }
    }

    if (nbits != 0) {
        pair_bweight = bitwise.nrealizations / nbits;

        if (bitwise.p_nbits) {
            pair_bweight /=
                bitwise.p_correction_nbits[
                    nbits1 * bitwise.p_nbits + nbits2];
        }
    }

    return pair_bweight;
}

// Bin/lookup index searches and the angular-weight interpolation, lifted
// verbatim from the historical macro-generated CUDA code (double, the wire
// format). The angular axes arrive from Python already converted to
// ascending cos(theta), so callers feed dot(r1, r2) directly.

CUCOUNT_HOST_DEVICE inline int search_bin_index(
    double value,
    const double *edges,
    int nbins)
{
    if (!edges || nbins <= 0) return -1;
    if (value < edges[0] || value >= edges[nbins]) return -1;

    int lo = 0;
    int hi = nbins;

    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;

        if (value >= edges[mid]) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}


CUCOUNT_HOST_DEVICE inline int get_sep_bin_index(
    double value,
    const double *sep,
    int shape,
    BIN_TYPE bin,
    bool sep_is_edges)
{
    const int nbins = sep_is_edges ? shape : shape - 1;

    if (bin == BIN_CUSTOM) {
        return search_bin_index(value, sep, nbins);
    }

    const double min = sep[0];
    const double max = sep[nbins];

    if (value < min || value >= max) return -1;

    if (bin == BIN_LIN) {
        const double step = sep[1] - sep[0];
        int ibin = (int)floor((value - min) / step);
        return (ibin >= 0 && ibin < nbins) ? ibin : -1;
    }

    if (bin == BIN_LOG) {
        if (value <= 0.) return -1;
        const double logstep = log(sep[1] / sep[0]);
        int ibin = (int)floor(log(value / min) / logstep);
        return (ibin >= 0 && ibin < nbins) ? ibin : -1;
    }

    return -1;
}


CUCOUNT_HOST_DEVICE inline int get_interp_sep_index(
    double x,
    const double *sep,
    int nsep,
    BIN_TYPE bin,
    double *frac)
{
    *frac = 0.;
    if (!sep || nsep < 2) return -1;

    if (x < sep[0] || x > sep[nsep - 1]) return -1;

    if (x == sep[nsep - 1]) {
        *frac = 1.;
        return nsep - 2;
    }

    int ibin = get_sep_bin_index(x, sep, nsep, bin, false);
    if (ibin < 0) return -1;

    double dx = sep[ibin + 1] - sep[ibin];
    *frac = (dx != 0.) ? (x - sep[ibin]) / dx : 0.;
    return ibin;
}


// Angular (PIP) upweight: multilinear interpolation over `sep` axes,
// piecewise-constant over `edges` axes, 1 outside the tabulated range.
template <int ND>
CUCOUNT_HOST_DEVICE inline double lookup_angular_weight(
    const double (&costheta)[ND],
    const AngularWeight& angular)
{
    if (!angular.weight) return 1.;

    int i0[ND];
    double frac[ND];

    CUCOUNT_UNROLL
    for (int idim = 0; idim < ND; idim++) {
        if (angular.sep_is_edges[idim]) {
            i0[idim] = get_sep_bin_index(
                costheta[idim],
                angular.sep[idim],
                (int) angular.shape[idim],
                angular.bin[idim],
                true);
            frac[idim] = 0.;
        }
        else {
            i0[idim] = get_interp_sep_index(
                costheta[idim],
                angular.sep[idim],
                (int) angular.shape[idim],
                angular.bin[idim],
                &frac[idim]);
        }

        if (i0[idim] < 0) return 1.;
    }

    bool any_interp = false;

    CUCOUNT_UNROLL
    for (int idim = 0; idim < ND; idim++) {
        any_interp = any_interp || !angular.sep_is_edges[idim];
    }

    if (!any_interp) {
        size_t idx = 0;

        CUCOUNT_UNROLL
        for (int idim = 0; idim < ND; idim++) {
            idx = idx * (size_t) angular.shape[idim] + (size_t) i0[idim];
        }

        return angular.weight[idx];
    }

    double result = 0.;
    const int ncorners = 1 << ND;

    for (int icorner = 0; icorner < ncorners; icorner++) {
        size_t idx = 0;
        double wcorner = 1.;

        CUCOUNT_UNROLL
        for (int idim = 0; idim < ND; idim++) {
            int ibin = i0[idim];

            if (!angular.sep_is_edges[idim]) {
                const int upper = (icorner >> idim) & 1;
                ibin += upper;
                wcorner *= upper
                    ? frac[idim]
                    : (1. - frac[idim]);
            }

            idx = idx * (size_t) angular.shape[idim] + (size_t) ibin;
        }

        result += wcorner * angular.weight[idx];
    }

    return result;
}



// ---------------------------------------------------------------------------
// Triplet math: the local line-of-sight frame and the real spherical-harmonic
// basis the factorized triplet counts project onto. Lifted from the
// DEFINE_BUILD_LOS_FRAME and DEFINE_COMPUTE_SPHERICAL_HARMONICS macros so
// both backends run one copy.
// ---------------------------------------------------------------------------

// Same values the CUDA count3close.h fixes; redefining them there with the
// same tokens is well-formed, so either header may be included first.
#ifndef ELLMAX
#define ELLMAX 5
#endif
#ifndef MMAX_SIZE
#define MMAX_SIZE 6
#endif


template <typename Float>
CUCOUNT_HOST_DEVICE inline Float clamp_unit(Float x) {
    return MIN((Float)1., MAX((Float)-1., x));
}


template <typename Float>
CUCOUNT_HOST_DEVICE inline void normalize3(Float *out, const Float *x) {
    Float norm = (Float)0.;
    CUCOUNT_UNROLL
    for (int icoord = 0; icoord < 3; icoord++) norm += x[icoord] * x[icoord];
    norm = sqrt(norm);

    if (norm > (Float)0.) {
        CUCOUNT_UNROLL
        for (int icoord = 0; icoord < 3; icoord++) out[icoord] = x[icoord] / norm;
    }
    else {
        CUCOUNT_UNROLL
        for (int icoord = 0; icoord < 3; icoord++) out[icoord] = (Float)0.;
    }
}


template <typename Float>
CUCOUNT_HOST_DEVICE inline void cross3(Float *out, const Float *a, const Float *b) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}


// An orthonormal frame with ez along ez_in; the reference axis is switched
// away from z near the pole so the cross product stays well conditioned.
template <typename Float>
CUCOUNT_HOST_DEVICE inline void build_local_frame(const Float *ez_in,
                                                  Float local_frame[3][3]) {
    Float ref[3];

    if (fabs((double)ez_in[2]) < 0.9) {
        ref[0] = (Float)0.; ref[1] = (Float)0.; ref[2] = (Float)1.;
    }
    else {
        ref[0] = (Float)1.; ref[1] = (Float)0.; ref[2] = (Float)0.;
    }

    CUCOUNT_UNROLL
    for (int icoord = 0; icoord < 3; icoord++) local_frame[0][icoord] = ez_in[icoord];

    Float proj = (Float)0.;
    CUCOUNT_UNROLL
    for (int icoord = 0; icoord < 3; icoord++) proj += ref[icoord] * local_frame[0][icoord];

    Float tmp[3];
    CUCOUNT_UNROLL
    for (int icoord = 0; icoord < 3; icoord++)
        tmp[icoord] = ref[icoord] - proj * local_frame[0][icoord];

    normalize3(local_frame[1], tmp);
    cross3(local_frame[2], local_frame[0], local_frame[1]);
}


// Which line of sight the triplet projection uses: the multipole axis of
// either leg names it, and the endpoint frame is the default.
CUCOUNT_HOST_DEVICE inline LOS_TYPE get_count3_los(const BinAttrs &battrs12,
                                                   const BinAttrs &battrs13) {
    LOS_TYPE los = LOS_FIRSTPOINT;

    if (battrs12.ndim > 1 && battrs12.var[1] == VAR_POLE) los = battrs12.los[1];
    else if (battrs13.ndim > 1 && battrs13.var[1] == VAR_POLE) los = battrs13.los[1];

    return los;
}


template <typename Float>
CUCOUNT_HOST_DEVICE inline void build_los_frame(const Float *sposition1, LOS_TYPE los,
                                                Float local_frame[3][3]) {
    CUCOUNT_UNROLL
    for (int i = 0; i < 3; i++) {
        CUCOUNT_UNROLL
        for (int j = 0; j < 3; j++) local_frame[i][j] = (Float)0.;
    }

    if (los == LOS_X) {
        local_frame[0][0] = (Float)1.;
        local_frame[1][1] = (Float)1.;
        local_frame[2][2] = (Float)1.;
    }
    else if (los == LOS_Y) {
        local_frame[0][1] = (Float)1.;
        local_frame[1][2] = (Float)1.;
        local_frame[2][0] = (Float)1.;
    }
    else if (los == LOS_Z) {
        local_frame[0][2] = (Float)1.;
        local_frame[1][0] = (Float)1.;
        local_frame[2][1] = (Float)1.;
    }
    else {
        build_local_frame(sposition1, local_frame);
    }
}


// cos(m phi) and sin(m phi) by repeated angle addition from (cos phi, sin phi).
template <typename Float>
CUCOUNT_HOST_DEVICE inline void compute_trig_up_to_m(int mmax, Float c1, Float s1,
                                                     Float cm[MMAX_SIZE],
                                                     Float sm[MMAX_SIZE]) {
    CUCOUNT_UNROLL
    for (int m = 0; m < MMAX_SIZE; m++) { cm[m] = (Float)0.; sm[m] = (Float)0.; }

    mmax = MIN(mmax, ELLMAX);

    cm[0] = (Float)1.;
    sm[0] = (Float)0.;
    if (mmax <= 0) return;

    cm[1] = c1;
    sm[1] = s1;
    if (mmax <= 1) return;

    CUCOUNT_UNROLL
    for (int m = 2; m < MMAX_SIZE; m++) {
        if (m > mmax) break;
        cm[m] = c1 * cm[m - 1] - s1 * sm[m - 1];
        sm[m] = s1 * cm[m - 1] + c1 * sm[m - 1];
    }
}


// Normalized associated Legendre functions Pbar_ell^m(mu) as closed forms up
// to ell = 5, which is ELLMAX; the binding rejects anything higher rather
// than letting it clamp silently.
template <typename Float>
CUCOUNT_HOST_DEVICE inline void compute_pbar_all_lmax5(int ellmax, Float mu,
                                                       Float P[MMAX_SIZE][MMAX_SIZE]) {
    ellmax = MIN(ellmax, ELLMAX);

    Float x = clamp_unit(mu);
    Float x2 = x * x;
    Float s2 = MAX((Float)0., (Float)1. - x2);
    Float s = sqrt(s2);

    CUCOUNT_UNROLL
    for (int ell = 0; ell < MMAX_SIZE; ell++) {
        CUCOUNT_UNROLL
        for (int m = 0; m < MMAX_SIZE; m++) P[ell][m] = (Float)0.;
    }

    P[0][0] = (Float)1.;
    if (ellmax <= 0) return;

    P[1][0] = x;
    P[1][1] = -(Float)0.70710678118654752440 * s;
    if (ellmax <= 1) return;

    Float x3 = x2 * x;

    P[2][0] = ((Float)0.5) * (((Float)3.) * x2 - (Float)1.);
    P[2][1] = -(Float)1.22474487139158904910 * x * s;
    P[2][2] =  (Float)0.61237243569579452455 * s2;
    if (ellmax <= 2) return;

    Float s3 = s2 * s;

    P[3][0] =  ((Float)0.5) * (((Float)5.) * x3 - ((Float)3.) * x);
    P[3][1] = -(Float)0.43301270189221932338 * ((((Float)5.) * x2) - (Float)1.) * s;
    P[3][2] =  (Float)1.36930639376291527536 * x * s2;
    P[3][3] = -(Float)0.55901699437494742410 * s3;
    if (ellmax <= 3) return;

    Float x4 = x2 * x2;
    Float s4 = s2 * s2;

    P[4][0] =  ((Float)0.125) * (((Float)35.) * x4 - ((Float)30.) * x2 + (Float)3.);
    P[4][1] = -(Float)0.55901699437494742410 * x * ((((Float)7.) * x2) - (Float)3.) * s;
    P[4][2] =  (Float)0.39528470752104741743 * ((((Float)7.) * x2) - (Float)1.) * s2;
    P[4][3] = -(Float)0.93541434669348534640 * x * s3;
    P[4][4] =  (Float)0.52291251658379721705 * s4;
    if (ellmax <= 4) return;

    Float x5 = x4 * x;
    Float s5 = s4 * s;

    P[5][0] =  ((Float)0.125) * (((Float)63.) * x5 - ((Float)70.) * x3 + ((Float)15.) * x);
    P[5][1] = -(Float)0.19882122822827110675 * ((((Float)21.) * x4) - ((Float)14.) * x2 + (Float)1.) * s;
    P[5][2] =  (Float)0.48412291827592711065 * x * ((((Float)3.) * x2) - (Float)1.) * s2;
    P[5][3] = -(Float)0.52291251658379721705 * ((((Float)9.) * x2) - (Float)1.) * s3;
    P[5][4] =  (Float)1.16926793336685668103 * x * s4;
    P[5][5] = -(Float)0.70156076002011400980 * s5;
}
}  // namespace pairmath
}  // namespace cucount

#ifdef CUCOUNT_NO_CUDA
// common.h's convenience macros are not part of the shared contract; keep
// them from leaking into the SIMD translation units that include this header.
#undef FLOAT
#undef INT
#undef POPCOUNT
#undef MIN
#undef MAX
#undef CLIP
#endif

#endif  // _CUCOUNT_PAIR_MATH_
