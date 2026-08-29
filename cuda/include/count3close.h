#ifndef _CUCOUNT_COUNT3CLOSE_
#define _CUCOUNT_COUNT3CLOSE_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"


#define ELLMAX 5
#define MMAX_SIZE 6


typedef enum {
    CLOSE_PAIR_12,
    CLOSE_PAIR_13,
    CLOSE_PAIR_23
} CLOSE_PAIR;


typedef struct DeviceCount3Layout {
    size_t nbins;
    size_t nprojs1;
    size_t nprojs2;
    size_t nprojs;
    size_t csize;
    size_t nells1;
    size_t nells2;
    int ellmax1;
    int ellmax2;
    size_t ells1[MMAX_SIZE];
    size_t ells2[MMAX_SIZE];
} DeviceCount3Layout;


DeviceCount3Layout make_device_count3_layout(const BinAttrs battrs12, const BinAttrs battrs13, const BinAttrs battrs23);


void count3_close(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs1,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    SelectionAttrs veto23,
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs,
    CLOSE_PAIR close_pair,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream);



#define DEFINE_BUILD_LOS_FRAME                                                     \
__device__ inline void normalize(FLOAT *out, const FLOAT *x)                       \
{                                                                                  \
    FLOAT norm = (FLOAT)0.;                                                        \
    _Pragma("unroll")                                                              \
    for (int icoord = 0; icoord < NDIM; icoord++) {                                \
        norm += x[icoord] * x[icoord];                                             \
    }                                                                              \
    norm = sqrt(norm);                                                             \
                                                                                   \
    if (norm > (FLOAT)0.) {                                                        \
        _Pragma("unroll")                                                          \
        for (int icoord = 0; icoord < NDIM; icoord++) {                            \
            out[icoord] = x[icoord] / norm;                                        \
        }                                                                          \
    }                                                                              \
    else {                                                                         \
        _Pragma("unroll")                                                          \
        for (int icoord = 0; icoord < NDIM; icoord++) {                            \
            out[icoord] = (FLOAT)0.;                                               \
        }                                                                          \
    }                                                                              \
}                                                                                  \
                                                                                   \
__device__ inline void cross3(FLOAT *out, const FLOAT *a, const FLOAT *b)          \
{                                                                                  \
    out[0] = a[1] * b[2] - a[2] * b[1];                                            \
    out[1] = a[2] * b[0] - a[0] * b[2];                                            \
    out[2] = a[0] * b[1] - a[1] * b[0];                                            \
}                                                                                  \
                                                                                   \
__device__ inline void build_local_frame(                                          \
    const FLOAT *ez_in,                                                            \
    FLOAT local_frame[3][NDIM])                                                    \
{                                                                                  \
    FLOAT ref[NDIM];                                                               \
                                                                                   \
    if (fabs((double)ez_in[2]) < 0.9) {                                            \
        ref[0] = (FLOAT)0.;                                                        \
        ref[1] = (FLOAT)0.;                                                        \
        ref[2] = (FLOAT)1.;                                                        \
    }                                                                              \
    else {                                                                         \
        ref[0] = (FLOAT)1.;                                                        \
        ref[1] = (FLOAT)0.;                                                        \
        ref[2] = (FLOAT)0.;                                                        \
    }                                                                              \
                                                                                   \
    _Pragma("unroll")                                                              \
    for (int icoord = 0; icoord < NDIM; icoord++) {                                \
        local_frame[0][icoord] = ez_in[icoord];                                    \
    }                                                                              \
                                                                                   \
    FLOAT proj = (FLOAT)0.;                                                        \
    _Pragma("unroll")                                                              \
    for (int icoord = 0; icoord < NDIM; icoord++) {                                \
        proj += ref[icoord] * local_frame[0][icoord];                              \
    }                                                                              \
                                                                                   \
    FLOAT tmp[NDIM];                                                               \
    _Pragma("unroll")                                                              \
    for (int icoord = 0; icoord < NDIM; icoord++) {                                \
        tmp[icoord] = ref[icoord] - proj * local_frame[0][icoord];                 \
    }                                                                              \
                                                                                   \
    normalize(local_frame[1], tmp);                                                \
    cross3(local_frame[2], local_frame[0], local_frame[1]);                        \
}                                                                                  \
                                                                                   \
__device__ inline LOS_TYPE get_count3_los(                                         \
    BinAttrs battrs12,                                                             \
    BinAttrs battrs13)                                                             \
{                                                                                  \
    LOS_TYPE los = LOS_FIRSTPOINT;                                                 \
                                                                                   \
    if (battrs12.ndim > 1 && battrs12.var[1] == VAR_POLE) {                        \
        los = battrs12.los[1];                                                     \
    }                                                                              \
    else if (battrs13.ndim > 1 && battrs13.var[1] == VAR_POLE) {                   \
        los = battrs13.los[1];                                                     \
    }                                                                              \
                                                                                   \
    return los;                                                                    \
}                                                                                  \
                                                                                   \
__device__ inline void build_los_frame(                                            \
    FLOAT *sposition1,                                                             \
    LOS_TYPE los,                                                                  \
    FLOAT local_frame[3][NDIM])                                                    \
{                                                                                  \
    _Pragma("unroll")                                                              \
    for (int i = 0; i < 3; i++) {                                                  \
        _Pragma("unroll")                                                          \
        for (int j = 0; j < NDIM; j++) {                                           \
            local_frame[i][j] = (FLOAT)0.;                                         \
        }                                                                          \
    }                                                                              \
                                                                                   \
    if (los == LOS_X) {                                                            \
        local_frame[0][0] = (FLOAT)1.;                                             \
        local_frame[1][1] = (FLOAT)1.;                                             \
        local_frame[2][2] = (FLOAT)1.;                                             \
    }                                                                              \
    else if (los == LOS_Y) {                                                       \
        local_frame[0][1] = (FLOAT)1.;                                             \
        local_frame[1][2] = (FLOAT)1.;                                             \
        local_frame[2][0] = (FLOAT)1.;                                             \
    }                                                                              \
    else if (los == LOS_Z) {                                                       \
        local_frame[0][2] = (FLOAT)1.;                                             \
        local_frame[1][0] = (FLOAT)1.;                                             \
        local_frame[2][1] = (FLOAT)1.;                                             \
    }                                                                              \
    else {                                                                         \
        build_local_frame(sposition1, local_frame);                                \
    }                                                                              \
}




#define DEFINE_COMPUTE_SPHERICAL_HARMONICS                                         \
__device__ inline void compute_trig_up_to_m(                                       \
    int mmax,                                                                      \
    FLOAT c1,                                                                      \
    FLOAT s1,                                                                      \
    FLOAT cm[MMAX_SIZE],                                                           \
    FLOAT sm[MMAX_SIZE])                                                           \
{                                                                                  \
    _Pragma("unroll")                                                              \
    for (int m = 0; m < MMAX_SIZE; m++) {                                          \
        cm[m] = (FLOAT)0.;                                                         \
        sm[m] = (FLOAT)0.;                                                         \
    }                                                                              \
                                                                                   \
    mmax = MIN(mmax, ELLMAX);                                                      \
                                                                                   \
    cm[0] = (FLOAT)1.;                                                             \
    sm[0] = (FLOAT)0.;                                                             \
    if (mmax <= 0) return;                                                         \
                                                                                   \
    cm[1] = c1;                                                                    \
    sm[1] = s1;                                                                    \
    if (mmax <= 1) return;                                                         \
                                                                                   \
    _Pragma("unroll")                                                              \
    for (int m = 2; m < MMAX_SIZE; m++) {                                          \
        if (m > mmax) break;                                                       \
        cm[m] = c1 * cm[m - 1] - s1 * sm[m - 1];                                  \
        sm[m] = s1 * cm[m - 1] + c1 * sm[m - 1];                                  \
    }                                                                              \
}                                                                                  \
__device__ inline void compute_pbar_all_lmax5(int ellmax, FLOAT mu, FLOAT P[MMAX_SIZE][MMAX_SIZE])                       \
{                                                                                                                          \
    ellmax = MIN(ellmax, ELLMAX);                                                                                          \
                                                                                                                           \
    FLOAT x  = clamp1(mu);                                                                                                 \
    FLOAT x2 = x * x;                                                                                                      \
    FLOAT s2 = MAX((FLOAT)0., (FLOAT)1. - x2);                                                                             \
    FLOAT s  = sqrt(s2);                                                                                                   \
                                                                                                                           \
    _Pragma("unroll")                                                                                                      \
    for (int ell = 0; ell < MMAX_SIZE; ell++) {                                                                            \
        _Pragma("unroll")                                                                                                  \
        for (int m = 0; m < MMAX_SIZE; m++) P[ell][m] = (FLOAT)0.;                                                        \
    }                                                                                                                      \
                                                                                                                           \
    P[0][0] = (FLOAT)1.;                                                                                                   \
    if (ellmax <= 0) return;                                                                                               \
                                                                                                                           \
    P[1][0] = x;                                                                                                           \
    P[1][1] = -(FLOAT)0.70710678118654752440 * s;                                                                          \
    if (ellmax <= 1) return;                                                                                               \
                                                                                                                           \
    FLOAT x3 = x2 * x;                                                                                                     \
                                                                                                                           \
    P[2][0] = ((FLOAT)0.5) * (((FLOAT)3.) * x2 - (FLOAT)1.);                                                              \
    P[2][1] = -(FLOAT)1.22474487139158904910 * x * s;                                                                      \
    P[2][2] =  (FLOAT)0.61237243569579452455 * s2;                                                                         \
    if (ellmax <= 2) return;                                                                                               \
                                                                                                                           \
    FLOAT s3 = s2 * s;                                                                                                     \
                                                                                                                           \
    P[3][0] =  ((FLOAT)0.5) * (((FLOAT)5.) * x3 - ((FLOAT)3.) * x);                                                       \
    P[3][1] = -(FLOAT)0.43301270189221932338 * ((((FLOAT)5.) * x2) - (FLOAT)1.) * s;                                     \
    P[3][2] =  (FLOAT)1.36930639376291527536 * x * s2;                                                                     \
    P[3][3] = -(FLOAT)0.55901699437494742410 * s3;                                                                         \
    if (ellmax <= 3) return;                                                                                               \
                                                                                                                           \
    FLOAT x4 = x2 * x2;                                                                                                    \
    FLOAT s4 = s2 * s2;                                                                                                    \
                                                                                                                           \
    P[4][0] =  ((FLOAT)0.125) * (((FLOAT)35.) * x4 - ((FLOAT)30.) * x2 + (FLOAT)3.);                                     \
    P[4][1] = -(FLOAT)0.55901699437494742410 * x * ((((FLOAT)7.) * x2) - (FLOAT)3.) * s;                                 \
    P[4][2] =  (FLOAT)0.39528470752104741743 * ((((FLOAT)7.) * x2) - (FLOAT)1.) * s2;                                    \
    P[4][3] = -(FLOAT)0.93541434669348534640 * x * s3;                                                                     \
    P[4][4] =  (FLOAT)0.52291251658379721705 * s4;                                                                         \
    if (ellmax <= 4) return;                                                                                               \
                                                                                                                           \
    FLOAT x5 = x4 * x;                                                                                                     \
    FLOAT s5 = s4 * s;                                                                                                     \
                                                                                                                           \
    P[5][0] =  ((FLOAT)0.125) * (((FLOAT)63.) * x5 - ((FLOAT)70.) * x3 + ((FLOAT)15.) * x);                              \
    P[5][1] = -(FLOAT)0.19882122822827110675 * ((((FLOAT)21.) * x4) - ((FLOAT)14.) * x2 + (FLOAT)1.) * s;               \
    P[5][2] =  (FLOAT)0.48412291827592711065 * x * ((((FLOAT)3.) * x2) - (FLOAT)1.) * s2;                                \
    P[5][3] = -(FLOAT)0.52291251658379721705 * ((((FLOAT)9.) * x2) - (FLOAT)1.) * s3;                                    \
    P[5][4] =  (FLOAT)1.16926793336685668103 * x * s4;                                                                     \
    P[5][5] = -(FLOAT)0.70156076002011400980 * s5;                                                                         \
}

#endif