#ifndef _CUCOUNT_COUNT3CLOSE_
#define _CUCOUNT_COUNT3CLOSE_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"


#ifndef SQRT1_2
#define SQRT1_2 0.70710678118654752440
#endif

#ifndef SQRT3_8
#define SQRT3_8 0.61237243569579452455
#endif

#ifndef SQRT3_2
#define SQRT3_2 1.22474487139158904910
#endif

#ifndef SQRT15_8
#define SQRT15_8 1.36930639376291527536
#endif

#ifndef SQRT5_4
#define SQRT5_4 1.11803398874989484820
#endif

#ifndef SQRT10_8
#define SQRT10_8 1.11803398874989484820
#endif

#ifndef SQRT70_16
#define SQRT70_16 2.09165006633518886818
#endif


__device__ inline FLOAT clamp1(FLOAT x)
{
    return MIN((FLOAT)1., MAX((FLOAT)-1., x));
}


__device__ inline void normalize(FLOAT *out, const FLOAT *x)
{
    FLOAT norm = 0.;
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        norm += x[icoord] * x[icoord];
    }
    norm = sqrt(norm);

    if (norm > 0.) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            out[icoord] = x[icoord] / norm;
        }
    }
    else {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            out[icoord] = 0.;
        }
    }
}

__device__ inline void cross3(FLOAT *out, const FLOAT *a, const FLOAT *b)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void build_local_frame(const FLOAT *ez_in, FLOAT local_frame[3][NDIM])
{
    FLOAT ref[NDIM];
    if (fabs((double)ez_in[2]) < 0.9) {
        ref[0] = (FLOAT)0.;
        ref[1] = (FLOAT)0.;
        ref[2] = (FLOAT)1.;
    }
    else {
        ref[0] = (FLOAT)1.;
        ref[1] = (FLOAT)0.;
        ref[2] = (FLOAT)0.;
    }

    // ez
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        local_frame[0][icoord] = ez_in[icoord];
    }

    FLOAT proj = 0.;
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        proj += ref[icoord] * local_frame[0][icoord];
    }

    FLOAT tmp[NDIM];
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        tmp[icoord] = ref[icoord] - proj * local_frame[0][icoord];
    }

    // ex
    normalize(local_frame[1], tmp);
    // ey
    cross3(local_frame[2], local_frame[0], local_frame[1]);
}


__device__ inline void compute_pbar_lmax4(FLOAT mu, FLOAT P[5][5])
{
    FLOAT x  = clamp1(mu);
    FLOAT x2 = x * x;
    FLOAT x3 = x2 * x;
    FLOAT x4 = x2 * x2;

    FLOAT s2 = MAX((FLOAT)0., (FLOAT)1. - x2);
    FLOAT s  = sqrt(s2);
    FLOAT s3 = s2 * s;
    FLOAT s4 = s2 * s2;

    #pragma unroll
    for (int l = 0; l <= 4; l++) {
        #pragma unroll
        for (int m = 0; m <= 4; m++) {
            P[l][m] = (FLOAT)0.;
        }
    }

    P[0][0] = (FLOAT)1.;

    P[1][0] = x;
    P[1][1] = -(FLOAT)0.70710678118654752440 * s;

    P[2][0] = ((FLOAT)0.5) * (((FLOAT)3.) * x2 - (FLOAT)1.);
    P[2][1] = -(FLOAT)1.22474487139158904910 * x * s;
    P[2][2] =  (FLOAT)0.61237243569579452455 * s2;

    P[3][0] = ((FLOAT)0.5) * (((FLOAT)5.) * x3 - ((FLOAT)3.) * x);
    P[3][1] = -(FLOAT)0.43301270189221932338 * ((((FLOAT)5.) * x2) - (FLOAT)1.) * s;
    P[3][2] =  (FLOAT)1.36930639376291527536 * x * s2;
    P[3][3] = -(FLOAT)0.55901699437494742410 * s3;

    P[4][0] = ((FLOAT)0.125) * (((FLOAT)35.) * x4 - ((FLOAT)30.) * x2 + (FLOAT)3.);
    P[4][1] = -(FLOAT)0.55901699437494742410 * x * ((((FLOAT)7.) * x2) - (FLOAT)3.) * s;
    P[4][2] =  (FLOAT)0.39528470752104741743 * ((((FLOAT)7.) * x2) - (FLOAT)1.) * s2;
    P[4][3] = -(FLOAT)0.93541434669348534640 * x * s3;
    P[4][4] =  (FLOAT)0.52291251658379721705 * s4;
}

__device__ inline void compute_trig_mmax4(FLOAT c1, FLOAT s1, FLOAT cm[5], FLOAT sm[5])
{
    FLOAT c2 = c1 * c1;
    FLOAT s2 = s1 * s1;

    cm[0] = (FLOAT)1.;
    sm[0] = (FLOAT)0.;

    cm[1] = c1;
    sm[1] = s1;

    cm[2] = ((FLOAT)2.) * c2 - (FLOAT)1.;
    sm[2] = ((FLOAT)2.) * c1 * s1;

    cm[3] = c1 * ((((FLOAT)4.) * c2) - (FLOAT)3.);
    sm[3] = s1 * (((FLOAT)3.) - ((FLOAT)4.) * s2);

    cm[4] = ((FLOAT)8.) * c2 * c2 - ((FLOAT)8.) * c2 + (FLOAT)1.;
    sm[4] = ((FLOAT)4.) * c1 * s1 * ((((FLOAT)2.) * c2) - (FLOAT)1.);
}


typedef struct {
    size_t nprimaries;
    size_t npairs;
    size_t *offsets;    // device, size nprimaries + 1
    size_t *secondary;  // device, size npairs
} ClosePairs;


inline void free_device_close_pairs(ClosePairs *cpairs, DeviceMemoryBuffer *buffer)
{
    if (cpairs->offsets) {
        my_device_free(cpairs->offsets, buffer);
        cpairs->offsets = NULL;
    }
    if (cpairs->secondary) {
        my_device_free(cpairs->secondary, buffer);
        cpairs->secondary = NULL;
    }
    cpairs->nprimaries = 0;
    cpairs->npairs = 0;
}

void fill_close_pairs(ClosePairs *cpairs, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);


void count3_close(FLOAT *counts, ClosePairs close_ab, ClosePairs close_ac, const Mesh *list_mesh,
    MeshAttrs mattrs, BinAttrs *battrs[3], WeightAttrs wattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);



#endif