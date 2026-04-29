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


typedef enum {
    CLOSE_PAIR_12,
    CLOSE_PAIR_13,
    CLOSE_PAIR_23
} CLOSE_PAIR;


__device__ bool is_selected_pair_with_sattrs(
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *position1,
    FLOAT *position2,
    const SelectionAttrs &sattrs);


__device__ void set_angular_bounds_from_attrs(
    const FLOAT *sposition,
    const MeshAttrs &mattrs,
    int *bounds);


__device__ void set_cartesian_bounds_from_attrs(
    const FLOAT *position,
    const MeshAttrs &mattrs,
    int *bounds);


__device__ void build_local_frame(const FLOAT *ez_in, FLOAT local_frame[3][NDIM]);


__device__ void compute_trig_up_to_m(
    int mmax, FLOAT c1, FLOAT s1, FLOAT cm[5], FLOAT sm[5]);


__device__ void compute_pbar_row_lmax4(
    int ell, int mmax, FLOAT mu, FLOAT Prow[5]);



typedef struct DeviceCount3Layout {
    size_t nbins;
    size_t nprojs;
    size_t csize;
    size_t nells1;
    size_t nells2;
    size_t ells1[4];
    size_t ells2[4];
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


#endif