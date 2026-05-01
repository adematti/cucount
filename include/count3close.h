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
    int mmax, FLOAT c1, FLOAT s1, FLOAT cm[ELLMAX], FLOAT sm[ELLMAX]);


__device__ void compute_pbar_row_lmax5(
    int ell, int mmax, FLOAT mu, FLOAT Prow[ELLMAX]);



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