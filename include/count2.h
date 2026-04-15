#ifndef _CUCOUNT_COUNT2_
#define _CUCOUNT_COUNT2_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"

// count2.h helpers

extern __device__ __constant__ MeshAttrs device_mattrs;
extern __device__ __constant__ SelectionAttrs device_sattrs;

__device__ void set_cartesian_bounds(FLOAT *bounds, int *shape);

__device__ void set_angular_bounds(FLOAT *bounds, int *shape);

__device__ int get_bessel(int ell1, int ell2, int m1, int m2);

__device__ void set_legendre(FLOAT *legendre, int ellmax, int m, int sign, double x, double y);

__global__ void reduce_add_kernel(const FLOAT *in, size_t size, FLOAT *out, size_t stride);

__device__ inline FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize);

__device__ inline int wrap_periodic_int(int idx, int meshsize);

__device__ void set_angular_bounds(FLOAT *sposition, int *bounds);

__device__ void set_cartesian_bounds(FLOAT *position, int *bounds);

__device__ inline void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2);

__device__ inline void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2);

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2);

__device__ inline bool is_selected_pair(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2);

__device__ void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2);

#define BESSEL_XMIN 0.1

__device__ FLOAT get_bessel(int ell, FLOAT x);

__global__ void reduce_add_kernel(const FLOAT *block_counts, size_t nblocks, FLOAT *counts, size_t csize);

__device__ inline int get_edge_bin_index(FLOAT value, const FLOAT *edges, int nbins);

__device__ inline int get_bin_index(const BinAttrs *battrs, int index, FLOAT value);

__device__ inline int get_interp_bin_index(FLOAT x, const FLOAT *sep, int nsep, FLOAT *frac);

template <int ND> __device__ inline FLOAT lookup_angular_weight(const FLOAT (&costheta)[ND], const AngularWeight& angular);

size_t get_count2_size(IndexValue index_value1, IndexValue index_value2, char names[][SIZE_NAME]);

template <typename Op> __device__ inline void for_each_selected_pair_angular(size_t ii, Mesh mesh1, Mesh mesh2, Op& op);

template <typename Op> __device__ inline void for_each_selected_pair_cartesian(size_t ii, Mesh mesh1, Mesh mesh2, Op& op);

void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream);


#endif