#ifndef _CUCOUNT_COUNT2_
#define _CUCOUNT_COUNT2_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"

// count2.h helpers

__device__ void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2);

__global__ void reduce_add_kernel(const FLOAT *in, size_t size, FLOAT *out, size_t stride);

__device__ FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize);

__device__ int wrap_periodic_int(int idx, int meshsize);

__device__ void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2);

__device__ void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2);

__device__ FLOAT dot(const FLOAT *position1, const FLOAT *position2);


#define BESSEL_XMIN 0.1

__device__ FLOAT get_bessel(int ell, FLOAT x);

__global__ void reduce_add_kernel(const FLOAT *block_counts, size_t nblocks, FLOAT *counts, size_t csize);

__device__ int get_edge_bin_index(FLOAT value, const FLOAT *edges, int nbins);

__device__ int get_bin_index(const BinAttrs *battrs, int index, FLOAT value);

__device__ int get_interp_bin_index(FLOAT x, const FLOAT *sep, int nsep, FLOAT *frac);


typedef struct DeviceCount2Layout {
    size_t nbins;
    size_t csize;
    size_t nells;
    size_t ells[10];
    bool ells_even;
} DeviceCount2Layout;


size_t fill_ells(const BinAttrs *battrs, int index, size_t *ells);

size_t get_count2_weight_names(IndexValue index_value1, IndexValue index_value2,
                        char names[][SIZE_NAME]);


template <int ND> __device__ FLOAT lookup_angular_weight(const FLOAT (&costheta)[ND], const AngularWeight& angular)
{
    if (!angular.weight) return (FLOAT)1.;

    // ------------------------------------------------------------
    // Interpolation mode: sep[] is provided
    // ------------------------------------------------------------
    if (angular.sep[0] != NULL) {
        int i0[ND];
        FLOAT frac[ND];

        #pragma unroll
        for (int idim = 0; idim < ND; idim++) {
            const int nsep = (int)angular.shape[idim];
            if (nsep < 2) return (FLOAT)1.;
            if (!angular.sep[idim]) return (FLOAT)1.;

            i0[idim] = get_interp_bin_index(costheta[idim], angular.sep[idim], nsep, &frac[idim]);

            if (i0[idim] < 0) return (FLOAT)1.;
        }

        FLOAT result = (FLOAT)0.;
        const int ncorners = 1 << ND;

        for (int icorner = 0; icorner < ncorners; icorner++) {
            size_t idx = 0;
            FLOAT wcorner = (FLOAT)1.;

            #pragma unroll
            for (int idim = 0; idim < ND; idim++) {
                const int upper = (icorner >> idim) & 1;
                const int ibin = i0[idim] + upper;

                idx = idx * (size_t)angular.shape[idim] + (size_t)ibin;
                wcorner *= upper ? frac[idim] : ((FLOAT)1. - frac[idim]);
            }

            result += wcorner * angular.weight[idx];
        }

        return result;
    }

    // ------------------------------------------------------------
    // Binned lookup mode: edges[] is provided
    // ------------------------------------------------------------
    int ibin[ND];

    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        const int nbins = (int)angular.shape[idim];
        if (nbins <= 0) return (FLOAT)1.;
        if (!angular.edges[idim]) return (FLOAT)1.;

        ibin[idim] = get_edge_bin_index(costheta[idim], angular.edges[idim], nbins);

        if (ibin[idim] < 0) return (FLOAT)1.;
    }

    size_t idx = 0;
    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        idx = idx * (size_t)angular.shape[idim] + (size_t)ibin[idim];
    }

    return angular.weight[idx];
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream);


#endif