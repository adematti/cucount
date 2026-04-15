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

__device__ FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize);

__device__ int wrap_periodic_int(int idx, int meshsize);

__device__ void set_angular_bounds(FLOAT *sposition, int *bounds);

__device__ void set_cartesian_bounds(FLOAT *position, int *bounds);

__device__ void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2);

__device__ void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2);

__device__ FLOAT dot(const FLOAT *position1, const FLOAT *position2);

__device__ bool is_selected_pair(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2);

__device__ void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2);

#define BESSEL_XMIN 0.1

__device__ FLOAT get_bessel(int ell, FLOAT x);

__global__ void reduce_add_kernel(const FLOAT *block_counts, size_t nblocks, FLOAT *counts, size_t csize);

__device__ int get_edge_bin_index(FLOAT value, const FLOAT *edges, int nbins);

__device__ int get_bin_index(const BinAttrs *battrs, int index, FLOAT value);

__device__ int get_interp_bin_index(FLOAT x, const FLOAT *sep, int nsep, FLOAT *frac);

template <int ND> __device__ FLOAT lookup_angular_weight(const FLOAT (&costheta)[ND], const AngularWeight& angular);

size_t get_count2_size(IndexValue index_value1, IndexValue index_value2, char names[][SIZE_NAME]);

template <typename Op> __device__ void for_each_selected_pair_angular(size_t ii, Mesh mesh1, Mesh mesh2, Op& op)
{
    FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);

    int bounds[2 * NDIM];
    set_angular_bounds(sposition1, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * device_mattrs.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, device_mattrs.meshsize[1]);
            int icell = iphi_true + icth_n;

            int np2 = mesh2.nparticles[icell];
            size_t cum2 = mesh2.cumnparticles[icell];

            FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
            FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
            FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);

            for (int jj = 0; jj < np2; jj++) {
                FLOAT *position2 = &(positions2[NDIM * jj]);
                FLOAT *sposition2 = &(spositions2[NDIM * jj]);
                FLOAT *value2 = &(values2[mesh2.index_value.size * jj]);

                if (!is_selected_pair(sposition1, sposition2, position1, position2)) continue;

                op(ii, cum2 + (size_t)jj, position1, sposition1, position2, sposition2, value2);
            }
        }
    }
}

template <typename Op> __device__ void for_each_selected_pair_cartesian(size_t ii, Mesh mesh1, Mesh mesh2, Op& op)
{
    FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);

    int bounds[2 * NDIM];
    set_cartesian_bounds(position1, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int) device_mattrs.meshsize[0])
                 * device_mattrs.meshsize[2] * device_mattrs.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int) device_mattrs.meshsize[1])
                     * device_mattrs.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int) device_mattrs.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int np2 = mesh2.nparticles[icell];
                size_t cum2 = mesh2.cumnparticles[icell];

                FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
                FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
                FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);

                for (int jj = 0; jj < np2; jj++) {
                    FLOAT *position2 = &(positions2[NDIM * jj]);
                    FLOAT *sposition2 = &(spositions2[NDIM * jj]);
                    FLOAT *value2 = &(values2[mesh2.index_value.size * jj]);

                    if (!is_selected_pair(sposition1, sposition2, position1, position2)) continue;

                    op(ii, cum2 + (size_t)jj, position1, sposition1, position2, sposition2, value2);
                }
            }
        }
    }
}

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