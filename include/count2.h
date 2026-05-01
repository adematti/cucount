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

__device__ void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2, const MeshAttrs &mattrs);

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


__device__ int get_sep_bin_index(FLOAT value, const FLOAT *sep, int shape, BIN_TYPE bin, bool sep_is_edges);


__device__ inline int get_interp_sep_index(
    FLOAT x,
    const FLOAT *sep,
    int nsep,
    BIN_TYPE bin,
    FLOAT *frac)
{
    *frac = (FLOAT)0.;

    if (!sep || nsep < 2) return -1;

    // Preserve old interpolation convention:
    // valid range is [sep[0], sep[nsep - 1]]
    if (x < sep[0] || x > sep[nsep - 1]) return -1;

    // Explicitly include the upper endpoint.
    if (x == sep[nsep - 1]) {
        *frac = (FLOAT)1.;
        return nsep - 2;
    }

    int ibin = get_sep_bin_index(x, sep, nsep, bin, false);

    if (ibin < 0) return -1;

    FLOAT dx = sep[ibin + 1] - sep[ibin];
    *frac = (dx != (FLOAT)0.) ? (x - sep[ibin]) / dx : (FLOAT)0.;

    return ibin;
}


template <int ND> __device__ FLOAT lookup_angular_weight(
    const FLOAT (&costheta)[ND],
    const AngularWeight& angular)
{
    if (!angular.weight) return (FLOAT)1.;

    int i0[ND];
    FLOAT frac[ND];

    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        if (angular.sep_is_edges[idim]) {
            i0[idim] = get_sep_bin_index(
                costheta[idim],
                angular.sep[idim],
                (int) angular.shape[idim],
                angular.bin[idim],
                true
            );
            frac[idim] = (FLOAT)0.;
        } else {
            i0[idim] = get_interp_sep_index(
                costheta[idim],
                angular.sep[idim],
                (int) angular.shape[idim],
                angular.bin[idim],
                &frac[idim]
            );
        }

        if (i0[idim] < 0) return (FLOAT)1.;
    }

    // ------------------------------------------------------------
    // Check if any interpolation is needed
    // ------------------------------------------------------------
    bool any_interp = false;

    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        any_interp = any_interp || !angular.sep_is_edges[idim];
    }

    // ------------------------------------------------------------
    // Pure binned lookup
    // ------------------------------------------------------------
    if (!any_interp) {
        size_t idx = 0;

        #pragma unroll
        for (int idim = 0; idim < ND; idim++) {
            idx = idx * (size_t) angular.shape[idim] + (size_t) i0[idim];
        }

        return angular.weight[idx];
    }

    // ------------------------------------------------------------
    // Mixed / interpolation lookup
    // ------------------------------------------------------------
    FLOAT result = (FLOAT)0.;
    const int ncorners = 1 << ND;

    for (int icorner = 0; icorner < ncorners; icorner++) {
        size_t idx = 0;
        FLOAT wcorner = (FLOAT)1.;

        #pragma unroll
        for (int idim = 0; idim < ND; idim++) {
            int ibin = i0[idim];

            if (!angular.sep_is_edges[idim]) {
                const int upper = (icorner >> idim) & 1;
                ibin += upper;
                wcorner *= upper ? frac[idim] : ((FLOAT)1. - frac[idim]);
            }

            idx = idx * (size_t) angular.shape[idim] + (size_t) ibin;
        }

        result += wcorner * angular.weight[idx];
    }

    return result;
}


__device__ bool is_selected_pair(
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *position1,
    FLOAT *position2,
    const SelectionAttrs &sattrs,
    const MeshAttrs &mattrs);


__device__ void set_angular_bounds(
    const FLOAT *sposition,
    const MeshAttrs &mattrs,
    int *bounds);


__device__ void set_cartesian_bounds(
    const FLOAT *position,
    const MeshAttrs &mattrs,
    int *bounds);


// ============================================================================
// Candidate iteration
// ============================================================================

template <typename Op>
__device__ void for_each_candidate_angular(
    FLOAT *center_sposition,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    int bounds[2 * NDIM];
    set_angular_bounds(center_sposition, target_mattrs, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)target_mattrs.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(
                iphi,
                (int)target_mattrs.meshsize[1]);

            int icell = iphi_true + icth_n;

            int np = target_mesh.nparticles[icell];
            size_t cum = target_mesh.cumnparticles[icell];

            FLOAT *positions  = &(target_mesh.positions[NDIM * cum]);
            FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);
            FLOAT *values     = &(target_mesh.values[target_mesh.index_value.size * cum]);

            for (int j = 0; j < np; j++) {
                op(
                    cum + (size_t)j,
                    &(positions[NDIM * j]),
                    &(spositions[NDIM * j]),
                    &(values[target_mesh.index_value.size * j])
                );
            }
        }
    }
}


template <typename Op>
__device__ void for_each_candidate_cartesian(
    FLOAT *center_position,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    int bounds[2 * NDIM];
    set_cartesian_bounds(center_position, target_mattrs, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int)target_mattrs.meshsize[0])
                 * (int)target_mattrs.meshsize[2]
                 * (int)target_mattrs.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int)target_mattrs.meshsize[1])
                     * (int)target_mattrs.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int)target_mattrs.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int np = target_mesh.nparticles[icell];
                size_t cum = target_mesh.cumnparticles[icell];

                FLOAT *positions  = &(target_mesh.positions[NDIM * cum]);
                FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);
                FLOAT *values     = &(target_mesh.values[target_mesh.index_value.size * cum]);

                for (int j = 0; j < np; j++) {
                    op(
                        cum + (size_t)j,
                        &(positions[NDIM * j]),
                        &(spositions[NDIM * j]),
                        &(values[target_mesh.index_value.size * j])
                    );
                }
            }
        }
    }
}


template <MESH_TYPE TARGET_MESH_TYPE, typename Op>
__device__ void for_each_candidate(
    FLOAT *center_position,
    FLOAT *center_sposition,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    if constexpr (TARGET_MESH_TYPE == MESH_ANGULAR) {
        for_each_candidate_angular(
            center_sposition,
            target_mesh,
            target_mattrs,
            op);
    }
    else if constexpr (TARGET_MESH_TYPE == MESH_CARTESIAN) {
        for_each_candidate_cartesian(
            center_position,
            target_mesh,
            target_mattrs,
            op);
    }
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream);


#endif