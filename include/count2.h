#ifndef _CUCOUNT_COUNT2_
#define _CUCOUNT_COUNT2_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"

// count2.h helpers

__device__ __constant__ MeshAttrs device_mattrs;
__device__ __constant__ SelectionAttrs device_sattrs;


__device__ inline FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize) {
    FLOAT half = 0.5 * boxsize;
    FLOAT x = dxyz + half;
    x = fmod(x, boxsize);  // negative if x is negative
    if (x < 0) x += boxsize;
    return x - half;
}


__device__ inline int wrap_periodic_int(int idx, int meshsize) {
    int r = idx % meshsize;
    return (r < 0) ? r + meshsize : r;
}


__device__ void set_angular_bounds(FLOAT *sposition, int *bounds) {
    FLOAT cth, phi;
    int icth, iphi;
    FLOAT theta, th_hi, th_lo;
    FLOAT phi_hi, phi_lo;
    FLOAT cth_max, cth_min;

    cth = sposition[2];
    phi = atan2(sposition[1], sposition[0]);
    if (phi < 0) phi += 2 * M_PI; // Wrap phi into [0, 2π]

    // Compute pixel indices
    icth = (cth >= 1) ? (device_mattrs.meshsize[0] - 1) : (int)(0.5 * (1 + cth) * device_mattrs.meshsize[0]);
    iphi = (int)(0.5 * phi / M_PI * device_mattrs.meshsize[1]);

    // Compute angular bounds
    theta = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.5)) / device_mattrs.meshsize[0]);
    th_hi = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.0)) / device_mattrs.meshsize[0]);
    th_lo = acos(-1.0 + 2.0 * ((FLOAT)(icth + 1.0)) / device_mattrs.meshsize[0]);
    phi_hi = 2 * M_PI * ((FLOAT)(iphi + 1.0) / device_mattrs.meshsize[1]);
    phi_lo = 2 * M_PI * ((FLOAT)(iphi + 0.0) / device_mattrs.meshsize[1]);
    FLOAT smax = acos(device_mattrs.smax);

    // Handle edge cases for angular bounds
    if (th_hi > M_PI - smax) {
        cth_min = -1;
        cth_max = cos(th_lo - smax);
        bounds[2] = 0;
        bounds[3] = device_mattrs.meshsize[1] - 1;
    } else if (th_lo < smax) {
        cth_min = cos(th_hi + smax);
        cth_max = 1;
        bounds[2] = 0;
        bounds[3] = device_mattrs.meshsize[1] - 1;
    } else {  // theta-stripe
        FLOAT dphi, calpha = cos(smax);
        cth_min = cos(th_hi + smax);
        cth_max = cos(th_lo - smax);

        if (theta < 0.5 * M_PI) {
            FLOAT cth_lo = cos(th_lo);
            dphi = acos(sqrt((calpha * calpha - cth_lo * cth_lo) / (1 - cth_lo * cth_lo)));
        } else {
            FLOAT cth_hi = cos(th_hi);
            dphi = acos(sqrt((calpha * calpha - cth_hi * cth_hi) / (1 - cth_hi * cth_hi)));
        }

        if (dphi < M_PI) {
            FLOAT phi_min = phi_lo - dphi;
            FLOAT phi_max = phi_hi + dphi;
            bounds[2] = (int)(floor(0.5 * phi_min / M_PI * device_mattrs.meshsize[1]));
            bounds[3] = (int)(floor(0.5 * phi_max / M_PI * device_mattrs.meshsize[1]));
        } else {
            bounds[2] = 0;
            bounds[3] = device_mattrs.meshsize[1] - 1;
        }
    }

    // Apply mask bounds
    cth_min = MAX(cth_min, device_mattrs.boxcenter[0] - device_mattrs.boxsize[0] / 2.);
    cth_max = MIN(cth_max, device_mattrs.boxcenter[0] + device_mattrs.boxsize[0] / 2.);

    bounds[0] = (int)(0.5 * (1 + cth_min) * device_mattrs.meshsize[0]);
    bounds[1] = (int)(0.5 * (1 + cth_max) * device_mattrs.meshsize[0]);
    if (bounds[0] < 0) bounds[0] = 0;
    if (bounds[1] >= device_mattrs.meshsize[0]) bounds[1] = device_mattrs.meshsize[0] - 1;
}


__device__ void set_cartesian_bounds(FLOAT *position, int *bounds) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        int meshsize = (int) device_mattrs.meshsize[axis];
        FLOAT offset = device_mattrs.boxcenter[axis] - device_mattrs.boxsize[axis] / 2;
        int index = (int) floor((position[axis] - offset) * meshsize / device_mattrs.boxsize[axis]);
        index = wrap_periodic_int(index, meshsize);
        int delta = (int) ceil(device_mattrs.smax / device_mattrs.boxsize[axis] * meshsize);
        bounds[2 * axis] = index - delta;
        bounds[2 * axis + 1] = index + delta;
        if (device_mattrs.periodic == 0) {
            bounds[2 * axis] = MAX(bounds[2 * axis], 0);
            bounds[2 * axis + 1] = MIN(bounds[2 * axis + 1], meshsize - 1);
        }
        else if (2 * delta + 1 >= meshsize) {  // make sure to visit each cell at most once
            bounds[2 * axis] = 0;
            bounds[2 * axis + 1] = meshsize - 1;
        }
    }
}


__device__ inline void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        add[axis] = position1[axis] + position2[axis];
        //if (device_mattrs.periodic) add[axis] = wrap_periodic_float(add[axis], device_mattrs.boxsize[axis]);
    }
}

__device__ inline void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        diff[axis] = position1[axis] - position2[axis];
        if (device_mattrs.periodic) diff[axis] = wrap_periodic_float(diff[axis], device_mattrs.boxsize[axis]);
    }
}

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2) {
    FLOAT d = 0.;
    for (size_t axis = 0; axis < NDIM; axis++) d += position1[axis] * position2[axis];
    return d;
}

__device__ inline bool is_selected_pair(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    for (size_t i = 0; i < device_sattrs.ndim; i++) {
        int var = device_sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= device_sattrs.smin[i]) && (costheta <= device_sattrs.smax[i]);
            //if (selected) printf("costheta %d %f %.4f %.4f ", i, costheta, device_sattrs.smin[i], device_sattrs.smax[i]);
        }
        if (var == VAR_S) {
            FLOAT diff[NDIM];
            difference(diff, position2, position1);
            const FLOAT s2 = dot(diff, diff);
            selected &= (s2 >= device_sattrs.smin[i] * device_sattrs.smin[i]) && (s2 <= device_sattrs.smax[i] * device_sattrs.smax[i]);
        }
    }
    return selected;
}


__device__ void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2) {
    if ((ellmin % 2 == 0) && (ellstep % 2 == 0)) {
        for (int ell = ellmin; ell <= ellmax; ell+=ellstep) {
            if (ell == 0) {
                legendre_cache[ell] = 1.;
            }
            else if (ell == 2) {
                legendre_cache[ell] = (3.0 * mu2 - 1.0) / 2.0;
            }
            else if (ell == 4) {
                FLOAT mu4 = mu2 * mu2;
                legendre_cache[ell] = (35.0 * mu4 - 30.0 * mu2 + 3.0) / 8.0;
            }
            else if (ell == 6) {
                FLOAT mu4 = mu2 * mu2;
                FLOAT mu6 = mu4 * mu2;
                legendre_cache[ell] = (231.0 * mu6 - 315.0 * mu4 + 105.0 * mu2 - 5.0) / 16.0;
            }
            else if (ell == 6) {
                FLOAT mu4 = mu2 * mu2;
                FLOAT mu6 = mu4 * mu2;
                FLOAT mu8 = mu4 * mu4;
                legendre_cache[ell] = (6435.0 * mu8 - 12012.0 * mu6 + 6930.0 * mu4 - 1260.0 * mu2 + 35.0) / 128.0;
            }
            else {
                legendre_cache[ell] = 0.;
            }
        }
    }
    else {
        legendre_cache[0] = 1.0; // P_0(mu) = 1
        legendre_cache[1] = mu; // P_1(mu) = mu
        for (int ell = 2; ell <= ellmax; ell++) {
            legendre_cache[ell] = ((2.0 * ell - 1.0) * mu * legendre_cache[ell - 1] - (ell - 1.0) * legendre_cache[ell - 2]) / ell;
        }
    }
}


#define BESSEL_XMIN 0.1


__device__ FLOAT get_bessel(int ell, FLOAT x) {
    if (x < BESSEL_XMIN) {
        FLOAT x2 = x * x;
        switch (ell) {
            case 0:
                return 1. - x2 / 6. + x2 * x2 / 120. - x2 * x2 * x2 / 5040.;
            case 2:
                return x2 / 15. - x2 * x2 / 210. + x2 * x2 * x2 / 11340.;
            case 4:
                return x2 * x2 / 945.- x2 * x2 * x2 / 10395.;
            default:
                return 0.0;  // optionally handle unsupported ℓ values
        }
    } else {
        FLOAT invx  = 1.0 / x;
        FLOAT invx2 = invx * invx;
        FLOAT invx3, invx4;
        switch (ell) {
            case 0:
                return sin(x) * invx;
            case 2:
                return (3.0 * invx2 - 1.0) * sin(x) * invx - 3.0 * cos(x) * invx2;
            case 4:
                invx3 = invx2 * invx;
                invx4 = invx2 * invx2;
                //return sin(x) * (invx - 45.0 * invx3 + 105.0 * invx4) - cos(x) * (10.0 * invx - 105.0 * invx3);
                return 5 * (2 * invx2 - 21 * invx4) * cos(x) + (invx - 45 * invx3 + 105 * invx2 * invx3) * sin(x);
            default:
                return 0.0;
        }
    }
}


__global__ void reduce_add_kernel(
    const FLOAT *block_counts,
    size_t nblocks,
    FLOAT *counts,
    size_t csize)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < csize; i += stride) {
        FLOAT sum = 0;
        for (size_t iblock = 0; iblock < nblocks; iblock++) {
            sum += block_counts[iblock * csize + i];
        }
        counts[i] += sum;
    }
}


template <typename Op>
__device__ inline void for_each_selected_pair_angular(
    size_t ii,
    Mesh mesh1,
    Mesh mesh2,
    Op& op)
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

template <typename Op>
__device__ inline void for_each_selected_pair_cartesian(
    size_t ii,
    Mesh mesh1,
    Mesh mesh2,
    Op& op)
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


__device__ inline int get_edge_bin_index(FLOAT value, const FLOAT *edges, int nbins)
{
    if (!edges || nbins <= 0) return -1;
    if ((value >= edges[nbins]) || (value < edges[0])) return -1;

    int ibin;
    for (ibin = nbins - 1; ibin >= 0; ibin--) {
        if (value >= edges[ibin]) break;
    }

    return ibin;
}


__device__ inline int get_bin_index(const BinAttrs *battrs, int index, FLOAT value)
{
    int ibin;

    if (battrs->asize[index] > 0) {
        ibin = get_edge_bin_index(value, battrs->array[index], (int)battrs->asize[index] - 1);
    }
    else {
        ibin = (int) floor((value - battrs->min[index]) / battrs->step[index]);
    }

    if (ibin < 0 || ibin >= battrs->shape[index]) return -1;
    return ibin;
}


__device__ inline int get_interp_bin_index(
    FLOAT x,
    const FLOAT *sep,
    int nsep,
    FLOAT *frac)
{
    *frac = (FLOAT)0.;

    if (!sep || nsep < 2) return -1;

    // Same convention as your original 1D code:
    // outside [sep[0], sep[nsep-1]] => no reweighting
    if (x < sep[0] || x > sep[nsep - 1]) return -1;

    for (int i = 0; i < nsep - 1; i++) {
        if (x <= sep[i + 1]) {
            FLOAT dx = sep[i + 1] - sep[i];
            *frac = (dx != (FLOAT)0.) ? (x - sep[i]) / dx : (FLOAT)0.;
            return i;
        }
    }

    return -1;
}


template <int ND> _device__ inline FLOAT lookup_angular_weight(const FLOAT (&costheta)[ND], const WeightAngular& angular)
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


void copy_bin_attrs_to_device(
    BinAttrs *device_battrs,
    const BinAttrs *host_battrs,
    DeviceMemoryBuffer *buffer)
{
    *device_battrs = *host_battrs;

    for (size_t idim = 0; idim < MAX_NBIN; idim++) {
        device_battrs->array[idim] = NULL;
    }

    for (size_t idim = 0; idim < host_battrs->ndim; idim++) {
        if (host_battrs->asize[idim] > 0) {
            FLOAT *device_array = (FLOAT*) my_device_malloc(
                host_battrs->asize[idim] * sizeof(FLOAT), buffer);

            CUDA_CHECK(cudaMemcpy(
                device_array,
                host_battrs->array[idim],
                host_battrs->asize[idim] * sizeof(FLOAT),
                cudaMemcpyHostToDevice));

            device_battrs->array[idim] = device_array;
        }
    }
}

void free_device_bin_attrs(
    BinAttrs *device_battrs,
    DeviceMemoryBuffer *buffer)
{
    for (size_t idim = 0; idim < device_battrs->ndim; idim++) {
        if (device_battrs->array[idim]) {
            my_device_free(device_battrs->array[idim], buffer);
            device_battrs->array[idim] = NULL;
        }
    }
}


void copy_weight_attrs_to_device(
    WeightAttrs *device_attrs,
    const WeightAttrs *host_attrs,
    DeviceMemoryBuffer *buffer)
{
    *device_attrs = *host_attrs;

    // -----------------
    // BitwiseWeight
    // -----------------
    device_attrs->bitwise.p_correction_nbits = NULL;

    if (host_attrs->bitwise.p_nbits > 0) {
        const size_t size =
            host_attrs->bitwise.p_nbits * host_attrs->bitwise.p_nbits;

        FLOAT *device_p_correction_nbits = (FLOAT*) my_device_malloc(
            size * sizeof(FLOAT), buffer);

        CUDA_CHECK(cudaMemcpy(
            device_p_correction_nbits,
            host_attrs->bitwise.p_correction_nbits,
            size * sizeof(FLOAT),
            cudaMemcpyHostToDevice));

        device_attrs->bitwise.p_correction_nbits = device_p_correction_nbits;
    }

    // -----------------
    // AngularWeight
    // -----------------
    for (size_t idim = 0; idim < MAX_NBIN; idim++) {
        device_attrs->angular.sep[idim] = NULL;
        device_attrs->angular.edges[idim] = NULL;
    }
    device_attrs->angular.weight = NULL;

    if (host_attrs->angular.ndim == 0) return;

    for (size_t idim = 0; idim < host_attrs->angular.ndim; idim++) {
        if (host_attrs->angular.sep[idim] && host_attrs->angular.edges[idim]) {
            log_message(
                LOG_LEVEL_ERROR,
                "AngularWeight axis %zu has both sep and edges.\n",
                idim);
            exit(EXIT_FAILURE);
        }

        if (host_attrs->angular.sep[idim]) {
            FLOAT *device_sep = (FLOAT*) my_device_malloc(
                host_attrs->angular.shape[idim] * sizeof(FLOAT), buffer);

            CUDA_CHECK(cudaMemcpy(
                device_sep,
                host_attrs->angular.sep[idim],
                host_attrs->angular.shape[idim] * sizeof(FLOAT),
                cudaMemcpyHostToDevice));

            device_attrs->angular.sep[idim] = device_sep;
        }
        else if (host_attrs->angular.edges[idim]) {
            FLOAT *device_edges = (FLOAT*) my_device_malloc(
                (host_attrs->angular.shape[idim] + 1) * sizeof(FLOAT), buffer);

            CUDA_CHECK(cudaMemcpy(
                device_edges,
                host_attrs->angular.edges[idim],
                (host_attrs->angular.shape[idim] + 1) * sizeof(FLOAT),
                cudaMemcpyHostToDevice));

            device_attrs->angular.edges[idim] = device_edges;
        }
    }

    if (host_attrs->angular.size > 0) {
        FLOAT *device_weight = (FLOAT*) my_device_malloc(
            host_attrs->angular.size * sizeof(FLOAT), buffer);

        CUDA_CHECK(cudaMemcpy(
            device_weight,
            host_attrs->angular.weight,
            host_attrs->angular.size * sizeof(FLOAT),
            cudaMemcpyHostToDevice));

        device_attrs->angular.weight = device_weight;
    }
}


void free_device_weight_attrs(
    WeightAttrs *device_attrs,
    DeviceMemoryBuffer *buffer)
{
    // -----------------
    // AngularWeight
    // -----------------
    for (size_t idim = 0; idim < device_attrs->angular.ndim; idim++) {
        if (device_attrs->angular.sep[idim]) {
            my_device_free(device_attrs->angular.sep[idim], buffer);
            device_attrs->angular.sep[idim] = NULL;
        }
        if (device_attrs->angular.edges[idim]) {
            my_device_free(device_attrs->angular.edges[idim], buffer);
            device_attrs->angular.edges[idim] = NULL;
        }
    }

    if (device_attrs->angular.weight) {
        my_device_free(device_attrs->angular.weight, buffer);
        device_attrs->angular.weight = NULL;
    }

    // -----------------
    // BitwiseWeight
    // -----------------
    if (device_attrs->bitwise.p_correction_nbits) {
        my_device_free(device_attrs->bitwise.p_correction_nbits, buffer);
        device_attrs->bitwise.p_correction_nbits = NULL;
    }
}


size_t get_count2_size(IndexValue index_value1, IndexValue index_value2,
                        char names[][SIZE_NAME])
{
    // To check/modify when adding new weighting scheme
    int s1 = (index_value1.size_spin > 0);
    int s2 = (index_value2.size_spin > 0);
    size_t n = 1 + s1 + s2;

    if (names == NULL) {
        return n;
    }
    /* clear first buffers to be safe */
    size_t i;
    for (i = 0; i < MAX_NWEIGHT; ++i) {
        names[i][0] = '\0';
    }

    if (s1 && s2) {
        strncpy(names[0], "weight_plus_plus", SIZE_NAME-1);
        strncpy(names[1], "weight_plus_cross", SIZE_NAME-1);
        strncpy(names[2], "weight_cross_cross", SIZE_NAME-1);
    } else if (s1 ^ s2) {
        strncpy(names[0], "weight_plus", SIZE_NAME-1);
        strncpy(names[1], "weight_cross", SIZE_NAME-1);
    } else {
        strncpy(names[0], "weight", SIZE_NAME-1);
    }

    return n;
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream);


#endif