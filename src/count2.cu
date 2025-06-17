#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"

__device__ __constant__ MeshAttrs device_mattrs; // [0] = cth, [1] = phi
__device__ __constant__ SelectionAttrs device_sattrs; // [0] = min, [1] = max
__device__ __constant__ BinAttrs device_battrs;


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
    FLOAT smax = device_sattrs.max * DTORAD;

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


__device__ inline void addition(FLOAT *diff; const FLOAT *position1, const FLOAT *position2) {
    for (size_t i = 0; i < NDIM; i++) diff[i] = position1[i] + position2[i];
}

__device__ inline void difference(FLOAT *diff; const FLOAT *position1, const FLOAT *position2) {
    for (size_t i = 0; i < NDIM; i++) diff[i] = position1[i] - position2[i];
}

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2) {
    FLOAT d = 0.0;
    for (size_t i = 0; i < NDIM; i++) d += position1[i] * position2[i];
    return d;
}

__device__ inline bool is_selected(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    for (size_t i = 0; i < device_sattrs.ndim; i++) {
        VAR_TYPE var = device_sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= device_sattrs.smin[i]) && (costheta <= device_sattrs.smax[i]);
            //if (!selected) printf("costheta %f %.4f %.4f ", costheta, device_sattrs.smin, device_sattrs.smax);
        }
    }
    return selected;
}


__device__ inline FLOAT set_legendre(FLOAT legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2) {
    legendre_cache[0] = 1.0; // P_0(mu) = 1
    legendre_cache[1] = mu; // P_1(mu) = mu
    if ((ellmin % 2 == 0) && (ellstep % 2 == 0)) {
        for (int ell = ellmin; ell < ellmax; ell+=ellstep) {
            if (ell == 2) {
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
        for (int ell = 2; ell < ellmax; ell++) {
            legendre_cache[ell] = ((2.0 * ell - 1.0) * mu * legendre_cache[ell - 1] - (ell - 1.0) * legendre_cache[ell - 2]) / ell;
        }
    }
}


__device__ void add_weight(FLOAT *counts, FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2, FLOAT weight1, FLOAT weight2) {
    int ibin = 0;
    FLOAT diff[NDIM];
    difference(diff, position2, position1);
    const FLOAT s2 = dot(diff, diff);
    const FLOAT DEFAULT_VALUE = -1000.;
    FLOAT s = DEFAULT_VALUE;
    FLOAT mu = DEFAULT_VALUE;
    FLOAT mu2 = DEFAULT_VALUE;
    LOS_TYPE los = LOS_NONE;
    VAR_TYPE var = VAR_NONE;
    size_t ellmin, ellmax, ellstep;

    bool REQUIRED_S = 0, REQUIRED_MU = 0, REQUIRED_MU2 = 0, REQUIRED_POLE = 0;
    for (i = 0; i < device_battrs.ndim; i++) {
        var = device_battrs.var[i];
        if ((var == VAR_S) | (var = VAR_K)) {
            REQUIRED_S = 1;
        }
        if (var == VAR_MU) {
            los = device_battrs.los[i];
            REQUIRED_MU = 1;
        }
        if (var == VAR_POLE) {
            los = device_battrs.los[i];
            REQUIRED_POLE = 1;
            REQUIRED_MU2 = 1;
            ellmin = (size_t) device_battrs.min[i];
            ellmax = (size_t) device_battrs.max[i];
            ellstep = (size_t) device_battrs.step[i];
            if (!((ellmin % 2 == 0) && (ellstep % 2 == 0))) REQUIRED_MU = 1;
        }
    }
    REQUIRED_S |= REQUIRED_MU;

    if (REQUIRED_S) s = sqrt(s2);
    if (REQUIRED_MU2 || REQUIRED_MU) {
        FLOAT d;
        if (los == LOS_FIRSTPOINT) {
            d = dot(diff, sposition1);
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        else if (los == LOS_ENDPOINT) {
            mu = dot(diff, sposition2);
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        else {
            FLOAT vlos[NDIM];
            addition(vlos, sposition1, sposition2);
            d = dot(diff, vlos);
            if (REQUIRED_MU) mu = d / sqrt(dot(vlos, vlos)) / s;
            else: mu2 = d * d / dot(vlos, vlos) / s2;
        }
    }

    size_t i = 0, size = 1;
    for (i = 0; i < device_battrs.ndim; i++) {
        var = device_battrs.var[i];
        LOS_TYPE los = device_battrs.los[i];
        FLOAT value = 0;
        if (var == VAR_S) {
            value = s;
        }
        if (var == VAR_THETA) {
            value = acos(dot(sposition1, sposition2));
        }
        if (var == VAR_MU) {
            value = mu;
        }
        if ((var == VAR_S) || (var == VAR_THETA) || (var == VAR_MU)) {
            ibin *= device_battrs.shape[i];
            size *= device_battrs.shape[i];
            ibin += (int) (floor((value - device_battrs.min[i]) / device_battrs.step[i]));
        }
        else {
            break;
        }
    }
    if ((ibin >= 0) && (ibin < size)) {
        FLOAT weight = weight1 * weight2;
        if (i == device_battrs.ndim - 1) {
            atomicAdd(&(counts[ibin]), weight);
        }
        if (var == VAR_POLE) {
            FLOAT legendre_cache[MAX_POLE + 1];
            set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
            for (int ill; ill < device_battrs.shape[i]; ill++) {
                size_t ell = ill * ellstep + ellmin;
                atomicAdd(&(counts[ibin + ill]), weight * legendre_cache[ell]);
            }
        }
    }
}


__global__ void count2_kernel(FLOAT *counts, size_t nparticles1, FLOAT *mesh1_spositions, FLOAT *mesh1_positions, FLOAT *mesh1_weights,
                        size_t *mesh2_nparticles, size_t *mesh2_cumnparticles, FLOAT *mesh2_spositions, FLOAT *mesh2_positions, FLOAT *mesh2_weights) {
    // Shared memory for local histogram
    extern __shared__ FLOAT local_counts[];

    size_t tid = threadIdx.x;

    // Initialize local histogram
    for (size_t i = tid; i < device_battrs.size; i += blockDim.x) local_counts[i] = 0.;

    __syncthreads();

    // Global thread index
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    // Process particles
    for (size_t ii = gid; ii < nparticles1; ii += stride) {
        FLOAT *position1 = &(mesh1_positions[NDIM * ii]);
        FLOAT *sposition1 = &(mesh1_spositions[NDIM * ii]);
        FLOAT weight1 = mesh1_weights[ii];
        if (device_mattrs.type == MESH_ANGULAR) {
            int bounds[4];
            set_angular_bounds(sposition1, bounds);
            //printf("%d %d %d %d\n", bounds[0], bounds[1], bounds[2], bounds[3]);

            for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
                int icth_n = icth * device_mattrs.meshsize[1];
                for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
                    int iphi_true = (iphi + device_mattrs.meshsize[1]) % device_mattrs.meshsize[1];
                    int icell = iphi_true + icth_n;
                    int np2 = mesh2_nparticles[icell];
                    FLOAT *positions2 = &(mesh2_positions[NDIM * mesh2_cumnparticles[icell]]);
                    FLOAT *spositions2 = &(mesh2_spositions[NDIM * mesh2_cumnparticles[icell]]);
                    FLOAT *weights2 = &(mesh2_weights[mesh2_cumnparticles[icell]]);

                    for (size_t jj = 0; jj < np2; jj++) {
                        if (!is_selected(sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]))) {
                            continue;
                        }
                        add_weight(local_counts, sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]), weight1, weights2[jj]);
                    }
                }
            }
        }
    }
    __syncthreads();

    // Combine local histograms into global histogram
    for (size_t i = tid; i < device_battrs.size; i += blockDim.x) {
        atomicAdd(&counts[i], local_counts[i]);
    }
}

// Helper function for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


static void copy_mesh_to_device(Mesh mesh, size_t **mesh_nparticles, size_t **mesh_cumnparticles, FLOAT **mesh_spositions, FLOAT **mesh_positions, FLOAT **mesh_weights) {
    CUDA_CHECK(cudaMalloc((void **)mesh_nparticles, mesh.size * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(*mesh_nparticles, mesh.nparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)mesh_cumnparticles, mesh.size * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(*mesh_cumnparticles, mesh.cumnparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)mesh_spositions, NDIM * mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(*mesh_spositions, mesh.spositions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)mesh_positions, NDIM * mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(*mesh_positions, mesh.positions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)mesh_weights, mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(*mesh_weights, mesh.weights, mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));
}


static void free_mesh_from_device(size_t *mesh_nparticles, size_t *mesh_cumnparticles, FLOAT *mesh_spositions, FLOAT *mesh_positions, FLOAT *mesh_weights) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(mesh_nparticles));
    CUDA_CHECK(cudaFree(mesh_cumnparticles));
    CUDA_CHECK(cudaFree(mesh_spositions));
    CUDA_CHECK(cudaFree(mesh_positions));
    CUDA_CHECK(cudaFree(mesh_weights));
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, const BinAttrs battrs, const int nblocks, const int nthreads_per_block) {

    // Device pointers
    FLOAT *device_mesh1_positions, *device_mesh2_positions;
    FLOAT *device_mesh1_spositions, *device_mesh2_spositions;
    FLOAT *device_mesh1_weights, *device_mesh2_weights;
    size_t *device_mesh1_nparticles, *device_mesh2_nparticles;
    size_t *device_mesh1_cumnparticles, *device_mesh2_cumnparticles;
    FLOAT *device_counts;

    // CUDA timing events
    cudaEvent_t start, stop;
    float elapsed_time;

    // Initialize histograms
    for (int i = 0; i < battrs.size; i++) counts[i] = 0;

    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_sattrs, &sattrs, sizeof(SelectionAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_battrs, &battrs, sizeof(BinAttrs)));

    // Allocate GPU memory and copy data
    copy_mesh_to_device(list_mesh[0], &device_mesh1_nparticles, &device_mesh1_cumnparticles, &device_mesh1_spositions, &device_mesh1_positions, &device_mesh1_weights);
    copy_mesh_to_device(list_mesh[1], &device_mesh2_nparticles, &device_mesh2_cumnparticles, &device_mesh2_spositions, &device_mesh2_positions, &device_mesh2_weights);

    CUDA_CHECK(cudaMalloc((void **)&device_counts, battrs.size * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(device_counts, counts, battrs.size * sizeof(FLOAT), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int this_nblocks = nblocks;
    int this_nthreads_per_block = nthreads_per_block;
    if (nthreads_per_block <= 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &this_nblocks,
            &this_nthreads_per_block,
            count2_kernel,
            battrs.size * sizeof(FLOAT),
            0
        );
    }
    log_message(LOG_LEVEL_INFO, "Running counts with %d blocks and %d threads per block.\n", this_nblocks, this_nthreads_per_block);

    CUDA_CHECK(cudaEventRecord(start, 0));
    count2_kernel<<<this_nblocks, this_nthreads_per_block, battrs.size * sizeof(FLOAT)>>>(device_counts, list_mesh[0].total_nparticles, device_mesh1_spositions, device_mesh1_positions, device_mesh1_weights, device_mesh2_nparticles, device_mesh2_cumnparticles, device_mesh2_spositions, device_mesh2_positions, device_mesh2_weights);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    log_message(LOG_LEVEL_INFO, "Time elapsed: %3.1f ms\n", elapsed_time);

    // Copy histograms back to host
    CUDA_CHECK(cudaMemcpy(counts, device_counts, battrs.size * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    // Free GPU memory
    free_mesh_from_device(device_mesh1_nparticles, device_mesh1_cumnparticles, device_mesh1_spositions, device_mesh1_positions, device_mesh1_weights);
    free_mesh_from_device(device_mesh2_nparticles, device_mesh2_cumnparticles, device_mesh2_spositions, device_mesh2_positions, device_mesh2_weights);

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}