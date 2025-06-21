#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"

__device__ __constant__ MeshAttrs device_mattrs; // [0] = cth, [1] = phi
__device__ __constant__ SelectionAttrs device_sattrs; // [0] = min, [1] = max
__device__ BinAttrs device_battrs;


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
        FLOAT offset = device_mattrs.boxcenter[axis] - device_mattrs.boxsize[axis] / 2;
        int index = (int) ((position[axis] - offset) * device_mattrs.meshsize[axis] / device_mattrs.boxsize[axis]);
        int delta = (int) (device_mattrs.smax / device_mattrs.boxsize[axis] * device_mattrs.meshsize[axis]) + 1;
        bounds[2 * axis] = MAX(index - delta, 0);
        bounds[2 * axis + 1] = MIN(index + delta, device_mattrs.meshsize[axis] - 1);
        //bounds[2 * axis] = 0;
        //bounds[2 * axis + 1] = device_mattrs.meshsize[axis] - 1;
    }
}

__device__ inline void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) add[axis] = position1[axis] + position2[axis];
}

__device__ inline void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) diff[axis] = position1[axis] - position2[axis];
}

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2) {
    FLOAT d = 0.;
    for (size_t axis = 0; axis < NDIM; axis++) d += position1[axis] * position2[axis];
    return d;
}

__device__ inline bool is_selected(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    for (size_t i = 0; i < device_sattrs.ndim; i++) {
        int var = device_sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= device_sattrs.smin[i]) && (costheta <= device_sattrs.smax[i]);
            //if (selected) printf("costheta %d %f %.4f %.4f ", i, costheta, device_sattrs.smin[i], device_sattrs.smax[i]);
        }
    }
    return selected;
}


__device__ inline void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2) {
    if ((ellmin % 2 == 0) && (ellstep % 2 == 0)) {
        for (int ell = ellmin; ell < ellmax; ell+=ellstep) {
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
    int los = LOS_NONE;
    int var = VAR_NONE;
    size_t ellmin, ellmax, ellstep;

    bool REQUIRED_S = 0, REQUIRED_MU = 0, REQUIRED_MU2 = 0;
    size_t i = 0;
    for (i = 0; i < device_battrs.ndim; i++) {
        var = device_battrs.var[i];
        if ((var == VAR_S) | (var == VAR_K)) {
            REQUIRED_S = 1;
        }
        if (var == VAR_MU) {
            los = device_battrs.los[i];
            REQUIRED_MU = 1;
        }
        if (var == VAR_POLE) {
            los = device_battrs.los[i];
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
            else mu2 = d * d / dot(vlos, vlos) / s2;
        }
        if (s == 0) {
            mu = 0.;
            mu2 = 0.;
        };
    }

    for (i = 0; i < device_battrs.ndim; i++) {
        var = device_battrs.var[i];
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
            int ibin_loc = (int) (floor((value - device_battrs.min[i]) / device_battrs.step[i]));
            if ((ibin_loc >= 0) && (ibin_loc < device_battrs.shape[i])) ibin = ibin * device_battrs.shape[i] + ibin_loc;
            else return;
        }
        else {
            break;
        }
    }
    FLOAT weight = weight1 * weight2;
    if (i == device_battrs.ndim) {
        atomicAdd(&(counts[ibin]), weight);
    }
    if (var == VAR_POLE) {
        FLOAT legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
        for (int ill = 0; ill < device_battrs.shape[i]; ill++) {
            size_t ell = ill * ellstep + ellmin;
            atomicAdd(&(counts[ibin * device_battrs.shape[i] + ill]), weight * (2 * ell + 1) * legendre_cache[ell]);
        }
    }
}


__global__ void count2_kernel(FLOAT *block_counts, Mesh mesh1, Mesh mesh2) {

    size_t tid = threadIdx.x;

    // Initialize local histogram
    FLOAT *local_counts = &block_counts[blockIdx.x * device_battrs.size];

    // Zero initialize histogram for this block
    for (int i = tid; i < device_battrs.size; i += blockDim.x) local_counts[i] = 0;

    __syncthreads();

    // Global thread index
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    //printf("%d %d  ", device_battrs.ndim, device_battrs.var[1]);

    // Process particles
    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);
        FLOAT weight1 = mesh1.weights[ii];
        int bounds[2 * NDIM];
        if (device_mattrs.type == MESH_ANGULAR) {
            set_angular_bounds(sposition1, bounds);
            //printf("%d %d %d %d\n", bounds[0], bounds[1], bounds[2], bounds[3]);

            for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
                int icth_n = icth * device_mattrs.meshsize[1];
                for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
                    int iphi_true = (iphi + device_mattrs.meshsize[1]) % device_mattrs.meshsize[1];
                    int icell = iphi_true + icth_n;
                    int np2 = mesh2.nparticles[icell];
                    FLOAT *positions2 = &(mesh2.positions[NDIM * mesh2.cumnparticles[icell]]);
                    FLOAT *spositions2 = &(mesh2.spositions[NDIM * mesh2.cumnparticles[icell]]);
                    FLOAT *weights2 = &(mesh2.weights[mesh2.cumnparticles[icell]]);

                    for (size_t jj = 0; jj < np2; jj++) {
                        if (!is_selected(sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]))) {
                            continue;
                        }
                        add_weight(local_counts, sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]), weight1, weights2[jj]);
                    }
                }
            }
        }
        if (device_mattrs.type == MESH_CARTESIAN) {
            set_cartesian_bounds(position1, bounds);
            //printf("%d %d %d %d\n", bounds[0], bounds[1], bounds[2], bounds[3]);
            for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
                int ix_n = ix * device_mattrs.meshsize[2] * device_mattrs.meshsize[1];
                for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
                    int iy_n = iy * device_mattrs.meshsize[2];
                    for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                        int icell = ix_n + iy_n + iz;
                        int np2 = mesh2.nparticles[icell];
                        FLOAT *positions2 = &(mesh2.positions[NDIM * mesh2.cumnparticles[icell]]);
                        FLOAT *spositions2 = &(mesh2.spositions[NDIM * mesh2.cumnparticles[icell]]);
                        FLOAT *weights2 = &(mesh2.weights[mesh2.cumnparticles[icell]]);
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


static void copy_mesh_to_device(Mesh &mesh, Mesh &device_mesh) {
    // Allocate memory for the struct itself on the device
    CUDA_CHECK(cudaMalloc((void**) &device_mesh, sizeof(Mesh)));
    // Copy the struct itself to the device
    CUDA_CHECK(cudaMemcpy(device_mesh, &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &(device_mesh.nparticles), mesh.size * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(device_mesh.nparticles, mesh.nparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &(device_mesh.cumnparticles), mesh.size * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(device_mesh.cumnparticles, mesh.cumnparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &(device_mesh.spositions), NDIM * mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(device_mesh.spositions, mesh.spositions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &(device_mesh.positions), NDIM * mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(device_mesh.positions, mesh.positions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &(device_mesh.weights), mesh.total_nparticles * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(device_mesh.weights, mesh.weights, mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));
}


static void free_mesh_from_device(Mesh &device_mesh) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(device_mesh.nparticles));
    CUDA_CHECK(cudaFree(device_mesh.cumnparticles));
    CUDA_CHECK(cudaFree(device_mesh.spositions));
    CUDA_CHECK(cudaFree(device_mesh.positions));
    CUDA_CHECK(cudaFree(device_mesh.weights));
    CUDA_CHECK(cudaFree(&device_mesh));
}


__global__ void reduce_kernel(const FLOAT *block_counts, int nblocks, FLOAT *final_counts, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    FLOAT sum = 0;
    for (int b = 0; b < nblocks; ++b) {
        sum += block_counts[b * size + i];
    }
    final_counts[i] = sum;
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, const BinAttrs battrs, const int nblocks, const int nthreads_per_block) {

    // Device pointers
    Mesh device_mesh1, device_mesh2; // Device struct pointer
    FLOAT *block_counts, *final_counts;

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
    copy_mesh_to_device(list_mesh[0], device_mesh1);
    copy_mesh_to_device(list_mesh[1], device_mesh2);

    int this_nblocks = nblocks;
    int this_nthreads_per_block = nthreads_per_block;
    if (nthreads_per_block <= 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &this_nblocks,
            &this_nthreads_per_block,
            count2_kernel,
            0,
            0
        );
    }
    log_message(LOG_LEVEL_INFO, "Running counts with %d blocks and %d threads per block.\n", this_nblocks, this_nthreads_per_block);

    // allocate histogram arrays
    CUDA_CHECK(cudaMalloc((void **)&block_counts, this_nblocks * battrs.size * sizeof(FLOAT)));
    CUDA_CHECK(cudaMalloc(&final_counts,  battrs.size * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemset(final_counts, 0, battrs.size * sizeof(FLOAT)));

    // Create CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    count2_kernel<<<this_nblocks, this_nthreads_per_block>>>(block_counts, device_mesh1, device_mesh2);
    CUDA_CHECK(cudaDeviceSynchronize());
    reduce_kernel<<<this_nblocks, this_nthreads_per_block>>>(block_counts, this_nblocks, final_counts, battrs.size);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    log_message(LOG_LEVEL_INFO, "Time elapsed: %3.1f ms.\n", elapsed_time);

    // Copy histograms back to host
    CUDA_CHECK(cudaMemcpy(counts, final_counts, battrs.size * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(block_counts));
    CUDA_CHECK(cudaFree(final_counts));

    free_mesh_from_device(device_mesh1);
    free_mesh_from_device(device_mesh2);

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}