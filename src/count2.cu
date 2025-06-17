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
    } else {
        FLOAT dphi, calpha = cos(smax);
        cth_min = cos(th_hi + smax);
        cth_max = cos(th_lo - smax);

        if (theta < 0.5 * M_PI) {
            FLOAT c_thlo = cos(th_lo);
            dphi = acos(sqrt((calpha * calpha - c_thlo * c_thlo) / (1 - c_thlo * c_thlo)));
        } else {
            FLOAT c_thhi = cos(th_hi);
            dphi = acos(sqrt((calpha * calpha - c_thhi * c_thhi) / (1 - c_thhi * c_thhi)));
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


    bounds[0] = 0;
    bounds[1] = device_mattrs.meshsize[0] - 1;
    bounds[2] = 0;
    bounds[3] = device_mattrs.meshsize[1] - 1;

}


__device__ bool is_selected(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    if (device_sattrs.var == VAR_THETA) {
        FLOAT costheta = sposition1[0] * sposition2[0] + sposition1[1] * sposition2[1] + sposition1[2] * sposition2[2];
        selected = (costheta >= device_sattrs.smin) && (costheta <= device_sattrs.smax);
        //if (!selected) printf("costheta %f %.4f %.4f ", costheta, device_sattrs.smin, device_sattrs.smax);
    }
    return selected;
}


__device__ void add_weight(FLOAT *counts, FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2, FLOAT weight1, FLOAT weight2) {
    int ibin = 0;
    if (device_battrs.var == VAR_S) {
        FLOAT dist = 0, diff;
        for (size_t axis = 0; axis < NDIM; axis++) {
            diff = position2[axis] - position1[axis];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        ibin = (int) (floor((dist - device_battrs.min) / device_battrs.step));
    }
    if ((ibin >= 0) && (ibin < device_battrs.nbins)) {
        FLOAT weight = weight1 * weight2;
        atomicAdd(&(counts[ibin]), weight);
    }
}


__global__ void count2_kernel(FLOAT *counts, size_t nparticles1, FLOAT *mesh1_spositions, FLOAT *mesh1_positions, FLOAT *mesh1_weights,
                        size_t *mesh2_nparticles, size_t *mesh2_cumnparticles, FLOAT *mesh2_spositions, FLOAT *mesh2_positions, FLOAT *mesh2_weights) {
    // Shared memory for local histogram
    extern __shared__ FLOAT local_counts[];

    size_t tid = threadIdx.x;

    // Initialize local histogram
    for (size_t i = tid; i < device_battrs.nbins; i += blockDim.x) {
        local_counts[i] = 0.;
    }
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
    for (size_t i = tid; i < device_battrs.nbins; i += blockDim.x) {
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


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, const BinAttrs battrs, const PoleAttrs pattrs, const int nblocks, const int nthreads_per_block) {

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
    for (int i = 0; i < battrs.nbins; i++) counts[i] = 0;

    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_sattrs, &sattrs, sizeof(SelectionAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_battrs, &battrs, sizeof(BinAttrs)));

    // Allocate GPU memory and copy data
    copy_mesh_to_device(list_mesh[0], &device_mesh1_nparticles, &device_mesh1_cumnparticles, &device_mesh1_spositions, &device_mesh1_positions, &device_mesh1_weights);
    copy_mesh_to_device(list_mesh[1], &device_mesh2_nparticles, &device_mesh2_cumnparticles, &device_mesh2_spositions, &device_mesh2_positions, &device_mesh2_weights);

    CUDA_CHECK(cudaMalloc((void **)&device_counts, battrs.nbins * sizeof(FLOAT)));
    CUDA_CHECK(cudaMemcpy(device_counts, counts, battrs.nbins * sizeof(FLOAT), cudaMemcpyHostToDevice));

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
            battrs.nbins * sizeof(FLOAT),
            0
        );
    }
    log_message(LOG_LEVEL_INFO, "Running counts with %d blocks and %d threads per block.\n", this_nblocks, this_nthreads_per_block);

    CUDA_CHECK(cudaEventRecord(start, 0));
    count2_kernel<<<this_nblocks, this_nthreads_per_block, battrs.nbins * sizeof(FLOAT)>>>(device_counts, list_mesh[0].total_nparticles, device_mesh1_spositions, device_mesh1_positions, device_mesh1_weights, device_mesh2_nparticles, device_mesh2_cumnparticles, device_mesh2_spositions, device_mesh2_positions, device_mesh2_weights);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    log_message(LOG_LEVEL_INFO, "Time elapsed: %3.1f ms\n", elapsed_time);

    // Copy histograms back to host
    CUDA_CHECK(cudaMemcpy(counts, device_counts, battrs.nbins * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    // Free GPU memory
    free_mesh_from_device(device_mesh1_nparticles, device_mesh1_cumnparticles, device_mesh1_spositions, device_mesh1_positions, device_mesh1_weights);
    free_mesh_from_device(device_mesh2_nparticles, device_mesh2_cumnparticles, device_mesh2_spositions, device_mesh2_positions, device_mesh2_weights);

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}