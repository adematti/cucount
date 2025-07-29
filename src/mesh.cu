#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "common.h"

// Function to calculate the Cartesian distance
__device__ FLOAT cartesian_distance(const FLOAT *position) {
    FLOAT rr = 0.0;
    for (size_t i = 0; i < NDIM; i++) rr += position[i] * position[i];
    return sqrt(rr); // Added sqrt to compute actual distance
}

// Function to convert Cartesian coordinates to spherical coordinates
__device__ void cartesian_to_sphere(const FLOAT *position, FLOAT *r, FLOAT *cth, FLOAT *phi) {
    *r = cartesian_distance(position);

    if (*r == 0) {
        *cth = 1.0;
        *phi = 0.0;
    } else {
        FLOAT x_norm = position[0] / *r;
        FLOAT y_norm = position[1] / *r;
        FLOAT z_norm = position[2] / *r;

        *cth = z_norm;
        if (x_norm == 0 && y_norm == 0) {
            *phi = 0.0;
        } else {
            *phi = atan2(y_norm, x_norm);
            if (*phi < 0) {
                *phi += 2 * M_PI;
            }
        }
    }
}


__device__ FLOAT wrap_angle(FLOAT phi) {
    // Wrap phi into the range [0, 2 * M_PI]
    phi = fmod(phi, 2 * M_PI); // Use modulo to handle wrapping
    if (phi < 0) {
        phi += 2 * M_PI; // Ensure phi is positive
    }
    return phi;
}


__device__ size_t angular_to_cell_index(const MeshAttrs mattrs, const FLOAT cth, const FLOAT phi) {
    // Returns the pixel index for coordinates cth (cos(theta)) and phi (azimuthal angle)

    // Compute pixel indices
    int icth = (cth == 1) ? (mattrs.meshsize[0] - 1) : (int)(0.5 * (1 + cth) * mattrs.meshsize[0]);
    int iphi = (int)(0.5 * wrap_angle(phi) / M_PI * mattrs.meshsize[1]);

    // Return combined pixel index
    return iphi + icth * mattrs.meshsize[1];
}

__device__ size_t cartesian_to_cell_index(const MeshAttrs mattrs, const FLOAT* position) {
    size_t index = 0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        index *= mattrs.meshsize[axis];
        FLOAT offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;
        index += (int) ((position[axis] - offset) * mattrs.meshsize[axis] / mattrs.boxsize[axis]);
    }
    return index;
}

__device__ void _find_extent(const MeshAttrs mattrs, const FLOAT *position, FLOAT *extent) {

    FLOAT cth, phi, r;
    if (mattrs.type == MESH_ANGULAR) {
        cartesian_to_sphere(position, &r, &cth, &phi);
        extent[0] = fmin(cth, extent[0]);
        extent[1] = fmax(cth, extent[1]);
        extent[2] = fmin(phi, extent[2]);
        extent[3] = fmax(phi, extent[3]);
    }
    else {
        for (size_t axis = 0; axis < NDIM; axis++) {
            extent[2 * axis] = fmin(position[axis], extent[2 * axis]);
            extent[2 * axis + 1] = fmax(position[axis], extent[2 * axis + 1]);
        }
    }
}


__global__ void find_extent_kernel(FLOAT *extent, Particles particles, const MeshAttrs mattrs) {

    extern __shared__ FLOAT sdata[];
    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    FLOAT textent[2 * NDIM];
    for (size_t axis = 0; axis < NDIM; axis++) {
        textent[2 * axis] = INFINITY;         // min = position
        textent[2 * axis + 1] = -INFINITY;     // max = position
    }

    for (size_t i = gid; i < particles.size; i += stride) {
        const FLOAT *position = &(particles.positions[NDIM * i]);
        _find_extent(mattrs, position, textent);
    }
    for (size_t axis = 0; axis < NDIM; axis++) {
        sdata[2 * NDIM * tid + 2 * axis] = textent[2 * axis];
        sdata[2 * NDIM * tid + 2 * axis + 1] = textent[2 * axis + 1];
    }
    __syncthreads();

    // reduction: safe for any blockDim.x
    // find largest power-of-two <= blockDim.x
    unsigned int max_s = 1;
    while (max_s * 2 <= blockDim.x) max_s *= 2;

    for (unsigned int s = max_s; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < blockDim.x) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                sdata[2 * NDIM * tid + 2 * axis] = fmin(sdata[2 * NDIM * tid + 2 * axis], sdata[2 * NDIM * (tid + s) + 2 * axis]);
                sdata[2 * NDIM * tid + 2 * axis + 1] = fmax(sdata[2 * NDIM * tid + 2 * axis + 1], sdata[2 * NDIM * (tid + s) + 2 * axis + 1]);
            }
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        for (size_t i = 0; i < 2 * NDIM; i++) extent[2 * NDIM * blockIdx.x + i] = sdata[i];
    }
}


void set_mesh_attrs(const Particles *list_particles, MeshAttrs *mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream) {

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(find_extent_kernel, nblocks, nthreads_per_block, buffer);

    size_t sum_nparticles = 0, n_nparticles = 0;
    FLOAT extent[2 * NDIM];
    for (size_t axis = 0; axis < NDIM; axis++) {
        extent[2 * axis] = INFINITY;
        extent[2 * axis + 1] = -INFINITY;
    }

    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.size == 0) continue;
        FLOAT *device_block_extent = (FLOAT*) my_device_malloc(nblocks * 2 * NDIM * sizeof(FLOAT), buffer);
        find_extent_kernel<<<nblocks, nthreads_per_block, nthreads_per_block * 2 * NDIM * sizeof(FLOAT), stream>>>(device_block_extent, particles, *mattrs);
        CUDA_CHECK(cudaDeviceSynchronize(stream));
        FLOAT *block_extent = (FLOAT*) my_malloc(nblocks * 2 * NDIM * sizeof(FLOAT));
        cudaMemcpyAsync(block_extent, device_block_extent, nblocks * 2 * NDIM * sizeof(FLOAT), cudaMemcpyDeviceToHost, stream);
        for (size_t i = 0; i < nblocks; i++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                extent[2 * axis] = MIN(block_extent[2 * NDIM * i + 2 * axis], extent[2 * axis]);
                extent[2 * axis + 1] = MAX(block_extent[2 * NDIM * i + 2 * axis + 1], extent[2 * axis + 1]);
            }
        }
        my_device_free(device_block_extent, buffer);
        free(block_extent);
        sum_nparticles += particles.size;
        n_nparticles += 1;
    }

    if (mattrs->type == MESH_ANGULAR) {
        FLOAT fsky = (extent[1] - extent[0]) * (extent[3] - extent[2]) / (4 * M_PI);
        log_message(LOG_LEVEL_INFO, "Enclosing fractional area is %.4f [%.4f %.4f] x [%.4f %.4f].\n", fsky, extent[0], extent[1], extent[2], extent[3]);

        if (mattrs->meshsize[0] * mattrs->meshsize[1] == 0) {
            FLOAT theta_max = acos(mattrs->smax);
            int nside1 = 5 * (int)(M_PI / theta_max);
            size_t nparticles = sum_nparticles / n_nparticles;
            int nside2 = MIN((int)(sqrt(0.25 * nparticles / fsky)), 2048);  // cap to avoid blowing up the memory
            mattrs->meshsize[0] = (size_t) MAX(MIN(nside1, nside2), 1);
            mattrs->meshsize[1] = 2 * mattrs->meshsize[0];
        }
        mattrs->boxsize[0] = extent[1] - extent[0];
        mattrs->boxsize[1] = extent[3] - extent[2];
        mattrs->boxcenter[0] = (extent[0] + extent[1]) / 2.;
        mattrs->boxcenter[1] = (extent[2] + extent[3]) / 2.;
        size_t meshsize = mattrs->meshsize[0] * mattrs->meshsize[1];
        FLOAT pixel_resolution = sqrt(4 * M_PI / meshsize) / DTORAD;
        log_message(LOG_LEVEL_INFO, "Mesh size is %d = %d x %d.\n", meshsize, mattrs->meshsize[0], mattrs->meshsize[1]);
        log_message(LOG_LEVEL_INFO, "Pixel resolution is %.4lf deg.\n", pixel_resolution);
        for (size_t axis = 2; axis < NDIM; axis++) mattrs->meshsize[axis] = 1;
    }

    if (mattrs->type == MESH_CARTESIAN) {
        FLOAT volume = 1.;
        for (size_t axis = 0; axis < NDIM; axis ++) {
            mattrs->boxsize[axis] = 1.001 * (extent[2 * axis + 1] - extent[2 * axis]);
            mattrs->boxcenter[axis] = (extent[2 * axis] + extent[2 * axis + 1]) / 2.;
            volume *= mattrs->boxsize[axis];
        }
        log_message(LOG_LEVEL_INFO, "Enclosing volume is %.4f [%.4f %.4f] x [%.4f %.4f] x [%.4f %.4f].\n", volume, extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]);

        size_t meshsize = 1;
        if (mattrs->meshsize[0] == 0) {
            int nside1 = (int) (16.0 * pow(volume, 1. / 3.) / mattrs->smax);
            int nside2 = (int) pow(0.5 * sum_nparticles / n_nparticles, 1. / 3.);
            for (size_t axis = 0; axis < NDIM; axis ++) {
                mattrs->meshsize[axis] = (size_t) MAX(MIN(nside1, nside2), 1);
                meshsize *= mattrs->meshsize[axis];
            }
        }
        FLOAT voxel_resolution = volume / meshsize;
        log_message(LOG_LEVEL_INFO, "Mesh size is %d = %d x %d x %d.\n", meshsize, mattrs->meshsize[0], mattrs->meshsize[1], mattrs->meshsize[2]);
        log_message(LOG_LEVEL_INFO, "Voxel resolution is %.4lf.\n", voxel_resolution);
    }

}


__device__ size_t _get_cell_index(const MeshAttrs mattrs, const FLOAT *position) {
    size_t index;
    if (mattrs.type == MESH_ANGULAR) {
        FLOAT cth, phi, r;
        cartesian_to_sphere(position, &r, &cth, &phi);
        index = angular_to_cell_index(mattrs, cth, phi);
    }
    else {
        index = cartesian_to_cell_index(mattrs, position);
    }
    return index;
}


__global__ void count_particles_kernel(const MeshAttrs mattrs, const Particles particles, size_t *index, Mesh mesh) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t i = gid; i < particles.size; i += stride) {
        const FLOAT *position = &(particles.positions[NDIM * i]);
        index[i] = _get_cell_index(mattrs, position);
        atomicAddSizet(&(mesh.nparticles[index[i]]), 1);
    }
}


// CUDA kernel to compute cumulative sum and reset nparticles
__global__ void cumulative_count_particles_kernel(size_t *nparticles, size_t *cumnparticles, size_t mesh_size) {
    // Shared memory for cumulative sum computation
    extern __shared__ size_t shared_data[];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load nparticles into shared memory
    if (tid < mesh_size) {
        shared_data[threadIdx.x] = nparticles[tid];
    } else {
        shared_data[threadIdx.x] = 0; // Handle out-of-bounds threads
    }
    __syncthreads();

    // Perform cumulative sum using parallel reduction
    for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
        size_t temp = 0;
        if (threadIdx.x >= stride) {
            temp = shared_data[threadIdx.x - stride];
        }
        __syncthreads();
        shared_data[threadIdx.x] += temp;
        __syncthreads();
    }

    // Write cumulative sum to cumnparticles
    if (tid < mesh_size) {
        cumnparticles[tid] = (threadIdx.x == 0) ? shared_data[threadIdx.x] : shared_data[threadIdx.x - 1];
        nparticles[tid] = 0; // Reset nparticles
    }
}


__global__ void fill_particles_kernel(const Particles particles, const size_t *index, Mesh mesh) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t i = gid; i < particles.size; i += stride) {
        size_t idx = index[i];
        const FLOAT* position = &(particles.positions[NDIM * i]);
        FLOAT sposition[NDIM];
        FLOAT r = cartesian_distance(position);

        for (size_t axis = 0; axis < NDIM; axis++) sposition[axis] = position[axis] / r;

        size_t local_index = atomicAddSizet(&(mesh.nparticles[idx]), 1);
        size_t offset = mesh.cumnparticles[idx] + local_index;

        for (size_t axis = 0; axis < NDIM; axis++) {
            mesh.positions[NDIM * offset + axis] = position[axis];
            mesh.spositions[NDIM * offset + axis] = sposition[axis];
        }
        mesh.weights[offset] = particles.weights[i];
    }
}



void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream) {

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count_particles_kernel, nblocks, nthreads_per_block, buffer);

    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.size == 0) continue;
        Mesh &mesh = list_mesh[imesh];
        mesh.size = 1;
        for (size_t axis = 0; axis < NDIM; axis++) mesh.size *= mattrs.meshsize[axis];
        // Allocate memory for mesh variables
        mesh.total_nparticles = particles.size;
        size_t *index = (size_t*) mmy_device_malloc(particles.size * sizeof(size_t), buffer);
        mesh.nparticles = (size_t*) my_device_malloc(mesh.size * sizeof(size_t), buffer);
        CUDA_CHECK(cudaMemset(mesh.nparticles, 0, mesh.size * sizeof(size_t)));
        mesh.cumnparticles = (size_t*) my_device_malloc(mesh.size * sizeof(size_t), buffer);
        CUDA_CHECK(cudaMemset(mesh.cumnparticles, 0, mesh.size * sizeof(size_t)));
        mesh.positions = (FLOAT*)  my_device_malloc(NDIM * particles.size * sizeof(FLOAT), buffer);
        mesh.spositions = (FLOAT*)  my_device_malloc(NDIM * particles.size * sizeof(FLOAT), buffer);
        mesh.weights = (FLOAT*)  my_device_malloc(particles.size * sizeof(FLOAT), buffer);

        // Assign particle positions to boxes
        count_particles_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(mattrs, particles, index, mesh);
        CUDA_CHECK(cudaDeviceSynchronize(stream));
        thrust::device_ptr<size_t> d_nparticles(mesh.nparticles);
        thrust::device_ptr<size_t> d_cumnparticles(mesh.cumnparticles);
        thrust::exclusive_scan(d_nparticles, d_nparticles + mesh.size, d_cumnparticles);
        cudaMemset(mesh.nparticles, 0, mesh.size * sizeof(size_t));  // reset
        CUDA_CHECK(cudaDeviceSynchronize(stream));
        fill_particles_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(particles, index, mesh);
        CUDA_CHECK(cudaDeviceSynchronize(stream));
        my_device_free(index, buffer);
    }
    log_message(LOG_LEVEL_INFO, "Mesh variables successfully set.\n");
}
