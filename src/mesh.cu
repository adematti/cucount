#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
//#include <thrust/device_ptr.h>
//#include <thrust/scan.h>
#include "common.h"
#include "utils.h"


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


__device__ inline int wrap_periodic_int(int idx, int meshsize) {
    int r = idx % meshsize;
    return (r < 0) ? r + meshsize : r;
}


__device__ size_t cartesian_to_cell_index(const MeshAttrs mattrs, const FLOAT* position) {
    size_t index = 0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        index *= mattrs.meshsize[axis];
        FLOAT offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;
        int index_axis = (int) floor((position[axis] - offset) * mattrs.meshsize[axis] / mattrs.boxsize[axis]);
        index += wrap_periodic_int(index_axis, (int) mattrs.meshsize[axis]);
    }
    return index;
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



__device__ inline size_t my_atomicAddSizet(size_t* address, size_t val) {
    return atomicAdd((unsigned long long int*)address, (unsigned long long int)val);
}


__global__ void count_particles_kernel(const MeshAttrs mattrs, const Particles particles, size_t *index, Mesh mesh) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;
    IndexValue index_value = particles.index_value;

    for (size_t i = gid; i < particles.size; i += stride) {
        if ((index_value.size_individual_weight) && (particles.values[i * index_value.size + index_value.start_individual_weight] == 0.)) continue;
        const FLOAT *position = &(particles.positions[NDIM * i]);
        index[i] = _get_cell_index(mattrs, position);
        my_atomicAddSizet(&(mesh.nparticles[index[i]]), 1);
    }
}


__global__ void fill_particles_kernel(const Particles particles, const size_t *index, Mesh mesh) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;
    IndexValue index_value = particles.index_value;

    for (size_t i = gid; i < particles.size; i += stride) {
        if ((index_value.size_individual_weight) && (particles.values[i * index_value.size + index_value.start_individual_weight] == 0.)) continue;  // skip particles with zero weight; they count for nothing
        const FLOAT* position = &(particles.positions[NDIM * i]);
        const FLOAT *value = &(particles.values[index_value.size * i]);
        FLOAT sposition[NDIM];
        FLOAT r = cartesian_distance(position);
        for (size_t axis = 0; axis < NDIM; axis++) sposition[axis] = position[axis] / r;

        size_t idx = index[i];
        size_t local_index = my_atomicAddSizet(&(mesh.nparticles[idx]), 1);
        size_t offset = mesh.cumnparticles[idx] + local_index;

        for (size_t axis = 0; axis < NDIM; axis++) {
            mesh.positions[NDIM * offset + axis] = position[axis];
            mesh.spositions[NDIM * offset + axis] = sposition[axis];
        }
        for (size_t ivalue = 0; ivalue < index_value.size; ivalue++) {
            mesh.values[index_value.size * offset + ivalue] = value[ivalue];
        }
    }
}


void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream) {

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count_particles_kernel, nblocks, nthreads_per_block, buffer);

    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.size == 0) continue;
        Mesh &mesh = list_mesh[imesh];
        mesh.index_value = particles.index_value;
        mesh.size = 1;
        for (size_t axis = 0; axis < NDIM; axis++) mesh.size *= mattrs.meshsize[axis];
        // Allocate memory for mesh variables
        //printf("Allocating mesh %d\n", imesh);
        size_t *index = (size_t*) my_device_malloc(particles.size * sizeof(size_t), buffer);
        mesh.nparticles = (size_t*) my_device_malloc(mesh.size * sizeof(size_t), buffer);
        CUDA_CHECK(cudaMemset(mesh.nparticles, 0, mesh.size * sizeof(size_t)));
        mesh.cumnparticles = (size_t*) my_device_malloc(mesh.size * sizeof(size_t), buffer);
        CUDA_CHECK(cudaMemset(mesh.cumnparticles, 0, mesh.size * sizeof(size_t)));
        mesh.positions = (FLOAT*) my_device_malloc(NDIM * particles.size * sizeof(FLOAT), buffer);
        mesh.spositions = (FLOAT*) my_device_malloc(NDIM * particles.size * sizeof(FLOAT), buffer);
        mesh.values = (FLOAT*) my_device_malloc(particles.index_value.size * particles.size * sizeof(FLOAT), buffer);
        // Assign particle positions to boxes
        CUDA_CHECK(cudaDeviceSynchronize());
        count_particles_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(mattrs, particles, index, mesh);
        CUDA_CHECK(cudaDeviceSynchronize());
        /*
        thrust::device_ptr<size_t> d_nparticles = thrust::device_pointer_cast(mesh.nparticles);
        thrust::device_ptr<size_t> d_cumnparticles = thrust::device_pointer_cast(mesh.cumnparticles);
        thrust::exclusive_scan(d_nparticles, d_nparticles + mesh.size, d_cumnparticles);
        */
        exclusive_scan_size_t_device(mesh.nparticles, mesh.cumnparticles, mesh.size, buffer);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(&(mesh.total_nparticles), mesh.cumnparticles + (mesh.size - 1), sizeof(size_t), cudaMemcpyDeviceToHost);
        size_t last_bin_count;
        cudaMemcpy(&last_bin_count, mesh.nparticles + (mesh.size - 1), sizeof(size_t), cudaMemcpyDeviceToHost);
        mesh.total_nparticles += last_bin_count;
        //printf("Total nparticles %d %d %d %d\n", imesh, particles.size, mesh.total_nparticles, mesh.size);
        cudaMemset(mesh.nparticles, 0, mesh.size * sizeof(size_t));  // reset
        CUDA_CHECK(cudaDeviceSynchronize());
        fill_particles_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(particles, index, mesh);
        CUDA_CHECK(cudaDeviceSynchronize());
        my_device_free(index, buffer);
    }
    log_message(LOG_LEVEL_DEBUG, "Mesh variables successfully set.\n");
}
