#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"

__device__ static MeshAttrs device_mattrs;


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


__device__ size_t angular_to_cell(const FLOAT cth, const FLOAT phi) {
    // Returns the pixel index for coordinates cth (cos(theta)) and phi (azimuthal angle)

    // Compute pixel indices
    int icth = (cth == 1) ? (device_mattrs.meshsize[0] - 1) : (int)(0.5 * (1 + cth) * device_mattrs.meshsize[0]);
    int iphi = (int)(0.5 * wrap_angle(phi) / M_PI * device_mattrs.meshsize[1]);

    // Return combined pixel index
    return iphi + icth * device_mattrs.meshsize[1];
}

__device__ size_t cartesian_to_cell(const FLOAT* position) {
    size_t index = 0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        index *= device_mattrs.meshsize[axis];
        FLOAT offset = device_mattrs.boxcenter[axis] - device_mattrs.boxsize[axis] / 2;
        index += (int) ((position[axis] - offset) * device_mattrs.meshsize[axis] / device_mattrs.boxsize[axis]);
    }
    return index;
}


__global__ void find_extent_kernel(Particles particles, FLOAT *extent) {

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    FLOAT min[NDIM] = {0.}, max[NDIM] = {0.};
    for (size_t i = gid; i < particles.size; i += stride) {
        const FLOAT *position = &(particles.positions[NDIM * i]);
        if (device_mattrs.type == MESH_ANGULAR) {
            FLOAT cth, phi, r;
            cartesian_to_sphere(position, &r, &cth, &phi);
            if (i == gid) {
                min[0] = max[0] = cth;
                min[1] = max[0] = phi;
            }
            min[0] = MIN(cth, min[0]);
            max[0] = MAX(cth, max[0]);
            min[1] = MIN(phi, min[1]);
            max[1] = MAX(phi, max[1]);
        }
        else {
            for (size_t axis = 0; axis < NDIM; axis++) {
                if (i == gid) {
                    min[axis] = position[axis];
                    max[axis] = position[axis];
                }
                min[axis] = MIN(position[axis], min[axis]);
                max[axis] = MAX(position[axis], max[axis]);
            }
        }
    }
    for (size_t axis = 0; axis < NDIM; axis++) {
        sdata[2 * NDIM * tid + 2 * axis] = min[axis];
        sdata[2 * NDIM * tid + 2 * axis + 1] = max[axis];
    }

    __syncthreads();

    // Do reduction in shared mem
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                sdata[2 * NDIM * tid + 2 * axis] = MIN(sdata[2 * NDIM * tid + 2 * axis], sdata[2 * NDIM * (tid + s) + 2 * axis]);
                sdata[2 * NDIM * tid + 2 * axis + 1] = MAX(sdata[2 * NDIM * tid + 2 * axis + 1], sdata[2 * NDIM * (tid + s) + 2 * axis + 1]);
            }
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        for (size_t i = 0; i < 2 * NDIM; i++) extent[2 * NDIM * blockIdx.x + i] = sdata[i];
    }
}


void set_mesh_attrs(const Particles *list_particles, MeshAttrs *mattrs) {

    if ((nblocks <= 0) || (nthreads_per_block <= 0)) cudaOccupancyMaxPotentialBlockSize(&nblocks, &nthreads_per_block, find_extent_kernel, 0, 0);
    log_message(LOG_LEVEL_INFO, "Configured kernel with %d blocks and %d threads per block.\n", nblocks, nthreads_per_block);

    size_t sum_nparticles = 0, n_nparticles = 0;
    FLOAT extent[2 * NDIM];
    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));

    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.size == 0) continue;
        Particles device_particles;
        copy_particles_to_device(particles, &device_particles, 1);
        FLOAT *block_extent, *device_block_extent;
        cudaMalloc(&device_block_extent, nblocks * 2 * NDIM * sizeof(FLOAT));
        find_extent_kernel<<<nblocks, nthreads_per_block, nblocks * 2 * NDIM * sizeof(FLOAT)>>>(device_particles, device_block_extent);
        block_extent = (FLOAT*) my_malloc(nblocks * 2 * NDIM * sizeof(FLOAT));
        cudaMemcpy(block_extent, device_block_extent, nblocks * 2 * NDIM * sizeof(FLOAT), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nblocks; i++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                if ((imesh == 0) && (i == 0)) {
                    extent[2 * axis] = block_extent[2 * axis];
                    extent[2 * axis + 1] = block_extent[2 * axis + 1];
                }
                extent[2 * axis] = MIN(block_extent[2 * NDIM * i + 2 * axis], extent[2 * axis]);
                extent[2 * axis + 1] = MIN(block_extent[2 * NDIM * i + 2 * axis + 1], extent[2 * axis + 1]);
            }
        }
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
        for (size_t axis = 2; axis < NDIM; axis ++) mattrs->meshsize[axis] = 1;
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


__global__ void set_cell_index_kernel(const Particles particles, size_t *index) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t i = gid; i < particles.size; i += stride) {
        const FLOAT *position = &(particles.positions[NDIM * i]);
        if (device_mattrs.type == MESH_ANGULAR) {
            FLOAT cth, phi, r;
            cartesian_to_sphere(position, &r, &cth, &phi);
            index[i] = angular_to_cell(cth, phi);
        }
        else {
            index[i] = cartesian_to_cell(position);
        }
    }
}


__global__ void assign_particles_to_mesh_kernel(const Particles particles, const size_t *index, Mesh mesh) {

    size_t tid = threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    // Initialize box variables
    for (size_t i = gid; i < particles.size; i += stride) {
        size_t idx = index[i];
        atomicAddSizet(&(mesh.nparticles[idx]), 1);
    }
    __syncthreads();

    // Compute number of particles up to this cell
    __shared__ size_t total_nparticles;
    if (tid == 0) total_nparticles = 0;
    __syncthreads();

    for (size_t i = gid; i < mesh.size; i++) {
        mesh.cumnparticles[i] = total_nparticles;
        atomicAddSizet(&total_nparticles, mesh.nparticles[i]);
        mesh.nparticles[i] = 0; // Reset for reuse
    }
    __syncthreads();

    // Process particles
    for (size_t i = gid; i < particles.size; i += stride) {
        size_t idx = index[i];
        const FLOAT* position = &(particles.positions[NDIM * i]);
        FLOAT sposition[NDIM];
        FLOAT r = cartesian_distance(position);
        for (size_t axis = 0; axis < NDIM; axis++) sposition[axis] = position[axis] / r;
        size_t offset = NDIM * (mesh.cumnparticles[idx] + atomicAddSizet(&(mesh.nparticles[idx]), 1));
        for (size_t axis = 0; axis < NDIM; axis++) {
            mesh.positions[offset + axis] = position[axis];
            mesh.spositions[offset + axis] = sposition[axis];
        }
        offset = mesh.cumnparticles[idx] + mesh.nparticles[idx] - 1;
        mesh.weights[offset] = particles.weights[i];
    }
}



// Function to set the mesh
void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs) {

    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.size == 0) continue;
        Mesh &mesh = list_mesh[imesh];
        mesh.size = 1;
        for (size_t axis = 0; axis < NDIM; axis++) mesh.size *= mattrs.meshsize[axis];
        Particles device_particles;
        copy_particles_to_device(particles, &device_particles, 1);
        size_t *index;
        CUDA_CHECK(cudaMalloc((void **) &index, particles.size * sizeof(size_t)));
        set_cell_index_kernel<<<nblocks, nthreads_per_block>>>(device_particles, index);
        // Allocate memory for mesh variables
        CUDA_CHECK(cudaMalloc((void**) &(mesh.nparticles), mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMemset(mesh.nparticles, 0, mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc((void**) &(mesh.cumnparticles), mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMemset(mesh.cumnparticles, 0, mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc((void**) &(mesh.positions), NDIM * particles.size * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc((void**) &(mesh.spositions), NDIM * particles.size * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc((void**) &(mesh.weights), particles.size * sizeof(size_t)));
        mesh.total_nparticles = particles.size;
        Mesh device_mesh;
        copy_mesh_to_device(mesh, &device_mesh, 1);
        // Assign particle positions to boxes
        assign_particles_to_mesh_kernel<<<nblocks, nthreads_per_block>>>(device_particles, index, device_mesh);
        CUDA_CHECK(cudaFree(index));
    }
    log_message(LOG_LEVEL_INFO, "Mesh variables successfully set.\n");

}