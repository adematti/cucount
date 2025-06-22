#include <stdlib.h>
#include <cuda.h>
#include "common.h"


// Wrapper for calloc with error handling
void* my_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr) {
        log_message(LOG_LEVEL_ERROR, "Memory allocation failed in my_calloc for %zu elements of size %zu.\n", num, size);
        exit(EXIT_FAILURE); // Exit the program on allocation failure
    }
    return ptr;
}

// Wrapper for malloc with error handling
void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        log_message(LOG_LEVEL_ERROR, "Memory allocation failed in my_malloc for size %zu.\n", size);
        exit(EXIT_FAILURE); // Exit the program on allocation failure
    }
    return ptr;
}

void copy_particles_to_device(Particles particles, Particles *device_particles, bool struct_only) {
    CUDA_CHECK(cudaMalloc((void**) device_particles, sizeof(Particles)));
    CUDA_CHECK(cudaMemcpy(device_particles, &particles, sizeof(Particles), cudaMemcpyHostToDevice));
    if (struct_only) {
        device_particles->positions = particles.positions;
        device_particles->weights = particles.weights;
    }
    else {
        CUDA_CHECK(cudaMalloc((void **) &(device_particles->positions), NDIM * particles.size * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->positions, particles.positions, NDIM * particles.size * sizeof(FLOAT), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_particles->weights), mesh.total_nparticles * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->weights, particles.weights, particles.size * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }
}


void copy_mesh_to_device(Mesh mesh, Mesh *device_mesh, bool struct_only) {
    CUDA_CHECK(cudaMalloc((void**) device_mesh, sizeof(Mesh)));
    CUDA_CHECK(cudaMemcpy(device_mesh, &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));
    if (struct_only) {
        device_mesh->nparticles = mesh.nparticles;
        device_mesh->cumnparticles = mesh.cumnparticles;
        device_mesh->spositions = mesh.spositions;
        device_mesh->positions = mesh.positions;
        device_mesh->weights = mesh.weights;
    }
    else {
        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->nparticles), mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(device_mesh->nparticles, mesh.nparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->cumnparticles), mesh.size * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(device_mesh->cumnparticles, mesh.cumnparticles, mesh.size * sizeof(size_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->spositions), NDIM * mesh.total_nparticles * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_mesh->spositions, mesh.spositions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->positions), NDIM * mesh.total_nparticles * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_mesh->positions, mesh.positions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->weights), mesh.total_nparticles * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_mesh->weights, mesh.weights, mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }
}


void free_device_particles(Particles *particles) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(particles->positions));
    CUDA_CHECK(cudaFree(particles->weights));
}

void free_device_mesh(Mesh *mesh) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(mesh->nparticles));
    CUDA_CHECK(cudaFree(mesh->cumnparticles));
    CUDA_CHECK(cudaFree(mesh->spositions));
    CUDA_CHECK(cudaFree(mesh->positions));
    CUDA_CHECK(cudaFree(mesh->weights));
}


// Global variables for block and thread configuration
int nblocks = 0;
int nthreads_per_block = 0;

// Function to determine block and thread configuration
void configure_cuda_kernel(void (*kernel)(void)) {
    if (nthreads_per_block <= 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &nblocks,
            &nthreads_per_block,
            kernel,
            0,
            0
        );
    }
    log_message(LOG_LEVEL_INFO, "Configured kernel with %d blocks and %d threads per block.\n", nblocks, nthreads_per_block);
}