#include <stdlib.h>
#include <stdio.h>
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

// C-compliant device malloc: returns pointer or NULL on error
void* my_device_malloc(size_t nbytes, DeviceMemoryBuffer* buffer) {
    if ((buffer) && (buffer->size > 0)) {
        // Use pre-allocated buffer if enough space
        // printf("nbytes, size = %zu, %zu, %zu %zu\n", nbytes, buffer->offset, buffer->offset + nbytes, buffer->size);
        if (buffer->offset + nbytes > buffer->size) {
            log_message(LOG_LEVEL_ERROR, "DeviceMemoryBuffer: not enough space for allocation (%zu requested, %zu available)\n", nbytes, buffer->size - buffer->offset);
            exit(EXIT_FAILURE);
        }
        char* p = (char*)buffer->ptr + buffer->offset;
        buffer->offset += nbytes;
        return (void*)p;
    } else {
        // Fallback to cudaMalloc
        void* ptr = NULL;
        CUDA_CHECK(cudaMalloc((void **)&ptr, nbytes));
        return ptr;
    }
}

// Example: free fallback allocations (not buffer allocations)
void my_device_free(void* ptr, DeviceMemoryBuffer* buffer) {
    if ((buffer) && (buffer->size > 0)) {
        // Only free if not part of the buffer
        // Do nothing
    }
    else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void copy_particles_to_device(Particles particles, Particles *device_particles, int mode) {
    // mode == 0: copy C struct and arrays to device
    // mode == 1: copy C struct only to device
    // mode == 2: copy arrays only to device
    if (mode == 2) {
        device_particles->size = particles.size;
    } else {
        CUDA_CHECK(cudaMalloc((void**) device_particles, sizeof(Particles)));
        CUDA_CHECK(cudaMemcpy(device_particles, &particles, sizeof(Particles), cudaMemcpyHostToDevice));
    }
    if (mode == 1) {
        device_particles->positions = particles.positions;
        device_particles->weights = particles.weights;
    }
    else {
        CUDA_CHECK(cudaMalloc((void **) &(device_particles->positions), NDIM * particles.size * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->positions, particles.positions, NDIM * particles.size * sizeof(FLOAT), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &(device_particles->weights), particles.size * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->weights, particles.weights, particles.size * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }
}


void copy_particles_to_host(Particles particles, Particles *host_particles, int mode) {
    // mode == 0: copy C struct and arrays to host
    // mode == 1: copy C struct only to host
    // mode == 2: copy arrays only to host
    if (mode == 2) {
        host_particles->size = particles.size;
    } else {
        CUDA_CHECK(cudaMemcpy(host_particles, &particles, sizeof(Particles), cudaMemcpyDeviceToHost));
    }
    if (mode == 1) {
        host_particles->positions = particles.positions;
        host_particles->weights = particles.weights;
    }
    else {
        host_particles->positions = (FLOAT*) my_malloc(NDIM * particles.size * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_particles->positions, particles.positions, NDIM * particles.size * sizeof(FLOAT), cudaMemcpyDeviceToHost));

        host_particles->weights = (FLOAT*) my_malloc(particles.size * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_particles->weights, particles.weights, particles.size * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    }
}


void copy_mesh_to_device(Mesh mesh, Mesh *device_mesh, int mode) {
    if (mode == 2) {
        device_mesh->size = mesh.size;
        device_mesh->total_nparticles = mesh.total_nparticles;
    } else {
        CUDA_CHECK(cudaMalloc((void**) device_mesh, sizeof(Mesh)));
        CUDA_CHECK(cudaMemcpy(device_mesh, &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));
    }
    if (mode == 1) {
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


void copy_mesh_to_host(Mesh mesh, Mesh *host_mesh, int mode) {
    if (mode == 2) {
        host_mesh->size = mesh.size;
        host_mesh->total_nparticles = mesh.total_nparticles;
    } else {
        CUDA_CHECK(cudaMemcpy(host_mesh, &mesh, sizeof(Mesh), cudaMemcpyDeviceToHost));
    }
    if (mode == 1) {
        host_mesh->nparticles = mesh.nparticles;
        host_mesh->cumnparticles = mesh.cumnparticles;
        host_mesh->spositions = mesh.spositions;
        host_mesh->positions = mesh.positions;
        host_mesh->weights = mesh.weights;
    }
    else {
        host_mesh->nparticles = (size_t*) my_malloc(mesh.size * sizeof(size_t));
        CUDA_CHECK(cudaMemcpy(host_mesh->nparticles, mesh.nparticles, mesh.size * sizeof(size_t), cudaMemcpyDeviceToHost));

        host_mesh->cumnparticles = (size_t*) my_malloc(mesh.size * sizeof(size_t));
        CUDA_CHECK(cudaMemcpy(host_mesh->cumnparticles, mesh.cumnparticles, mesh.size * sizeof(size_t), cudaMemcpyDeviceToHost));

        host_mesh->spositions = (FLOAT*) my_malloc(NDIM * mesh.total_nparticles * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_mesh->spositions, mesh.spositions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyDeviceToHost));

        host_mesh->positions = (FLOAT*) my_malloc(NDIM * mesh.total_nparticles * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_mesh->positions, mesh.positions, NDIM * mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyDeviceToHost));

        host_mesh->weights = (FLOAT*) my_malloc(mesh.total_nparticles * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_mesh->weights, mesh.weights, mesh.total_nparticles * sizeof(FLOAT), cudaMemcpyDeviceToHost));
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


void free_host_particles(Particles *particles) {
    // Free GPU memory
    free(particles->positions);
    free(particles->weights);
}

void free_host_mesh(Mesh *mesh) {
    // Free GPU memory
    free(mesh->nparticles);
    free(mesh->cumnparticles);
    free(mesh->spositions);
    free(mesh->positions);
    free(mesh->weights);
}