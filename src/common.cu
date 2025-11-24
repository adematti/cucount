#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "common.h"


IndexValue get_index_value(int size_spin, int size_individual_weight, int size_bitwise_weight, int size_negative_weight) {
    // To check/modify when adding new weighting scheme
    IndexValue index_value = {0};  // sets everything to 0
    if (size_spin) {
        index_value.size_spin = size_spin;
        index_value.size += size_spin;
    }
    if (size_individual_weight) {
        index_value.start_individual_weight = index_value.size;
        index_value.size_individual_weight = size_individual_weight;
        index_value.size += size_individual_weight;
    }
    if (size_bitwise_weight) {
        index_value.start_bitwise_weight = index_value.size;
        index_value.size_bitwise_weight = size_bitwise_weight;
        index_value.size += size_bitwise_weight;
    }
    if (size_negative_weight) {
        index_value.start_negative_weight = index_value.size;
        index_value.size_negative_weight = size_negative_weight;
        index_value.size += size_negative_weight;
    }
    return index_value;
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
        // printf("nbytes, size = %zu, %zu, %zu %zu\n", nbytes / 8, buffer->offset / 8, (buffer->offset + nbytes) / 8, (buffer->size) / 8);
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
        *device_particles = particles;
    } else {
        CUDA_CHECK(cudaMalloc((void**) device_particles, sizeof(Particles)));
        CUDA_CHECK(cudaMemcpy(device_particles, &particles, sizeof(Particles), cudaMemcpyHostToDevice));
    }
    if (mode == 1) {
        device_particles->positions = particles.positions;
        device_particles->values = particles.values;
    }
    else {
        CUDA_CHECK(cudaMalloc((void **) &(device_particles->positions), NDIM * particles.size * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->positions, particles.positions, NDIM * particles.size * sizeof(FLOAT), cudaMemcpyHostToDevice));

        size_t nvalues = particles.index_value.size;
        CUDA_CHECK(cudaMalloc((void **) &(device_particles->values), particles.size * nvalues * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_particles->values, particles.values, particles.size * nvalues * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }
}


void copy_particles_to_host(Particles particles, Particles *host_particles, int mode) {
    // mode == 0: copy C struct and arrays to host
    // mode == 1: copy C struct only to host
    // mode == 2: copy arrays only to host
    if (mode == 2) {
        *host_particles = particles;
    } else {
        CUDA_CHECK(cudaMemcpy(host_particles, &particles, sizeof(Particles), cudaMemcpyDeviceToHost));
    }
    if (mode == 1) {
        host_particles->positions = particles.positions;
        host_particles->values = particles.values;
    }
    else {
        host_particles->positions = (FLOAT*) my_malloc(NDIM * particles.size * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_particles->positions, particles.positions, NDIM * particles.size * sizeof(FLOAT), cudaMemcpyDeviceToHost));

        size_t nvalues = particles.index_value.size;
        host_particles->values = (FLOAT*) my_malloc(particles.size * nvalues * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_particles->values, particles.values, particles.size * nvalues * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    }
}


void copy_mesh_to_device(Mesh mesh, Mesh *device_mesh, int mode) {
    if (mode == 2) {
        *device_mesh = mesh;
    } else {
        CUDA_CHECK(cudaMalloc((void**) device_mesh, sizeof(Mesh)));
        CUDA_CHECK(cudaMemcpy(device_mesh, &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));
    }
    if (mode == 1) {
        device_mesh->nparticles = mesh.nparticles;
        device_mesh->cumnparticles = mesh.cumnparticles;
        device_mesh->spositions = mesh.spositions;
        device_mesh->positions = mesh.positions;
        device_mesh->values = mesh.values;
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

        size_t nvalues = mesh.index_value.size;
        CUDA_CHECK(cudaMalloc((void **) &(device_mesh->values), mesh.total_nparticles * nvalues * sizeof(FLOAT)));
        CUDA_CHECK(cudaMemcpy(device_mesh->values, mesh.values, mesh.total_nparticles * nvalues * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }
}


void copy_mesh_to_host(Mesh mesh, Mesh *host_mesh, int mode) {
    if (mode == 2) {
        *host_mesh = mesh;
    } else {
        CUDA_CHECK(cudaMemcpy(host_mesh, &mesh, sizeof(Mesh), cudaMemcpyDeviceToHost));
    }
    if (mode == 1) {
        host_mesh->nparticles = mesh.nparticles;
        host_mesh->cumnparticles = mesh.cumnparticles;
        host_mesh->spositions = mesh.spositions;
        host_mesh->positions = mesh.positions;
        host_mesh->values = mesh.values;
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

        size_t nvalues = mesh.index_value.size;
        host_mesh->values = (FLOAT*) my_malloc(mesh.total_nparticles * nvalues * sizeof(FLOAT));
        CUDA_CHECK(cudaMemcpy(host_mesh->values, mesh.values, mesh.total_nparticles * nvalues * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    }
}


void free_device_particles(Particles *particles) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(particles->positions));
    CUDA_CHECK(cudaFree(particles->values));
}

void free_device_mesh(Mesh *mesh) {
    // Free GPU memory
    CUDA_CHECK(cudaFree(mesh->nparticles));
    CUDA_CHECK(cudaFree(mesh->cumnparticles));
    CUDA_CHECK(cudaFree(mesh->spositions));
    CUDA_CHECK(cudaFree(mesh->positions));
    CUDA_CHECK(cudaFree(mesh->values));
}


void free_host_particles(Particles *particles) {
    // Free host memory
    free(particles->positions);
    free(particles->values);
}

void free_host_mesh(Mesh *mesh) {
    // Free host memory
    free(mesh->nparticles);
    free(mesh->cumnparticles);
    free(mesh->spositions);
    free(mesh->positions);
    free(mesh->values);
}