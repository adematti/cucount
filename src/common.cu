#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"


IndexValue get_index_value(int size_split, int size_spin, int size_individual_weight, int size_bitwise_weight, int size_negative_weight) {
    // To check/modify when adding new weighting scheme
    IndexValue index_value = {0};  // sets everything to 0
    if (size_split) {
        index_value.start_split = index_value.size;
        index_value.size_split = size_split;
        index_value.size += size_split;
    }
    if (size_spin) {
        index_value.start_spin = index_value.size;
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


void copy_bin_attrs_to_device(BinAttrs *device_battrs, const BinAttrs *host_battrs, DeviceMemoryBuffer *buffer)
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

void free_device_bin_attrs(BinAttrs *device_battrs, DeviceMemoryBuffer *buffer)
{
    for (size_t idim = 0; idim < device_battrs->ndim; idim++) {
        if (device_battrs->array[idim]) {
            my_device_free(device_battrs->array[idim], buffer);
            device_battrs->array[idim] = NULL;
        }
    }
}


void copy_weight_attrs_to_device(WeightAttrs *device_attrs, const WeightAttrs *host_attrs, DeviceMemoryBuffer *buffer)
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


void free_device_weight_attrs(WeightAttrs *device_attrs, DeviceMemoryBuffer *buffer)
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


#define BLOCK_SIZE 512  // threads per block (scans 2*BLOCK_SIZE elements)


__global__ void scan_block_kernel(const size_t *in, size_t *out, size_t *block_sums, size_t n) {

    __shared__ size_t sdata[2 * BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t start = blockIdx.x * (2 * BLOCK_SIZE);

    size_t i1 = start + tid;
    size_t i2 = start + BLOCK_SIZE + tid;

    sdata[tid] = (i1 < n) ? in[i1] : 0;
    sdata[BLOCK_SIZE + tid] = (i2 < n) ? in[i2] : 0;

    __syncthreads();

    // upsweep
    for (size_t stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
        __syncthreads();
        size_t idx = (tid + 1) * (stride << 1) - 1;
        if (idx < 2 * BLOCK_SIZE) {
            sdata[idx] += sdata[idx - stride];
        }
    }

    // save block total & clear root for exclusive scan
    if (tid == 0) {
        if (block_sums) block_sums[blockIdx.x] = sdata[2 * BLOCK_SIZE - 1];
        sdata[2 * BLOCK_SIZE - 1] = 0;
    }

    // downsweep
    for (size_t stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        size_t idx = (tid + 1) * (stride << 1) - 1;
        if (idx < 2 * BLOCK_SIZE) {
            size_t t = sdata[idx - stride];
            sdata[idx - stride] = sdata[idx];
            sdata[idx] += t;
        }
    }
    __syncthreads();

    if (i1 < n) out[i1] = sdata[tid];
    if (i2 < n) out[i2] = sdata[BLOCK_SIZE + tid];
}

__global__ void add_block_sums_kernel(size_t *out, const size_t *block_sums, size_t n) {
    size_t tid = threadIdx.x;
    size_t start = blockIdx.x * (2 * BLOCK_SIZE);

    size_t offset = block_sums[blockIdx.x];

    size_t i1 = start + tid;
    size_t i2 = start + BLOCK_SIZE + tid;

    if (i1 < n) out[i1] += offset;
    if (i2 < n) out[i2] += offset;
}

void exclusive_scan_size_t_device(const size_t *d_in, size_t *d_out, size_t n, DeviceMemoryBuffer* buffer) {
    if (n == 0) return;

    const size_t ITEMS_PER_BLOCK = 2 * BLOCK_SIZE;
    size_t nblocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

    size_t *d_block_sums = NULL;
    size_t *d_block_sums_scanned = NULL;

    if (nblocks > 1) {
        d_block_sums = (size_t*) my_device_malloc(nblocks * sizeof(size_t), buffer);
    }

    // First pass: scan each block
    scan_block_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    if (nblocks > 1) {
        // Recursively scan block sums
        d_block_sums_scanned = (size_t*) my_device_malloc(nblocks * sizeof(size_t), buffer);
        exclusive_scan_size_t_device(d_block_sums, d_block_sums_scanned, nblocks, buffer);

        // Add block sums to outputs
        add_block_sums_kernel<<<nblocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);
        cudaDeviceSynchronize();

        my_device_free(d_block_sums, buffer);
        my_device_free(d_block_sums_scanned, buffer);
        if ((buffer) && (buffer->size > 0)) buffer->offset -= 2 * nblocks * sizeof(size_t);
    }
}