// exclusive_scan_size_t.cu
// Compile: nvcc -O3 -arch=sm_70 exclusive_scan_size_t.cu -o scan

#include <cuda_runtime.h>
#include <stdio.h>

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

