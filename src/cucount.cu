#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add(const float *a, const float *b, float *c, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    vector_add<<<grid_size, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();
}