#include "vector_add.h"

// CUDA kernel: runs on the GPU
// Each thread computes one element of the result
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
