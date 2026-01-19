#include "vector_mul.h"

// CUDA kernel for vector multiplication
__global__ void vectorMul(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
