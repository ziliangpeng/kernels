#include "elementwise_kernels.h"
#include <math.h>

// ============================================================================
// BINARY ELEMENT-WISE OPERATIONS
// ============================================================================

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

// CUDA kernel for vector multiplication
__global__ void vectorMul(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// ============================================================================
// TERNARY ELEMENT-WISE OPERATIONS
// ============================================================================

// CUDA kernel for fused multiply-add (FMA)
// Computes d[i] = a[i] * b[i] + c[i] in a single kernel
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        d[idx] = a[idx] * b[idx] + c[idx];
    }
}

// Vectorized FMA kernel using float4
// Each thread processes 4 elements at once using vectorized loads/stores
__global__ void vectorFMA_float4(const float *a, const float *b, const float *c, float *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread processes 4 elements
    // Note: n must be multiple of 4 (validated in host code)
    if (idx < n/4) {
        // Load 4 floats at once (single memory transaction)
        float4 va = ((float4*)a)[idx];
        float4 vb = ((float4*)b)[idx];
        float4 vc = ((float4*)c)[idx];

        // Compute 4 results
        float4 vd;
        vd.x = va.x * vb.x + vc.x;
        vd.y = va.y * vb.y + vc.y;
        vd.z = va.z * vb.z + vc.z;
        vd.w = va.w * vb.w + vc.w;

        // Store 4 floats at once (single memory transaction)
        ((float4*)d)[idx] = vd;
    }
}

// ============================================================================
// UNARY ELEMENT-WISE OPERATIONS
// ============================================================================

// Naive normalization kernel: output[i] = exp(input[i]) / sum_exp
// Used by naive softmax (no max subtraction - numerically unstable)
__global__ void naiveNormalizeKernel(const float *input, float sum_exp, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx]) / sum_exp;
    }
}

// Stable softmax normalization kernel: output[i] = exp(input[i] - max_val) / sum_exp
// Shared by multi-pass and fused softmax implementations
__global__ void softmaxNormalizeKernel(const float *input, float max_val, float sum_exp, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

// Device pointer version of naive normalization kernel with shared memory optimization
// Accepts device pointers for sum_exp to avoid host-device transfers
__global__ void naiveNormalizeKernel_DevicePtr(const float *input, const float *d_sum_exp, float *output, int n) {
    // Use shared memory to cache the global sum (read once per block instead of per thread)
    __shared__ float s_sum_exp;

    if (threadIdx.x == 0) {
        s_sum_exp = *d_sum_exp;
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx]) / s_sum_exp;
    }
}

// Device pointer version of softmax normalization kernel with shared memory optimization
// Accepts device pointers for max_val and sum_exp to avoid host-device transfers
__global__ void softmaxNormalizeKernel_DevicePtr(const float *input, const float *d_max_val, const float *d_sum_exp, float *output, int n) {
    // Use shared memory to cache global values (read once per block instead of per thread)
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;

    if (threadIdx.x == 0) {
        s_max_val = *d_max_val;
        s_sum_exp = *d_sum_exp;
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx] - s_max_val) / s_sum_exp;
    }
}
