#include "normalize.h"
#include <math.h>

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
