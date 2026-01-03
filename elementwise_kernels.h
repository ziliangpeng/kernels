#ifndef ELEMENTWISE_KERNELS_H
#define ELEMENTWISE_KERNELS_H

// ============================================================================
// BINARY ELEMENT-WISE OPERATIONS
// ============================================================================

// CUDA kernel for vector addition: c[i] = a[i] + b[i]
__global__ void vectorAdd(const float *a, const float *b, float *c, int n);

// CUDA kernel for vector multiplication: c[i] = a[i] * b[i]
__global__ void vectorMul(const float *a, const float *b, float *c, int n);

// ============================================================================
// TERNARY ELEMENT-WISE OPERATIONS
// ============================================================================

// CUDA kernel for fused multiply-add: d[i] = a[i] * b[i] + c[i]
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n);

// Vectorized FMA kernel using float4: processes 4 elements per thread
__global__ void vectorFMA_float4(const float *a, const float *b, const float *c, float *d, int n);

// ============================================================================
// UNARY ELEMENT-WISE OPERATIONS
// ============================================================================

// Naive normalization kernel: output[i] = exp(input[i]) / sum_exp
// Used by naive softmax (no max subtraction - numerically unstable)
__global__ void naiveNormalizeKernel(const float *input, float sum_exp, float *output, int n);

// Stable softmax normalization kernel: output[i] = exp(input[i] - max_val) / sum_exp
// Shared by multi-pass and fused softmax implementations
__global__ void softmaxNormalizeKernel(const float *input, float max_val, float sum_exp, float *output, int n);

// Device pointer version of naive normalization kernel (avoids host-device transfers)
__global__ void naiveNormalizeKernel_DevicePtr(const float *input, const float *d_sum_exp, float *output, int n);

// Device pointer version of softmax normalization kernel (avoids host-device transfers)
__global__ void softmaxNormalizeKernel_DevicePtr(const float *input, const float *d_max_val, const float *d_sum_exp, float *output, int n);

#endif
