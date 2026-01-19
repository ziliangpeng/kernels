#ifndef CUDA_ELEMENTWISE_NORMALIZE_H
#define CUDA_ELEMENTWISE_NORMALIZE_H

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

#endif  // CUDA_ELEMENTWISE_NORMALIZE_H
