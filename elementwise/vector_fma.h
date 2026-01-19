#ifndef CUDA_ELEMENTWISE_VECTOR_FMA_H
#define CUDA_ELEMENTWISE_VECTOR_FMA_H

// CUDA kernel for fused multiply-add: d[i] = a[i] * b[i] + c[i]
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n);

// Vectorized FMA kernel using float4: processes 4 elements per thread
// Note: n must be multiple of 4
__global__ void vectorFMA_float4(const float *a, const float *b, const float *c, float *d, int n);

#endif  // CUDA_ELEMENTWISE_VECTOR_FMA_H
