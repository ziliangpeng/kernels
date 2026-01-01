#ifndef VECTOR_KERNELS_H
#define VECTOR_KERNELS_H

// CUDA kernel for vector addition: c[i] = a[i] + b[i]
__global__ void vectorAdd(const float *a, const float *b, float *c, int n);

// CUDA kernel for vector multiplication: c[i] = a[i] * b[i]
__global__ void vectorMul(const float *a, const float *b, float *c, int n);

// CUDA kernel for fused multiply-add: d[i] = a[i] * b[i] + c[i]
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n);

// Vectorized FMA kernel using float4: processes 4 elements per thread
__global__ void vectorFMA_float4(const float *a, const float *b, const float *c, float *d, int n);

#endif
