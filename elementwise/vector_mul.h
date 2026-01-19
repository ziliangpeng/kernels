#ifndef CUDA_ELEMENTWISE_VECTOR_MUL_H
#define CUDA_ELEMENTWISE_VECTOR_MUL_H

// CUDA kernel for vector multiplication: c[i] = a[i] * b[i]
__global__ void vectorMul(const float *a, const float *b, float *c, int n);

#endif  // CUDA_ELEMENTWISE_VECTOR_MUL_H
