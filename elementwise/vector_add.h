#ifndef CUDA_ELEMENTWISE_VECTOR_ADD_H
#define CUDA_ELEMENTWISE_VECTOR_ADD_H

// CUDA kernel for vector addition: c[i] = a[i] + b[i]
__global__ void vectorAdd(const float *a, const float *b, float *c, int n);

#endif  // CUDA_ELEMENTWISE_VECTOR_ADD_H
