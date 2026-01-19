#ifndef CUDA_REDUCE_SUM_REDUCE_ATOMIC_H
#define CUDA_REDUCE_SUM_REDUCE_ATOMIC_H

// Atomic-based reduction kernel
// Extremely simple but serializes - all threads contend for same result location
__global__ void sumReductionKernel_Atomic(const float *input, float *result, int n);

// Atomic method: Simple single-kernel approach using atomicAdd
// All threads directly add to global result - hardware serializes access
float vectorSum_Atomic(const float *d_input, int n, int threadsPerBlock);

#endif  // CUDA_REDUCE_SUM_REDUCE_ATOMIC_H
