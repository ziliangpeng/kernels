#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H

// ============================================================================
// MAX REDUCTION
// ============================================================================

// CUDA kernels for max reduction (shared across modules)
__global__ void maxReductionKernel(const float *input, float *partialMaxs, int n);
__global__ void maxReductionKernel_Warp(const float *input, float *partialMaxs, int n);

// Fully GPU recursive max reduction
float vectorMax_GPU(const float *d_input, int n, int threadsPerBlock);
float vectorMax_GPU_Warp(const float *d_input, int n, int threadsPerBlock);

// ============================================================================
// SUM REDUCTION
// ============================================================================

// CUDA kernels for sum reduction (shared across modules)
__global__ void sumReductionKernel(const float *input, float *partialSums, int n);
__global__ void sumReductionKernel_Warp(const float *input, float *partialSums, int n);

// Fully GPU recursive sum reduction
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock);

// Option D: GPU with configurable CPU threshold
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

// Warp-optimized variants (use warp shuffles for final 32→1 reduction)
float vectorSum_GPU_Warp(const float *d_input, int n, int threadsPerBlock);
float vectorSum_Threshold_Warp(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

// Atomic method: Simple but serializes due to contention
float vectorSum_Atomic(const float *d_input, int n, int threadsPerBlock);

#endif
