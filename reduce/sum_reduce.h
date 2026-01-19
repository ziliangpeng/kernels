#ifndef CUDA_REDUCE_SUM_REDUCE_H
#define CUDA_REDUCE_SUM_REDUCE_H

// ============================================================================
// SUM REDUCTION KERNELS
// ============================================================================

// Block-level sum reduction kernel using shared memory
// Each block reduces its elements to a single partial sum
__global__ void sumReductionKernel(const float *input, float *partialSums, int n);

// Warp-optimized sum reduction kernel
// Uses shared memory for 256→32, then warp shuffles for 32→1
__global__ void sumReductionKernel_Warp(const float *input, float *partialSums, int n);

// ============================================================================
// SUM REDUCTION WRAPPER FUNCTIONS
// ============================================================================

// Fully GPU recursive sum reduction (reduce until 1 element)
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock);

// Warp-optimized version: Fully GPU recursive reduction
float vectorSum_GPU_Warp(const float *d_input, int n, int threadsPerBlock);

// GPU with configurable CPU threshold
// Reduces on GPU until size <= threshold, then finishes on CPU
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

// Warp-optimized version: GPU with configurable CPU threshold
float vectorSum_Threshold_Warp(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

#endif  // CUDA_REDUCE_SUM_REDUCE_H
