#ifndef CUDA_REDUCE_MAX_REDUCE_H
#define CUDA_REDUCE_MAX_REDUCE_H

// ============================================================================
// MAX REDUCTION KERNELS
// ============================================================================

// Block-level max reduction kernel using shared memory
__global__ void maxReductionKernel(const float *input, float *partialMaxs, int n);

// Warp-optimized max reduction kernel
// Uses shared memory for 256→32, then warp shuffles for 32→1
__global__ void maxReductionKernel_Warp(const float *input, float *partialMaxs, int n);

// ============================================================================
// MAX REDUCTION WRAPPER FUNCTIONS
// ============================================================================

// Fully GPU recursive max reduction
float vectorMax_GPU(const float *d_input, int n, int threadsPerBlock);

// Warp-optimized version
float vectorMax_GPU_Warp(const float *d_input, int n, int threadsPerBlock);

#endif  // CUDA_REDUCE_MAX_REDUCE_H
