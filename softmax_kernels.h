#ifndef SOFTMAX_KERNELS_H
#define SOFTMAX_KERNELS_H

// Naive (unstable) - for demonstration of overflow issues
// Returns execution time in milliseconds
float softmax_Naive(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Multi-pass (3 kernels) - stable baseline approach
// Stage 1: Find max, Stage 2: Compute exp-sum, Stage 3: Normalize
// Returns execution time in milliseconds
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Fused 3-kernel - block-level fusion with 3 kernel launches
// Kernel 1: Block stats (max + exp-sum), Kernel 2: Global reduce, Kernel 3: Normalize
// Returns execution time in milliseconds
float softmax_Fused(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Fused 2-kernel - merge global reduce + normalize (SKELETON - not implemented)
// Kernel 1: Block stats, Kernel 2: Global reduce + normalize in single pass
// Returns execution time in milliseconds
float softmax_Fused2(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Fused 1-kernel - ultimate optimization (SKELETON - not implemented)
// Single kernel with block-level statistics and warp-level communication
// Returns execution time in milliseconds
float softmax_Fused1(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Online algorithm - single pass streaming computation (SKELETON - not implemented)
// Updates running max and sum statistics in single pass
// Returns execution time in milliseconds
float softmax_Online(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
