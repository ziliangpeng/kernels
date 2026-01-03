#include "softmax_multipass.h"
#include "cuda_utils.h"
#include "reduce_kernels.h"
#include "elementwise_kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// MULTI-PASS SOFTMAX IMPLEMENTATION (Numerically Stable)
// ============================================================================
//
// ALGORITHM OVERVIEW:
// ------------------
// Computes softmax as: output[i] = exp(x[i] - max) / sum(exp(x[j] - max))
//
// Three-stage process:
//   Stage 1: Find max(x) via recursive reduction
//   Stage 2: Compute sum(exp(x - max)) via recursive reduction
//   Stage 3: Normalize by dividing exp(x[i] - max) / sum
//
// IMPLEMENTATION DETAILS:
// -----------------------
// Stage 1 - Max Reduction (3-4 kernel launches depending on size):
//   - Uses warp-optimized maxReductionKernel_Warp (from reduce_kernels.h)
//   - Identity value: -INFINITY (neutral element for max operator)
//   - Recursive reduction: n → numBlocks → ... → 1
//   - Result: max_val (copied to host)
//
// Stage 2 - Sum Reduction (3-4 kernel launches depending on size):
//   - Launch 1: expSumReductionKernel_Stable
//       * Computes exp(x[i] - max_val) for each element
//       * Reduces within each block using warp-optimized tree reduction
//       * Outputs partial sums: n elements → numBlocks partial sums
//   - Launch 2+: sumReductionKernel_Warp (recursive, from reduce_kernels.h)
//       * Sums the partial results from previous stage
//       * Does NOT apply exp() again (values already exponentiated)
//       * Continues until only 1 final sum remains
//   - Result: sum_exp (copied to host)
//
// Stage 3 - Normalization (1 kernel launch):
//   - softmaxNormalizeKernel
//       * Computes output[i] = exp(x[i] - max_val) / sum_exp
//       * Each thread handles one element independently
//       * Note: Recomputes exp(x[i] - max_val) - inefficient but simple
//
// WARP OPTIMIZATIONS:
// -------------------
// All reduction kernels use warp shuffle primitives for final 32→1 reduction:
//   - maxReductionKernel_Warp: Uses fmaxf with warp shuffles
//   - expSumReductionKernel_Stable: Uses addition with warp shuffles
//   - sumReductionKernel_Warp: Uses addition with warp shuffles
//   - ~8% speedup per kernel, eliminates 5 __syncthreads barriers each
//
// WHY IT'S NUMERICALLY STABLE:
// -----------------------------
// Key insight: Subtract max before exponential
//   - exp(x - max) ≤ exp(0) = 1 for all x (since x ≤ max by definition)
//   - Guarantees no overflow: largest exp value is exactly 1.0
//   - Result always in valid range [0, 1]
//
// Example with input [88, 89, 90]:
//   max_val = 90
//   exp(88 - 90) = exp(-2) = 0.135   ✓
//   exp(89 - 90) = exp(-1) = 0.368   ✓
//   exp(90 - 90) = exp(0)  = 1.000   ✓
//   Sum = 1.503
//   Output = [0.090, 0.245, 0.665]   ✓ Valid probabilities!
//
// COMPARISON WITH NAIVE:
// ----------------------
// | Aspect              | Naive              | Multi-Pass              |
// |---------------------|--------------------|-------------------------|
// | Stages              | 2 stages           | 3 stages                |
// | Kernel Launches     | 4 (1M elements)    | 7-9 (1M elements)       |
// | Max Subtraction     | ❌ None             | ✅ Yes                   |
// | Numerical Stability | ❌ Unstable         | ✅ Stable                |
// | GPU→CPU Transfers   | 1 (sum only)       | 2 (max, then sum)       |
// | Performance (1M)    | 0.187ms            | 0.381ms (2x slower)     |
// | Correctness         | ❌ Fails (NaN/Inf)  | ✅ Always correct        |
//
// Key differences:
//   1. Extra reduction stage for max (3 kernel launches)
//   2. Different identity values: -INFINITY for max vs 0.0f for sum
//   3. More kernel launches → more overhead (~0.01-0.02ms each)
//   4. Extra GPU→CPU transfer for max value (~0.01ms)
//   5. Normalize uses exp(x - max) instead of exp(x)
//   6. Both recompute exp in normalization (inefficient but simple)
//
// KERNEL LAUNCH COUNT:
// --------------------
// For 1M elements with 256 threads/block:
//   Stage 1 (Max):
//     Launch 1: maxReductionKernel_Warp (1M → 3,907 blocks)
//     Launch 2: maxReductionKernel_Warp (3,907 → 16 blocks)
//     Launch 3: maxReductionKernel_Warp (16 → 1 block)
//   Stage 2 (Sum):
//     Launch 4: expSumReductionKernel_Stable (1M → 3,907 blocks)
//     Launch 5: sumReductionKernel_Warp (3,907 → 16 blocks)
//     Launch 6: sumReductionKernel_Warp (16 → 1 block)
//   Stage 3 (Normalize):
//     Launch 7: softmaxNormalizeKernel (1M elements)
//   Total: 7 kernel launches (vs 4 for naive)
//
// WHY IT'S SLOWER THAN NAIVE:
// ---------------------------
// Main performance killer: Extra reduction stage for max
//   - Essentially doubles the reduction work
//   - Naive: 4 kernel launches
//   - Multi-pass: 7 kernel launches
//   - Each launch costs ~0.01-0.02ms overhead
//   - Difference: 3 extra launches × 0.015ms ≈ 0.045ms overhead
//
// Additional costs:
//   - Extra GPU→CPU transfer for max: ~0.01ms
//   - More memory allocations during recursive reduction
//   - Total overhead: ~0.05-0.10ms
//
// Trade-off: ~2x slower but always produces correct results!
//
// PERFORMANCE:
// ------------
// - Slower: ~0.381ms for 1M elements (median)
// - 2x slower than naive (but naive is broken!)
// - Still 2x slower than fused (which is also stable)
// - Suitable for small-medium workloads where correctness matters
//
// For best performance with stability, use fused method (3 kernel launches).
//
// ============================================================================

// Note: maxReductionKernel_Warp and vectorMax_GPU_Warp are from reduce_kernels.h

// Kernel: Compute exp(x - max) and reduce to sum (warp-optimized)
__global__ void expSumReductionKernel_Stable(const float *input, float max_val, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input and compute exp(x - max) for numerical stability
    sdata[tid] = (idx < n) ? expf(input[idx] - max_val) : 0.0f;
    __syncthreads();

    // Part 1: Shared memory reduction (blockDim → 64)
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Explicit stride=32 reduction (64 → 32)
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    // Part 2: Warp-level reduction for final 32 → 1 (no __syncthreads needed!)
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// Note: softmaxNormalizeKernel is now provided by elementwise_kernels.h
// Note: vectorMax_GPU is now provided by reduce_kernels.h

// Helper: GPU reduction to compute sum(exp(x - max))
float vectorExpSum_GPU(const float *d_input, float max_val, int n, int threadsPerBlock) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;
    bool firstStage = true;

    // Keep reducing until we have 1 element
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch appropriate kernel
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        if (firstStage) {
            // First stage: compute exp(x - max) from original input
            expSumReductionKernel_Stable<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, max_val, d_output, currentSize);
            firstStage = false;
        } else {
            // Subsequent stages: just sum partial results (already exp'd, don't exp again!)
            // Use warp-optimized kernel for ~8% speedup
            sumReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        }
        cudaCheckError(cudaGetLastError());

        // Free previous temp buffer (if we allocated it)
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output;
        currentSize = numBlocks;
        allocated = true;
    }

    // Copy final single element to host
    float result;
    cudaCheckError(cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup final buffer
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    return result;
}

// Host function: Multi-pass stable softmax (warp-optimized)
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    // Stage 1: Find max(x) using warp-optimized kernel
    float max_val = vectorMax_GPU_Warp(d_input, n, threadsPerBlock);

    // Stage 2: Compute sum(exp(x - max))
    float sum_exp = vectorExpSum_GPU(d_input, max_val, n, threadsPerBlock);

    // Debug output (commented out after fixing overflow bug)
    // printf("[DEBUG] n=%d, max_val=%f, sum_exp=%f\n", n, max_val, sum_exp);

    // Stage 3: Normalize: output[i] = exp(x[i] - max) / sum
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    softmaxNormalizeKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, max_val, sum_exp, d_output, n);
    cudaCheckError(cudaGetLastError());
    cudaDeviceSynchronize();

    return 0.0f;  // Timing handled by caller
}
