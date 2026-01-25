#include "batch_softmax_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// WARP-LEVEL BATCH SOFTMAX KERNEL - ONE WARP PER ROW
// ============================================================================
//
// This kernel assigns one warp (32 threads) to each row of the input matrix.
// It uses warp shuffle intrinsics for reductions, which are faster than
// shared memory and require no synchronization.
//
// LAUNCH CONFIGURATION:
// - Grid: (batch_size, 1, 1) - blockIdx.x indexes the row
// - Block: (32, 1, 1) - exactly one warp per block
//
// KEY OPTIMIZATIONS:
// 1. Zero shared memory usage (no allocation overhead)
// 2. Zero __syncthreads() calls (warp shuffles are implicitly synchronous)
// 3. Warp shuffles are 2-3x faster than shared memory (1 cycle vs 20-30 cycles)
// 4. Minimal launch overhead with single warp per block
//
// PERFORMANCE CHARACTERISTICS:
// - Excellent for small dims (32-1024): minimal overhead, efficient reductions
// - Each thread processes dim/32 elements
// - For very large dims, consider the naive kernel with more threads
//
// ALGORITHM PER ROW:
// Phase 1: Each thread finds local max, warp shuffle reduction to global max
// Phase 2: Each thread computes local sum, warp shuffle reduction to global sum
// Phase 3: Each thread normalizes its portion of elements
//
// NUMERICAL STABILITY:
// Uses max-subtraction trick: softmax(x) = exp(x - max) / sum(exp(x - max))
// ============================================================================

#define FULL_MASK 0xffffffff

__global__ void batch_softmax_warp_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    // Each block (single warp) processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Pointers to this row's data
    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;  // 0-31 (single warp)

    // ========================================================================
    // PHASE 1: Find maximum value using warp shuffle reduction
    // ========================================================================

    // Each thread processes dim/32 elements via stride loop
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += 32) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    // Warp shuffle reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(FULL_MASK, thread_max, offset);
        thread_max = fmaxf(thread_max, other_max);
    }

    // Broadcast global max to all lanes
    float row_max = __shfl_sync(FULL_MASK, thread_max, 0);

    // ========================================================================
    // PHASE 2: Compute exp(x - max), store in output, and sum
    // ========================================================================

    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += 32) {
        float val = expf(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate result
        thread_sum += val;
    }

    // Warp shuffle reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }

    // Broadcast global sum to all lanes
    float row_sum = __shfl_sync(FULL_MASK, thread_sum, 0);

    // ========================================================================
    // PHASE 3: Normalize output (reuse stored exp values)
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim; i += 32) {
        row_output[i] *= inv_sum;  // Normalize the stored exp values
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

WarpBatchSoftmax::WarpBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim) {
    // threadsPerBlock parameter is ignored - we always use 32 threads (1 warp)
}

void WarpBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Launch one block (single warp) per row
    batch_softmax_warp_kernel<<<batch_size, 32>>>(
        d_input, d_output, batch_size, dim
    );
    cudaCheckError(cudaGetLastError());
}
