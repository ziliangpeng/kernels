#include "batch_softmax_online_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// ONLINE WARP-LEVEL BATCH SOFTMAX - SINGLE-PASS STATISTICS
// ============================================================================
//
// This kernel uses the online softmax algorithm from the "Online normalizer
// calculation for softmax" paper (Milakov & Gimelshein, 2018).
//
// TRADITIONAL TWO-PASS VS ONLINE SINGLE-PASS:
// -------------------------------------------
// Two-pass (naive, warp kernels):
//   Pass 1: Find max = max(x_i)
//   Pass 2: Compute sum = Σ exp(x_i - max)
//   Pass 3: Normalize output = exp(x - max) / sum
//   Total: 2 reads for statistics + 1 read/write for normalization
//
// Online single-pass:
//   Pass 1: Compute (max, sum) together in one pass
//           For each element x:
//             old_max = max
//             max = fmaxf(max, x)
//             sum = sum * exp(old_max - max) + exp(x - max)
//   Pass 2: Normalize output = exp(x - max) / sum
//   Total: 1 read for statistics + 1 read/write for normalization
//
// MERGING (MAX, SUM) PAIRS:
// -------------------------
// When combining thread states during warp reduction:
//   merged_max = max(max1, max2)
//   merged_sum = sum1 * exp(max1 - merged_max) + sum2 * exp(max2 - merged_max)
//
// This is mathematically equivalent:
//   exp(x - merged_max) = exp(x - max1) * exp(max1 - merged_max)
//                       = exp(x - max2) * exp(max2 - merged_max)
//
// NaN PROTECTION:
// ---------------
// When max = -INFINITY, exp(-INFINITY - (-INFINITY)) = exp(NaN) = NaN
// Solution: Check for -INFINITY and use 0.0 for sum contribution
//
// ============================================================================

#define FULL_MASK 0xffffffff

// Helper: Warp-level reduction of (max, sum) pairs using shuffles
__device__ __forceinline__ void warpReduceOnline(float &thread_max, float &thread_sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Get values from lane (tid + offset)
        float other_max = __shfl_down_sync(FULL_MASK, thread_max, offset);
        float other_sum = __shfl_down_sync(FULL_MASK, thread_sum, offset);

        // Merge (max, sum) pairs using online formula
        float merged_max = fmaxf(thread_max, other_max);

        // NaN protection: avoid -INFINITY - (-INFINITY)
        float merged_sum = (isinf(thread_max) ? 0.0f : thread_sum * expf(thread_max - merged_max)) +
                          (isinf(other_max) ? 0.0f : other_sum * expf(other_max - merged_max));

        thread_max = merged_max;
        thread_sum = merged_sum;
    }
    // After reduction, lane 0 has the warp's (max, sum)
}

__global__ void batch_softmax_online_warp_kernel(
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
    // PHASE 1: Online computation of (max, sum) in single pass
    // ========================================================================

    // Each thread maintains its own online state
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // Process elements with stride loop
    for (int i = tid; i < dim; i += 32) {
        float x = row_input[i];

        // Online update formula
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, x);

        // Update sum with correction factor for new max
        // sum_new = sum_old * exp(old_max - new_max) + exp(x - new_max)
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max)) +
                    expf(x - thread_max);
    }

    // ========================================================================
    // PHASE 2: Warp shuffle reduction to combine all thread (max, sum) pairs
    // ========================================================================

    warpReduceOnline(thread_max, thread_sum);

    // Broadcast global (max, sum) to all lanes
    float row_max = __shfl_sync(FULL_MASK, thread_max, 0);
    float row_sum = __shfl_sync(FULL_MASK, thread_sum, 0);

    // ========================================================================
    // PHASE 3: Normalize output
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim; i += 32) {
        row_output[i] = expf(row_input[i] - row_max) * inv_sum;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

OnlineWarpBatchSoftmax::OnlineWarpBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim) {
    // threadsPerBlock parameter is ignored - we always use 32 threads (1 warp)
    (void)threadsPerBlock;
}

void OnlineWarpBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Launch one block (single warp) per row
    batch_softmax_online_warp_kernel<<<batch_size, 32>>>(
        d_input, d_output, batch_size, dim
    );
    cudaCheckError(cudaGetLastError());
}
