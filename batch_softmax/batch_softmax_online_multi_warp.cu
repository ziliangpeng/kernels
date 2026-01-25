#include "batch_softmax_online_multi_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>

// ============================================================================
// ONLINE MULTI-WARP BATCH SOFTMAX WITH VECTORIZED LOADS
// ============================================================================
//
// This kernel combines the online softmax algorithm with multi-warp parallelism
// and vectorized memory access for optimal performance on large dimensions.
//
// ONLINE SOFTMAX (Milakov & Gimelshein, 2018):
// --------------------------------------------
// Instead of separate max and sum passes, we compute both in a single pass:
//   For each element x:
//     old_max = max
//     max = fmaxf(max, x)
//     sum = sum * exp(old_max - max) + exp(x - max)
//
// This saves one full memory pass compared to the traditional approach.
//
// HYBRID REDUCTION FOR (MAX, SUM) PAIRS:
// --------------------------------------
// Phase 1: Each thread computes local (max, sum) via online updates
// Phase 2: Warp-level reduction using shuffles (merge (max,sum) pairs)
// Phase 3: Cross-warp reduction via shared memory
// Phase 4: Normalize output
//
// MERGING FORMULA:
// ----------------
// merged_max = max(max1, max2)
// merged_sum = sum1 * exp(max1 - merged_max) + sum2 * exp(max2 - merged_max)
//
// NaN PROTECTION:
// ---------------
// When max = -INFINITY, exp(-INFINITY - (-INFINITY)) = NaN
// Solution: Check for -INFINITY and use 0.0 for sum contribution
//
// ============================================================================

#define FULL_MASK 0xffffffff

// Helper: Merge two (max, sum) pairs using online softmax formula
__device__ __forceinline__ void mergeOnlinePair(
    float &max1, float &sum1,
    float max2, float sum2
) {
    float merged_max = fmaxf(max1, max2);

    // NaN protection: avoid exp(-INFINITY - (-INFINITY))
    float contrib1 = isinf(max1) ? 0.0f : sum1 * expf(max1 - merged_max);
    float contrib2 = isinf(max2) ? 0.0f : sum2 * expf(max2 - merged_max);

    max1 = merged_max;
    sum1 = contrib1 + contrib2;
}

// Helper: Warp-level reduction of (max, sum) pairs using shuffles
__device__ __forceinline__ void warpReduceOnlinePair(float &thread_max, float &thread_sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(FULL_MASK, thread_max, offset);
        float other_sum = __shfl_down_sync(FULL_MASK, thread_sum, offset);
        mergeOnlinePair(thread_max, thread_sum, other_max, other_sum);
    }
}

// Scalar kernel for non-vectorizable dimensions
template<int WARPS_PER_BLOCK>
__global__ void batch_softmax_online_multi_warp_scalar_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;

    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory for cross-warp reduction of (max, sum) pairs
    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // ========================================================================
    // PHASE 1: Online computation of (max, sum) - single pass over input
    // ========================================================================

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float x = row_input[i];

        // Online update: merge new element into running (max, sum)
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, x);

        // Update sum with correction factor for new max
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max))
                   + expf(x - thread_max);
    }

    // ========================================================================
    // PHASE 2: Warp-level reduction of (max, sum) pairs
    // ========================================================================

    warpReduceOnlinePair(thread_max, thread_sum);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // ========================================================================
    // PHASE 3: Cross-warp reduction (first warp reduces all warp results)
    // ========================================================================

    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    if (warp_id == 0) {
        // Load warp results (or identity values for inactive lanes)
        float val_max = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        float val_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;

        // Reduce across the first warp
        warpReduceOnlinePair(val_max, val_sum);

        if (lane_id == 0) {
            row_max_shared = val_max;
            row_sum_shared = val_sum;
        }
    }
    __syncthreads();

    float row_max = row_max_shared;
    float row_sum = row_sum_shared;

    // ========================================================================
    // PHASE 4: Normalize output
    // ========================================================================

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        row_output[i] = expf(row_input[i] - row_max) * inv_sum;
    }
}

// Vectorized kernel using float4 loads
template<int WARPS_PER_BLOCK>
__global__ void batch_softmax_online_multi_warp_vec4_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;

    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    const float4 *row_input4 = reinterpret_cast<const float4*>(row_input);
    float4 *row_output4 = reinterpret_cast<float4*>(row_output);
    int dim4 = dim / 4;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // ========================================================================
    // PHASE 1: Online computation with vectorized loads
    // ========================================================================

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];

        // Process 4 elements with online updates
        // Element x
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, vals.x);
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max))
                   + expf(vals.x - thread_max);

        // Element y
        old_max = thread_max;
        thread_max = fmaxf(thread_max, vals.y);
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max))
                   + expf(vals.y - thread_max);

        // Element z
        old_max = thread_max;
        thread_max = fmaxf(thread_max, vals.z);
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max))
                   + expf(vals.z - thread_max);

        // Element w
        old_max = thread_max;
        thread_max = fmaxf(thread_max, vals.w);
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max))
                   + expf(vals.w - thread_max);
    }

    // ========================================================================
    // PHASE 2: Warp-level reduction
    // ========================================================================

    warpReduceOnlinePair(thread_max, thread_sum);

    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // ========================================================================
    // PHASE 3: Cross-warp reduction
    // ========================================================================

    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    if (warp_id == 0) {
        float val_max = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        float val_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;

        warpReduceOnlinePair(val_max, val_sum);

        if (lane_id == 0) {
            row_max_shared = val_max;
            row_sum_shared = val_sum;
        }
    }
    __syncthreads();

    float row_max = row_max_shared;
    float row_sum = row_sum_shared;

    // ========================================================================
    // PHASE 4: Normalize output with vectorized stores
    // ========================================================================

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        float4 out;
        out.x = expf(vals.x - row_max) * inv_sum;
        out.y = expf(vals.y - row_max) * inv_sum;
        out.z = expf(vals.z - row_max) * inv_sum;
        out.w = expf(vals.w - row_max) * inv_sum;
        row_output4[i] = out;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

OnlineMultiWarpBatchSoftmax::OnlineMultiWarpBatchSoftmax(int batch_size, int dim, int num_warps)
    : batch_size(batch_size), dim(dim), num_warps(num_warps) {
    if (num_warps != 4 && num_warps != 8 && num_warps != 16) {
        fprintf(stderr, "Error: online_multi_warp only supports 4, 8, or 16 warps.\n");
        fprintf(stderr, "       Requested warps: %d\n", num_warps);
        throw std::invalid_argument("Unsupported num_warps for online_multi_warp");
    }

    use_vectorized = (dim % 4 == 0);
}

void OnlineMultiWarpBatchSoftmax::execute(const float *d_input, float *d_output) {
    if (use_vectorized) {
        if (num_warps == 4) {
            batch_softmax_online_multi_warp_vec4_kernel<4><<<batch_size, 4 * 32>>>(
                d_input, d_output, batch_size, dim);
        } else if (num_warps == 8) {
            batch_softmax_online_multi_warp_vec4_kernel<8><<<batch_size, 8 * 32>>>(
                d_input, d_output, batch_size, dim);
        } else if (num_warps == 16) {
            batch_softmax_online_multi_warp_vec4_kernel<16><<<batch_size, 16 * 32>>>(
                d_input, d_output, batch_size, dim);
        }
    } else {
        if (num_warps == 4) {
            batch_softmax_online_multi_warp_scalar_kernel<4><<<batch_size, 4 * 32>>>(
                d_input, d_output, batch_size, dim);
        } else if (num_warps == 8) {
            batch_softmax_online_multi_warp_scalar_kernel<8><<<batch_size, 8 * 32>>>(
                d_input, d_output, batch_size, dim);
        } else if (num_warps == 16) {
            batch_softmax_online_multi_warp_scalar_kernel<16><<<batch_size, 16 * 32>>>(
                d_input, d_output, batch_size, dim);
        }
    }
    cudaCheckError(cudaGetLastError());
}
