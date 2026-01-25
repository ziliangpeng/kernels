#include "batch_softmax_multi_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// MULTI-WARP BATCH SOFTMAX WITH VECTORIZED LOADS
// ============================================================================
//
// This kernel uses multiple warps per block (typically 8 warps = 256 threads)
// with a hybrid reduction strategy and vectorized memory access.
//
// HYBRID REDUCTION STRATEGY:
// --------------------------
// Traditional tree reduction (naive kernel):
//   256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1  (8 stages, all use shared mem)
//
// Our hybrid approach:
//   256 → 8 (warp shuffles within each warp, no shared mem, no sync)
//   8 → 1 (first warp reduces 8 values via shared memory)
//
// Savings:
//   - Warp shuffles: ~1 cycle per op (vs 20-30 for shared memory)
//   - Only 2 __syncthreads() per phase (vs 8 in tree reduction)
//   - Only 8 shared memory writes (vs 256 in tree reduction)
//
// VECTORIZED LOADS (float4):
// --------------------------
// GPU memory bandwidth is optimized for 128-bit (16 byte) transactions.
// Using float4 (4 floats = 16 bytes) instead of float (4 bytes):
//   - 4x fewer memory transactions
//   - Better memory coalescing
//   - Reduced instruction count
//
// Pattern:
//   float4 vals = reinterpret_cast<const float4*>(row_input)[i];
//   thread_max = fmaxf(thread_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
//
// Requirement: dim must be divisible by 4 for vectorized path.
// Fallback: Scalar path when dim % 4 != 0.
//
// ============================================================================

#define FULL_MASK 0xffffffff
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)  // 256 threads

// Scalar kernel for non-vectorizable dimensions
__global__ void batch_softmax_multi_warp_scalar_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory for cross-warp reduction
    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // ========================================================================
    // PHASE 1: Find maximum using hybrid reduction
    // ========================================================================

    // Each thread finds local max over its portion
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    // Warp-level reduction: 32 → 1 per warp using shuffles
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces all warp results
    __shared__ float row_max_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
        }
        if (lane_id == 0) {
            row_max_shared = val;
        }
    }
    __syncthreads();
    float row_max = row_max_shared;

    // ========================================================================
    // PHASE 2: Compute exp(x - max), store in output, and sum
    // ========================================================================

    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = expf(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate result
        thread_sum += val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces all warp results
    __shared__ float row_sum_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if (lane_id == 0) {
            row_sum_shared = val;
        }
    }
    __syncthreads();
    float row_sum = row_sum_shared;

    // ========================================================================
    // PHASE 3: Normalize output (reuse stored exp values)
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        row_output[i] *= inv_sum;  // Normalize the stored exp values
    }
}

// Vectorized kernel using float4 loads
__global__ void batch_softmax_multi_warp_vec4_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    // Reinterpret as float4 pointers
    const float4 *row_input4 = reinterpret_cast<const float4*>(row_input);
    float4 *row_output4 = reinterpret_cast<float4*>(row_output);
    int dim4 = dim / 4;  // Number of float4 elements

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory for cross-warp reduction
    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // ========================================================================
    // PHASE 1: Find maximum using vectorized loads + hybrid reduction
    // ========================================================================

    float thread_max = -INFINITY;
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        // Find max of 4 values
        float local_max = fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w));
        thread_max = fmaxf(thread_max, local_max);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }

    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces all warp results
    __shared__ float row_max_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
        }
        if (lane_id == 0) {
            row_max_shared = val;
        }
    }
    __syncthreads();
    float row_max = row_max_shared;

    // ========================================================================
    // PHASE 2: Compute exp(x - max), store in output, and sum (vectorized)
    // ========================================================================

    float thread_sum = 0.0f;
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        float4 out;
        out.x = expf(vals.x - row_max);
        out.y = expf(vals.y - row_max);
        out.z = expf(vals.z - row_max);
        out.w = expf(vals.w - row_max);
        row_output4[i] = out;  // Store intermediate result
        thread_sum += out.x + out.y + out.z + out.w;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces all warp results
    __shared__ float row_sum_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if (lane_id == 0) {
            row_sum_shared = val;
        }
    }
    __syncthreads();
    float row_sum = row_sum_shared;

    // ========================================================================
    // PHASE 3: Normalize output (reuse stored exp values, vectorized)
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 out = row_output4[i];  // Load stored exp values
        out.x *= inv_sum;
        out.y *= inv_sum;
        out.z *= inv_sum;
        out.w *= inv_sum;
        row_output4[i] = out;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

MultiWarpBatchSoftmax::MultiWarpBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim), threadsPerBlock(BLOCK_SIZE) {
    // Ignore threadsPerBlock parameter - we always use 256 (8 warps)
    (void)threadsPerBlock;

    // Check if we can use vectorized loads
    use_vectorized = (dim % 4 == 0);
}

void MultiWarpBatchSoftmax::execute(const float *d_input, float *d_output) {
    if (use_vectorized) {
        batch_softmax_multi_warp_vec4_kernel<<<batch_size, BLOCK_SIZE>>>(
            d_input, d_output, batch_size, dim
        );
    } else {
        batch_softmax_multi_warp_scalar_kernel<<<batch_size, BLOCK_SIZE>>>(
            d_input, d_output, batch_size, dim
        );
    }
    cudaCheckError(cudaGetLastError());
}
