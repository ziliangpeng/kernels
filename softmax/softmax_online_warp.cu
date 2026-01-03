#include "softmax_online_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// ============================================================================
// ONLINE SOFTMAX - WARP-LEVEL COOPERATIVE IMPLEMENTATION (PERFORMANCE)
// ============================================================================
//
// This implementation demonstrates advanced CUDA optimization techniques:
// 1. Warp shuffle primitives for fast intra-warp reductions
// 2. Cooperative groups for grid-wide synchronization
// 3. Single-kernel architecture eliminating launch overhead
//
// WARP SHUFFLE REDUCTION FOR (MAX, SUM) PAIRS:
// ---------------------------------------------
// The key challenge is reducing pairs of (max, sum) using warp shuffles.
// We need to shuffle both values and merge them correctly:
//
// For offset in [16, 8, 4, 2, 1]:  // 32 → 16 → 8 → 4 → 2 → 1
//   other_max = __shfl_down_sync(FULL_MASK, thread_max, offset)
//   other_sum = __shfl_down_sync(FULL_MASK, thread_sum, offset)
//
//   merged_max = max(thread_max, other_max)
//   merged_sum = thread_sum * exp(thread_max - merged_max) +
//                other_sum * exp(other_max - merged_max)
//
//   thread_max = merged_max
//   thread_sum = merged_sum
//
// After 5 rounds, lane 0 of each warp has the warp's (max, sum).
//
// COOPERATIVE LAUNCH ARCHITECTURE:
// ---------------------------------
// Single kernel with 3 phases separated by grid synchronization:
//
// Phase 1: Block-level statistics
//   - Each block computes (block_max, block_sum) using warp shuffles
//   - Stores results in d_block_maxes/d_block_sums
//
// Grid Sync 1: cooperative_groups::this_grid().sync()
//
// Phase 2: Global reduction (block 0 only)
//   - Block 0 reduces all block stats to (global_max, global_sum)
//   - Uses warp shuffles for efficiency
//
// Grid Sync 2: cooperative_groups::this_grid().sync()
//
// Phase 3: Normalize (all blocks)
//   - All blocks read global_max/global_sum and normalize their portion
//
// ============================================================================

#define FULL_MASK 0xffffffff

// Helper: Warp-level reduction of (max, sum) pairs using shuffles
// Each warp reduces 32 threads → 1 thread (lane 0)
__device__ void warpReduceMaxSum(float &thread_max, float &thread_sum) {
    // Reduce within warp using shuffles (no shared memory, no __syncthreads!)
    // Offsets: 16, 8, 4, 2, 1 (5 rounds to reduce 32 → 1)

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Get values from lane (tid + offset)
        float other_max = __shfl_down_sync(FULL_MASK, thread_max, offset);
        float other_sum = __shfl_down_sync(FULL_MASK, thread_sum, offset);

        // Merge (max, sum) pairs using online formula
        float merged_max = fmaxf(thread_max, other_max);

        // Avoid NaN from -INFINITY - (-INFINITY)
        float merged_sum = (isinf(thread_max) ? 0.0f : thread_sum * expf(thread_max - merged_max)) +
                          (isinf(other_max) ? 0.0f : other_sum * expf(other_max - merged_max));

        thread_max = merged_max;
        thread_sum = merged_sum;
    }
    // After this, lane 0 has the warp's (max, sum)
}

// Single cooperative kernel with 3 phases
__global__ void onlineWarp_CooperativeKernel(
    const float *input,
    float *block_maxes,
    float *block_sums,
    float *global_max,
    float *global_sum,
    float *output,
    int n,
    int numBlocks_stat
) {
    // Cooperative grid handle for synchronization
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];  // Size: warpsPerBlock * 2

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int warpsPerBlock = blockDim.x / 32;

    // ========================================================================
    // PHASE 1: Block-Level Online Statistics with Warp Optimization
    // ========================================================================

    if (bid < numBlocks_stat) {
        int idx = bid * blockDim.x + tid;
        int stride = numBlocks_stat * blockDim.x;

        // Step 1: Each thread maintains online (max, sum) state
        float thread_max = -INFINITY;
        float thread_sum = 0.0f;

        for (int i = idx; i < n; i += stride) {
            float x = input[i];
            float old_max = thread_max;
            thread_max = fmaxf(thread_max, x);

            // Online update with NaN protection
            thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max)) +
                        expf(x - thread_max);
        }

        // Step 2: Warp-level reduction using shuffles (32 → 1 per warp)
        warpReduceMaxSum(thread_max, thread_sum);

        // Step 3: Lane 0 of each warp writes to shared memory
        if (lane_id == 0) {
            sdata[warp_id] = thread_max;
            sdata[warp_id + warpsPerBlock] = thread_sum;
        }
        __syncthreads();

        // Step 4: First warp reduces all warp results to block result
        if (warp_id == 0) {
            // Load warp results (or -INFINITY/0 if no such warp)
            thread_max = (tid < warpsPerBlock) ? sdata[tid] : -INFINITY;
            thread_sum = (tid < warpsPerBlock) ? sdata[tid + warpsPerBlock] : 0.0f;

            // Reduce within first warp
            warpReduceMaxSum(thread_max, thread_sum);

            // Lane 0 writes block result
            if (lane_id == 0) {
                block_maxes[bid] = thread_max;
                block_sums[bid] = thread_sum;
            }
        }
    }

    // ========================================================================
    // GRID SYNC 1: Wait for all blocks to finish statistics
    // ========================================================================
    grid.sync();

    // ========================================================================
    // PHASE 2: Global Reduction (Block 0 Only)
    // ========================================================================

    if (bid == 0) {
        // Each thread accumulates multiple block (max, sum) pairs if needed
        float thread_max = -INFINITY;
        float thread_sum = 0.0f;

        // Grid-stride loop over all block statistics
        for (int i = tid; i < numBlocks_stat; i += blockDim.x) {
            float block_max = block_maxes[i];
            float block_sum = block_sums[i];

            // Merge with thread's running state
            float old_max = thread_max;
            thread_max = fmaxf(thread_max, block_max);

            // Online update with NaN protection
            thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max)) +
                        (isinf(block_max) ? 0.0f : block_sum * expf(block_max - thread_max));
        }

        // Warp-level reduction
        warpReduceMaxSum(thread_max, thread_sum);

        // Lane 0 of each warp writes to shared memory
        if (lane_id == 0) {
            sdata[warp_id] = thread_max;
            sdata[warp_id + warpsPerBlock] = thread_sum;
        }
        __syncthreads();

        // First warp reduces all warp results
        if (warp_id == 0) {
            thread_max = (tid < warpsPerBlock) ? sdata[tid] : -INFINITY;
            thread_sum = (tid < warpsPerBlock) ? sdata[tid + warpsPerBlock] : 0.0f;

            warpReduceMaxSum(thread_max, thread_sum);

            // Lane 0 writes global result
            if (lane_id == 0) {
                global_max[0] = thread_max;
                global_sum[0] = thread_sum;
            }
        }
    }

    // ========================================================================
    // GRID SYNC 2: Wait for global stats
    // ========================================================================
    grid.sync();

    // ========================================================================
    // PHASE 3: Normalize Output (All Blocks)
    // ========================================================================

    // All threads read global stats (broadcast from block 0)
    float g_max = global_max[0];
    float g_sum = global_sum[0];

    // Each block normalizes its portion of the output
    int norm_idx = bid * blockDim.x + tid;
    for (int i = norm_idx; i < n; i += gridDim.x * blockDim.x) {
        output[i] = expf(input[i] - g_max) / g_sum;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate intermediate buffers and check cooperative support
OnlineWarpSoftmax::OnlineWarpSoftmax(int n, int threadsPerBlock)
    : n(n), threadsPerBlock(threadsPerBlock) {

    // Check if device supports cooperative launch
    int device;
    cudaGetDevice(&device);

    int supportsCoop = 0;
    cudaDeviceGetAttribute(&supportsCoop, cudaDevAttrCooperativeLaunch, device);
    if (!supportsCoop) {
        fprintf(stderr, "ERROR: Device does not support cooperative launch!\n");
        exit(EXIT_FAILURE);
    }

    // Query max cooperative grid size using the proper CUDA API
    int maxCoopBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        onlineWarp_CooperativeKernel,
        threadsPerBlock,
        threadsPerBlock / 32 * 2 * sizeof(float)  // shared memory size
    );

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int numSMs = deviceProp.multiProcessorCount;
    maxCoopBlocks = numSMs * maxBlocksPerSM;

    // Calculate desired blocks (may exceed cooperative limit)
    int desiredBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Limit to max cooperative blocks
    numBlocks_stat = (desiredBlocks < maxCoopBlocks) ? desiredBlocks : maxCoopBlocks;

    // Allocate intermediate buffers
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks_stat * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks_stat * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));
}

// Execute: Pure kernel execution using cooperative launch
void OnlineWarpSoftmax::execute(const float *d_input, float *d_output) {
    // Shared memory size: 2x warpsPerBlock floats (for max and sum)
    int warpsPerBlock = threadsPerBlock / 32;
    size_t sharedMemSize = warpsPerBlock * 2 * sizeof(float);

    // Setup cooperative launch parameters
    void *kernelArgs[] = {
        (void*)&d_input,
        (void*)&d_block_maxes,
        (void*)&d_block_sums,
        (void*)&d_global_max,
        (void*)&d_global_sum,
        (void*)&d_output,
        (void*)&n,
        (void*)&numBlocks_stat
    };

    // Launch cooperative kernel
    cudaCheckError(cudaLaunchCooperativeKernel(
        (void*)onlineWarp_CooperativeKernel,
        numBlocks_stat,           // gridDim
        threadsPerBlock,          // blockDim
        kernelArgs,               // kernel arguments
        sharedMemSize,            // shared memory
        0                         // stream
    ));

    cudaCheckError(cudaGetLastError());
}

// Destructor: Free intermediate buffers
OnlineWarpSoftmax::~OnlineWarpSoftmax() {
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_OnlineWarp(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    OnlineWarpSoftmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
