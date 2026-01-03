#ifndef SOFTMAX_ONLINE_WARP_H
#define SOFTMAX_ONLINE_WARP_H

#include "softmax_kernel.h"

// Online Softmax - Warp-Level Cooperative Implementation (Performance)
//
// This implementation demonstrates advanced CUDA optimizations by using warp-level
// primitives and cooperative groups for maximum performance.
//
// Key Optimizations vs online_simple:
// -----------------------------------
// 1. Warp Shuffles: Uses __shfl_down_sync() for intra-warp reductions
//    - Eliminates shared memory bank conflicts within warps
//    - No __syncthreads() needed for warp-level operations
//    - ~8-15% faster than shared memory for small reductions
//
// 2. Cooperative Launch: Single kernel with grid synchronization
//    - Eliminates 2 kernel launch overheads (~5-10μs each)
//    - All phases (stats → reduce → normalize) in one kernel
//    - Requires cooperative_groups and special launch API
//
// Architecture: 1-Kernel Cooperative Launch with 3 Phases
// --------------------------------------------------------
// Phase 1: Block-level online statistics with warp optimization
//   - Each thread maintains local (max, sum) state
//   - Warp-level shuffle reduction (32 → 1 per warp)
//   - Cross-warp reduction in shared memory
//   - Outputs: d_block_maxes[numBlocks], d_block_sums[numBlocks]
//
// Grid Sync 1: Wait for all blocks to finish statistics
//
// Phase 2: Global reduction (single block reduces all block stats)
//   - Grid-stride loop with warp shuffles
//   - Final (global_max, global_sum) stored to device memory
//
// Grid Sync 2: Wait for global stats
//
// Phase 3: Normalize output (all blocks participate)
//   - Each block normalizes its portion of the array
//   - output[i] = exp(input[i] - global_max) / global_sum
//
// Warp Shuffle Pattern for (max, sum) Pairs:
// -------------------------------------------
// Reducing (max, sum) pairs using shuffles is tricky because we need to:
// 1. Shuffle both max and sum values
// 2. Merge using the online formula: merged_sum = sum_a * exp(max_a - merged_max) +
//                                                  sum_b * exp(max_b - merged_max)
// 3. Avoid NaN from -INFINITY - (-INFINITY)
//
// Example for 32 → 16 reduction:
//   other_max = __shfl_down_sync(0xffffffff, thread_max, 16);
//   other_sum = __shfl_down_sync(0xffffffff, thread_sum, 16);
//   merged_max = fmaxf(thread_max, other_max);
//   merged_sum = (isinf(thread_max) ? 0.0f : thread_sum * expf(thread_max - merged_max)) +
//                (isinf(other_max) ? 0.0f : other_sum * expf(other_max - merged_max));
//   thread_max = merged_max;
//   thread_sum = merged_sum;
//
// Cooperative Launch Requirements:
// ---------------------------------
// - CUDA 9.0+ (cooperative_groups)
// - Device must support cooperative launch (check with cudaDeviceGetAttribute)
// - Launch with cudaLaunchCooperativeKernel() API
// - Grid size limited by device's maxThreadsPerMultiProcessor
//
// Performance Characteristics:
// ----------------------------
// - Expected: ~0.035-0.040ms for 100K elements (8-15% faster than online_simple)
// - Fewer kernel launches (1 vs 3) saves ~10-20μs
// - Warp shuffles reduce shared memory pressure
// - May still be slightly slower than fused3 (~0.038ms) due to cooperative overhead
//
// Educational Value:
// ------------------
// - Demonstrates warp-level programming
// - Shows cooperative groups and grid synchronization
// - Illustrates performance vs complexity trade-offs
// - Good reference for advanced CUDA patterns
//
// Requirements:
// - CUDA 11.0+
// - Compute Capability 6.0+ (Pascal or newer)
// - C++11 or later

// Class-based interface for accurate profiling
class OnlineWarpSoftmax : public SoftmaxKernel {
private:
    float *d_block_maxes, *d_block_sums;
    float *d_global_max, *d_global_sum;
    int n, threadsPerBlock;
    int numBlocks_stat;   // Number of blocks for statistics phase
    int maxBlocksPerSM;   // Max blocks per SM (for cooperative launch)

public:
    // Constructor: Allocate intermediate buffers and check cooperative support
    OnlineWarpSoftmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution using cooperative launch
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free intermediate buffers
    ~OnlineWarpSoftmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_OnlineWarp(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
