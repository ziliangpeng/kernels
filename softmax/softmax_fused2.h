#ifndef SOFTMAX_FUSED2_H
#define SOFTMAX_FUSED2_H

#include "softmax_kernel.h"

// Fused 2-kernel softmax using cooperative groups with grid-stride loops
//
// Architecture:
// - Kernel 1: Block-level statistics with grid-stride loop (regular launch)
//   * Each thread processes multiple elements via grid-stride loop
//   * Reduces memory footprint and allows larger inputs
// - Kernel 2: Fused global reduce + normalize (cooperative launch with grid sync)
//   * Normalization also uses grid-stride loop
//
// Grid-Stride Loop Pattern:
//   Each thread handles multiple elements: for (i = idx; i < n; i += stride)
//   This allows processing arbitrarily large inputs with limited block count
//
// Input Size Support:
//   Supports very large inputs (tested up to 10M+ elements)
//   Limited only by available GPU memory, not by cooperative block count
//   Uses maximum available cooperative blocks (~1,056 on H100)
//   Each thread processes ~373 elements for 1M input (1M / (1056*256))
//
// Algorithm:
//   Kernel 1: Each block computes local max and local sum(exp(x - block_max))
//   Kernel 2: Cooperative kernel with THREE phases:
//     Phase 1: All blocks cooperate to compute global max
//     Grid sync #1
//     Phase 2: All blocks compute adjusted global sum
//     Grid sync #2
//     Phase 3: Each block normalizes its portion of output
//
// Performance Characteristics:
//   - SLOWER than 3-kernel for most practical workloads (4-14x slower)
//   - Bottleneck: Normalization phase limited by cooperative block count
//   - With only ~1,056 cooperative blocks, each thread processes many elements sequentially
//   - 3-kernel approach is faster because normalization uses full GPU parallelism
//   - Educational value: Demonstrates cooperative groups and their limitations
//
// Requirements:
//   - CUDA 9.0+ (cooperative groups)
//   - Compute capability 6.0+ (grid synchronization)
//   - Device must support cudaLaunchCooperativeKernel
//   - Input size must fit within GPU's cooperative block limit
//
// When to use:
//   - Input size ≤ 1M elements (on H100)
//   - Need maximum performance for medium-sized inputs
//   - Device supports cooperative launch
//   - For learning advanced CUDA techniques
//
// When NOT to use:
//   - Input size > 1M elements → Use softmax_Fused (3-kernel) instead
//   - Older GPUs (compute capability < 6.0) → Use softmax_MultiPass
//   - Maximum portability needed → Use softmax_MultiPass

// Class-based interface for accurate profiling
class Fused2Softmax : public SoftmaxKernel {
private:
    float *d_block_maxes, *d_block_sums;
    float *d_global_max, *d_global_sum;
    int n, threadsPerBlock;
    int numBlocks_K1, numBlocks_K2;

public:
    // Constructor: Allocate intermediate buffers
    Fused2Softmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free intermediate buffers
    ~Fused2Softmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_Fused2(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
