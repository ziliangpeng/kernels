#ifndef SOFTMAX_CUB_BLOCK_H
#define SOFTMAX_CUB_BLOCK_H

#include "softmax_kernel.h"

// CUB Block-Level softmax implementation
//
// Uses NVIDIA's CUB (CUDA Unbound) library for optimized block-level reductions.
// CUB is part of the CUDA Core Compute Libraries (CCCL) and provides highly
// optimized primitives that handle warp shuffles, shared memory, and bank conflicts
// automatically across all GPU architectures.
//
// Architecture: 3-kernel approach with CUB optimizations
// - Kernel 1: Block statistics using CUB BlockReduce (max + exp-sum)
// - Kernel 2: Global reduce using CUB BlockReduce (1 block)
// - Kernel 3: Normalize (reuses elementwise kernel)
//
// Benefits over hand-written reductions:
// - Less code: CUB handles shared memory management and synchronization
// - Better performance: NVIDIA-optimized for all GPU architectures
// - More maintainable: Template-based, type-safe interface
// - Portable: Works optimally on Pascal, Volta, Turing, Ampere, Hopper
//
// CUB BlockReduce Features:
// - Automatically selects optimal algorithm (warp shuffle vs shared memory)
// - Handles bank conflicts and memory coalescing
// - Provides Sum(), Max(), and custom reduction operators
// - Template specialization for block size at compile time
//
// Performance characteristics:
// - Expected: Similar or slightly better than hand-written fused3
// - CUB overhead is minimal (compile-time template specialization)
// - Memory access patterns are optimal
//
// Requirements:
// - CUDA 11.0+ (CUB is included in CUDA Toolkit)
// - C++11 or later

// Class-based interface for accurate profiling
class CubBlockSoftmax : public SoftmaxKernel {
private:
    float *d_block_maxes, *d_block_sums;
    float *d_global_max, *d_global_sum;
    int n, threadsPerBlock, numBlocks;

public:
    // Constructor: Allocate intermediate buffers
    CubBlockSoftmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free intermediate buffers
    ~CubBlockSoftmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_CubBlock(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
