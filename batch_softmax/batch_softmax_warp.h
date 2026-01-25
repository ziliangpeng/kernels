#ifndef BATCH_SOFTMAX_WARP_H
#define BATCH_SOFTMAX_WARP_H

#include "batch_softmax_kernel.h"

// Warp-level batch softmax kernel: one warp (32 threads) per row
// Uses warp shuffle intrinsics for ultra-low overhead reductions
// No shared memory needed, no __syncthreads() calls
//
// LAUNCH CONFIGURATION:
// - Grid: (batch_size, 1, 1) - one block per row
// - Block: (32, 1, 1) - single warp per block
//
// OPTIMAL USE CASE:
// - Small dimensions (dim <= 1024) where warp shuffles are sufficient
// - Very low kernel launch overhead
// - Each thread processes dim/32 elements via grid-stride loop
//
// ALGORITHM PER ROW:
// Phase 1: Each thread finds local max, warp shuffle reduction to global max
// Phase 2: Each thread computes local sum, warp shuffle reduction to global sum
// Phase 3: Each thread normalizes its elements
class WarpBatchSoftmax : public BatchSoftmaxKernel {
public:
    WarpBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~WarpBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
};

#endif  // BATCH_SOFTMAX_WARP_H
