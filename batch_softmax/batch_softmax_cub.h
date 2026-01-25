#ifndef BATCH_SOFTMAX_CUB_H
#define BATCH_SOFTMAX_CUB_H

#include "batch_softmax_kernel.h"

// CUB-based batch softmax implementation
//
// Uses NVIDIA's CUB (CUDA Unbound) library for block-level reductions.
// CUB provides highly optimized primitives that automatically select the
// best algorithm (warp shuffle vs shared memory) based on block size.
//
// Architecture: One CUDA block per row
// - cub::BlockReduce<float, BLOCK_SIZE> handles all reduction operations
// - Automatically uses warp shuffles within warps + shared memory across warps
// - Much simpler code than hand-written reductions with same/better performance
//
// Key CUB benefits:
// 1. Template specialization generates optimal code at compile time
// 2. Handles bank conflicts, synchronization, boundary conditions
// 3. Zero runtime overhead
// 4. Portable across GPU architectures
//
// Algorithm per row:
// Phase 1: Find max using CUB BlockReduce with cub::Max()
// Phase 2: Compute sum(exp(x - max)) using CUB BlockReduce Sum()
// Phase 3: Normalize: output = exp(x - max) / sum
class CubBatchSoftmax : public BatchSoftmaxKernel {
public:
    CubBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~CubBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
    int threadsPerBlock;
};

#endif  // BATCH_SOFTMAX_CUB_H
