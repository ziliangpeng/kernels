#ifndef BATCH_SOFTMAX_HYBRID_H
#define BATCH_SOFTMAX_HYBRID_H

#include "batch_softmax_kernel.h"
#include <memory>

// Hybrid batch softmax with adaptive kernel selection
//
// Automatically selects the optimal kernel based on dimension size:
// - dim <= 64:   Warp kernel (32 threads) - minimal overhead
// - dim > 64:    Multi-warp kernel (256 threads, with vectorization if dim % 4 == 0)
//
// This provides a single interface that performs well across all dimension
// sizes without requiring the user to manually select the best kernel.
//
// Selection rationale:
// - Small dims (<=64): Each thread processes <=2 elements, warp kernel is optimal
//   due to zero shared memory and minimal synchronization
// - Larger dims (>64): More threads help with instruction-level parallelism,
//   multi-warp hybrid reduction with vectorization is efficient
//
// The kernel is created once in the constructor and reused in execute() to
// avoid per-call allocation overhead.
class HybridBatchSoftmax : public BatchSoftmaxKernel {
public:
    HybridBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~HybridBatchSoftmax() override = default;

private:
    std::unique_ptr<BatchSoftmaxKernel> kernel_impl;
};

#endif  // BATCH_SOFTMAX_HYBRID_H
