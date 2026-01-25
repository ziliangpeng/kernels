#ifndef BATCH_SOFTMAX_ONLINE_WARP_H
#define BATCH_SOFTMAX_ONLINE_WARP_H

#include "batch_softmax_kernel.h"

// Online warp-level batch softmax with single-pass statistics
//
// Uses the online softmax algorithm that computes (max, sum) in a single pass
// by maintaining running statistics that update incrementally.
//
// Key formula for merging (max1, sum1) + (max2, sum2):
//   merged_max = max(max1, max2)
//   merged_sum = sum1 * exp(max1 - merged_max) + sum2 * exp(max2 - merged_max)
//
// Architecture: One warp (32 threads) per row
// - Each thread maintains its own (max, sum) state
// - Single pass through input data
// - Warp shuffle reduction to combine thread states
// - Second pass for normalization
//
// Advantages:
// 1. Only reads input data once during statistics phase (vs twice in two-pass)
// 2. Better cache efficiency for large dims
// 3. Mathematically equivalent to two-pass algorithm (numerically stable)
//
// Trade-offs:
// - More compute per element (exp() per element vs deferred)
// - Still needs second pass for normalization
// - Best for memory-bound scenarios (large dims)
class OnlineWarpBatchSoftmax : public BatchSoftmaxKernel {
public:
    OnlineWarpBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~OnlineWarpBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
};

#endif  // BATCH_SOFTMAX_ONLINE_WARP_H
