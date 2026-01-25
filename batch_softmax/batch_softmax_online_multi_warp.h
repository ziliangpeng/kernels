#ifndef BATCH_SOFTMAX_ONLINE_MULTI_WARP_H
#define BATCH_SOFTMAX_ONLINE_MULTI_WARP_H

#include "batch_softmax_kernel.h"

// Online multi-warp batch softmax with vectorized loads
//
// Combines the online softmax algorithm (single-pass statistics) with
// multi-warp parallelism and float4 vectorized memory access.
//
// ONLINE SOFTMAX ALGORITHM:
// -------------------------
// Traditional two-pass:
//   Pass 1: max = max(x_i)
//   Pass 2: sum = Σ exp(x_i - max)
//   Pass 3: output = exp(x - max) / sum
//   Total: 3 memory passes
//
// Online single-pass (Milakov & Gimelshein, 2018):
//   Pass 1: Compute (max, sum) together - for each x:
//           old_max = max; max = fmaxf(max, x)
//           sum = sum * exp(old_max - max) + exp(x - max)
//   Pass 2: output = exp(x - max) / sum
//   Total: 2 memory passes (saves one full read of input)
//
// MERGING (MAX, SUM) PAIRS:
// -------------------------
// When combining thread/warp states:
//   merged_max = max(max1, max2)
//   merged_sum = sum1 * exp(max1 - merged_max) + sum2 * exp(max2 - merged_max)
//
// ARCHITECTURE:
// -------------
// - Multiple warps per block (configurable: 4, 8, or 16 warps)
// - Each warp maintains its own (max, sum) state during the online pass
// - Warp-level reduction via shuffles (fast, no shared memory)
// - Cross-warp reduction via shared memory (minimal: only num_warps values)
// - Vectorized loads (float4) when dim % 4 == 0
//
// BENEFITS OVER MULTI-PASS:
// -------------------------
// - One fewer memory pass for statistics computation
// - Better for memory-bound scenarios (large dims)
// - Foundation for FlashAttention-style fused kernels
//
// TUNABLE PARAMETER:
// - num_warps: Number of warps per block (4, 8, or 16)
class OnlineMultiWarpBatchSoftmax : public BatchSoftmaxKernel {
public:
    // Constructor takes num_warps (4, 8, or 16)
    OnlineMultiWarpBatchSoftmax(int batch_size, int dim, int num_warps);
    void execute(const float *d_input, float *d_output) override;
    ~OnlineMultiWarpBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
    int num_warps;        // Number of warps per block (4, 8, or 16)
    bool use_vectorized;  // True if dim % 4 == 0
};

#endif  // BATCH_SOFTMAX_ONLINE_MULTI_WARP_H
