#ifndef BATCH_SOFTMAX_MULTI_WARP_H
#define BATCH_SOFTMAX_MULTI_WARP_H

#include "batch_softmax_kernel.h"

// Multi-warp batch softmax with vectorized loads
//
// Optimized for large dimensions (4K+) using multiple warps per block and
// float4 vectorized memory access for improved bandwidth utilization.
//
// Architecture: Multiple warps (configurable: 4, 8, or 16) per row, one block per row
// - Each warp computes partial (max, sum) via warp shuffles
// - Cross-warp reduction via shared memory (only num_warps values)
// - Vectorized loads (float4) when dim is divisible by 4
//
// Key optimizations:
// 1. Multiple warps: configurable threads vs 32, better utilization for large dims
// 2. Hybrid reduction: warp shuffles first, minimal shared memory second
// 3. Vectorized loads: 4x memory bandwidth improvement
// 4. Minimized synchronization: only 2 __syncthreads() per phase
//
// TUNABLE PARAMETER:
// - num_warps: Number of warps per block (4, 8, or 16)
//   - 4 warps = 128 threads: lower overhead, good for medium dims
//   - 8 warps = 256 threads: balanced (default)
//   - 16 warps = 512 threads: higher parallelism, good for very large dims
//
// Performance characteristics:
// - Best for dim >= 1024 where vectorization benefits outweigh overhead
// - For dim < 256, single warp kernel may be faster due to lower overhead
// - Vectorized path requires dim % 4 == 0
class MultiWarpBatchSoftmax : public BatchSoftmaxKernel {
public:
    // Constructor takes num_warps (4, 8, or 16) which determines threads per block
    // threadsPerBlock parameter is interpreted as num_warps for this kernel
    MultiWarpBatchSoftmax(int batch_size, int dim, int num_warps);
    void execute(const float *d_input, float *d_output) override;
    ~MultiWarpBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
    int num_warps;        // Number of warps per block (4, 8, or 16)
    bool use_vectorized;  // True if dim % 4 == 0
};

#endif  // BATCH_SOFTMAX_MULTI_WARP_H
