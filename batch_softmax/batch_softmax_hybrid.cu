#include "batch_softmax_hybrid.h"
#include "batch_softmax_warp.h"
#include "batch_softmax_multi_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// ============================================================================
// HYBRID BATCH SOFTMAX: Adaptive Kernel Selection
// ============================================================================
//
// This implementation provides a single interface that automatically selects
// the optimal kernel based on the dimension size at construction time.
//
// KERNEL SELECTION STRATEGY:
// --------------------------
// dim <= 64:    Warp kernel (32 threads)
//   - Each thread processes <=2 elements
//   - Zero shared memory, zero __syncthreads()
//   - Minimal launch overhead
//   - Best: Low latency for small dims
//
// dim > 64:     Multi-warp kernel (256 threads, with vectorization if dim % 4 == 0)
//   - 8 warps = more parallelism
//   - Hybrid reduction (warp shuffles + minimal shared mem)
//   - Vectorized loads (float4) for large dims when possible
//   - Best: Medium to large dims where ILP and bandwidth help
//
// IMPLEMENTATION:
// ---------------
// The appropriate kernel implementation is created once in the constructor
// and stored in a unique_ptr. The execute() method simply delegates to the
// stored kernel, avoiding per-call allocation overhead.
//
// ============================================================================

HybridBatchSoftmax::HybridBatchSoftmax(int batch_size, int dim, int threadsPerBlock) {
    (void)threadsPerBlock;  // Ignored - we select automatically

    // Select and create kernel based on dimension size
    // dim <= 64: warp kernel is optimal (each thread handles <=2 elements)
    // dim > 64: multi_warp kernel provides better parallelism and vectorization
    if (dim <= 64) {
        kernel_impl = std::make_unique<WarpBatchSoftmax>(batch_size, dim, 32);
    } else {
        kernel_impl = std::make_unique<MultiWarpBatchSoftmax>(batch_size, dim, 256);
    }
}

void HybridBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Dispatch to the pre-selected kernel
    kernel_impl->execute(d_input, d_output);
}
