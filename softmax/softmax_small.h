#ifndef SOFTMAX_SMALL_H
#define SOFTMAX_SMALL_H

#include "softmax_kernel.h"

// Single-block softmax kernel optimized for small-to-medium inputs
// Uses 256 threads (8 warps) with hybrid warp shuffle + shared memory reduction
// Minimal synchronization overhead (only 2 __syncthreads() calls)
// Optimal for 1K-8K elements, scales to any size (but inefficient for very large inputs)
class SmallSoftmax : public SoftmaxKernel {
public:
    SmallSoftmax(int n, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~SmallSoftmax() = default;

private:
    int n;
};

// Legacy C-style API for backwards compatibility
float softmax_Small(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif  // SOFTMAX_SMALL_H
