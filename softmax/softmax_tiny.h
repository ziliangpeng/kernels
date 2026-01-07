#ifndef SOFTMAX_TINY_H
#define SOFTMAX_TINY_H

#include "softmax_kernel.h"

// Single-warp softmax kernel optimized for small inputs
// Uses only 32 threads (1 warp) with warp shuffle primitives
// No shared memory, no __syncthreads(), minimal overhead
// Optimal for 512-1K elements, scales to any size (but inefficient for large inputs)
class TinySoftmax : public SoftmaxKernel {
public:
    TinySoftmax(int n, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~TinySoftmax() = default;

private:
    int n;
};

// Legacy C-style API for backwards compatibility
float softmax_Tiny(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif  // SOFTMAX_TINY_H
