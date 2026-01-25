#ifndef BATCH_SOFTMAX_NAIVE_H
#define BATCH_SOFTMAX_NAIVE_H

#include "batch_softmax_kernel.h"

// Naive batch softmax kernel: one CUDA block per row
// Uses block-level parallel reduction with shared memory
// Works for any dimension size, good baseline for comparison
//
// LAUNCH CONFIGURATION:
// - Grid: (batch_size, 1, 1) - one block per row
// - Block: (threadsPerBlock, 1, 1) - threads within block process one row
//
// ALGORITHM PER ROW:
// Phase 1: Find max (parallel reduction within block)
// Phase 2: Compute sum of exp(x - max) (parallel reduction)
// Phase 3: Normalize: output = exp(x - max) / sum
class NaiveBatchSoftmax : public BatchSoftmaxKernel {
public:
    NaiveBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~NaiveBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
    int threadsPerBlock;
};

#endif  // BATCH_SOFTMAX_NAIVE_H
