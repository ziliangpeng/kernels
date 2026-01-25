#ifndef MATMUL_COALESCED_H
#define MATMUL_COALESCED_H

#include "matmul_kernel.h"

// Global Memory Coalescing Optimization (Kernel 2 from siboehm.com)
//
// Key optimization: Reorganize thread indexing so consecutive threads access
// consecutive memory locations in B matrix.
//
// MEMORY ACCESS PATTERN:
// - A matrix: Row is broadcast across threads (not coalesced, but cached)
// - B matrix: Coalesced reads (consecutive threads access consecutive columns)
// - C matrix: Coalesced writes
//
// PERFORMANCE:
// Expected ~8.5% of cuBLAS (6.4x over naive) due to better memory access pattern.

class MatmulCoalesced : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension

public:
    MatmulCoalesced(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulCoalesced() override;
};

#endif
