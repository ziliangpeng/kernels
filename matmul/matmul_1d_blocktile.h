#ifndef MATMUL_1D_BLOCKTILE_H
#define MATMUL_1D_BLOCKTILE_H

#include "matmul_kernel.h"

// 1D Block Tiling Optimization (Kernel 4 from siboehm.com)
//
// Key optimization: Each thread computes multiple output elements (TM elements
// in a column of C), reducing thread management overhead and improving arithmetic
// intensity.
//
// CONFIGURATION:
// - BM=64, BN=64, BK=8, TM=8
// - Each thread computes 8 C elements in a column
// - Threads per block: (BM * BN) / TM = 64 * 64 / 8 = 512
//
// REGISTER USAGE:
// - threadResults[TM] array in registers for accumulation
// - Significantly improves arithmetic intensity
//
// PERFORMANCE:
// Expected ~36.5% of cuBLAS (2.8x over smem) due to increased work per thread.

class Matmul1DBlocktile : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed BM/BN/BK/TM)

public:
    Matmul1DBlocktile(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~Matmul1DBlocktile() override;
};

#endif
