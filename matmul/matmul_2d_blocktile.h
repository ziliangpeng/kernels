#ifndef MATMUL_2D_BLOCKTILE_H
#define MATMUL_2D_BLOCKTILE_H

#include "matmul_kernel.h"

// 2D Block Tiling Optimization (Kernel 5 from siboehm.com)
//
// Key optimization: Each thread computes a TM x TN tile of output elements,
// using the outer product formulation in the innermost loop.
//
// CONFIGURATION:
// - BM=128, BN=128, BK=8, TM=8, TN=8
// - Each thread computes 8x8 = 64 C elements
// - Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256
//
// OUTER PRODUCT:
// - Load TM elements of A column and TN elements of B row into registers
// - Compute outer product: regC[i][j] += regA[i] * regB[j]
// - This maximizes register reuse
//
// PERFORMANCE:
// Expected ~68.7% of cuBLAS (1.9x over 1D blocktile) due to 2D tiling.

class Matmul2DBlocktile : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed BM/BN/BK/TM/TN)

public:
    Matmul2DBlocktile(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~Matmul2DBlocktile() override;
};

#endif
