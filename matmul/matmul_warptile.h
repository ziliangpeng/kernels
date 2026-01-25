#ifndef MATMUL_WARPTILE_H
#define MATMUL_WARPTILE_H

#include "matmul_kernel.h"

// Warp Tiling Optimization (Kernel 10 from siboehm.com)
//
// Key optimization: Add warp-level tiling between block and thread levels.
// This organizes computation to maximize warp scheduler utilization.
//
// THREE-LEVEL HIERARCHY:
// - Block tile: BM x BN (e.g., 128 x 128)
// - Warp tile: WM x WN (e.g., 64 x 64)
// - Thread tile: TM x TN (e.g., 8 x 8)
//
// WARP ORGANIZATION:
// - Each warp (32 threads) computes a WM x WN tile
// - Warps are arranged to maximize scheduler efficiency
// - 4 warps per SM, each handling different output regions
//
// PERFORMANCE:
// Expected ~93.7% of cuBLAS (1.1x over vectorized) - near peak performance.

class MatmulWarptile : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed parameters)

public:
    MatmulWarptile(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWarptile() override;
};

#endif
