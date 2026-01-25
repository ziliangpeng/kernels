#ifndef MATMUL_SMEM_H
#define MATMUL_SMEM_H

#include "matmul_kernel.h"

// Shared Memory Caching Optimization (Kernel 3 from siboehm.com)
//
// Key optimization: Cache tiles of A and B matrices in shared memory to reduce
// global memory accesses. Each tile is loaded once from global memory and reused
// multiple times from shared memory.
//
// TILING STRATEGY:
// - BM x BK tile of A loaded to shared memory
// - BK x BN tile of B loaded to shared memory
// - Loop over K dimension in chunks of BK
// - __syncthreads() required after each tile load
//
// SHARED MEMORY USAGE:
// - As[BM][BK] + Bs[BK][BN] = 2 * BK * BM floats (for square tiles)
// - With BM=BK=BN=32: 2 * 32 * 32 * 4 = 8KB per block
//
// PERFORMANCE:
// Expected ~12.8% of cuBLAS (1.5x over coalesced) due to SMEM caching.

class MatmulSmem : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension

public:
    MatmulSmem(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulSmem() override;
};

#endif
