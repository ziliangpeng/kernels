#ifndef MATMUL_WGMMA_V2_H
#define MATMUL_WGMMA_V2_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v2" — Hopper WGMMA, m64n64k16, single warpgroup (128 thr).
//
// The reference implementation of rung 9c: everything in it is
// sweep-verified (243-combo brute force, wgmma_desc_sweep.cu):
//   - no-swizzle K-major interleave: core matrix = 8x8 halfs = 128B
//     contiguous; atom (mi, ki) at mi*128 + ki*1024 bytes
//   - descriptors LBO=1024 / SBO=128
//   - m64n64 accumulator mapping: 4 regs per 8-col group
//     (row, row+8) x (col, col+1); 8 groups
//   - fence.proxy.async.shared::cta between SMEM writes and wgmma issue
//     (THE correctness fix — __syncthreads does not order the async proxy)
//
// First passing TC number of the ladder: 104.36T @ N=4096.
// B is pre-transposed on device (untimed). FP16 in / FP32 accumulate.

class MatmulWgmmaV2 : public MatmulKernel {
public:
    MatmulWgmmaV2(int N, int blockDim);
    ~MatmulWgmmaV2() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;  // B transposed to NxK (K-major) for wgmma B operand
};

#endif // MATMUL_WGMMA_V2_H
