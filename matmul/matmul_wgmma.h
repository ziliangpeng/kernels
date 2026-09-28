#ifndef MATMUL_WGMMA_H
#define MATMUL_WGMMA_H

#include "matmul_kernel.h"

// Rung 9c: Hopper WGMMA (wgmma.mma_async, warp-group Tensor Core).
// v1 = correctness-first: SS operands (both from SMEM, K-major), no swizzle
// (mode 0), BK=16 so every descriptor sits at a pattern boundary (base
// offset 0), single-buffered, B pre-transposed to K-major on device
// (untimed). FP16 inputs, FP32 accumulate, FP32 out.
//
// A_s[BM=64][BK=16] halfs, row pitch 32B. Core matrix = 8 rows x 16B.
//   LBO = 16 (adjacent core matrices along K), SBO = 8 * 32 = 256 (along M).
// B_t_s[BN=128][BK=16] same (K-major after transpose). desc: trans 0,0.
//
// One warpgroup (128 threads) computes a 64x128 output tile; 64 f32
// accumulators per thread (m64n128 / 128 threads).

class MatmulWgmma : public MatmulKernel {
public:
    MatmulWgmma(int N, int blockDim);
    ~MatmulWgmma() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;  // B transposed to NxK (K-major) for wgmma B operand
};

#endif // MATMUL_WGMMA_H
