#ifndef MATMUL_WGMMA_V4_H
#define MATMUL_WGMMA_V4_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v4" — 128x128 CTA, 4 warpgroups in a 2x2 quadrant split,
// each running the SAME v2 m64n64 macro on its quadrant descriptor
// (base + wgM/wgN*2048: a 64x16 quadrant's atoms span 2048B).
// AI doubles (32 -> 64 FLOP/B). 135.82T.

class MatmulWgmmaV4 : public MatmulKernel {
public:
    MatmulWgmmaV4(int N, int blockDim);
    ~MatmulWgmmaV4() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V4_H
