#ifndef MATMUL_WGMMA_V7_H
#define MATMUL_WGMMA_V7_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v7" — 128x256 CTA, m64n256k16, 128 accs/thread.
// Controlled experiment that DISPROVED the bandwidth theory: AI 85 vs
// v6's 64, roofline ~290T — landed FLAT (162.73T). The remaining gap
// is barriers + issue rate, not bandwidth. Next levers: TMA, warp spec.

class MatmulWgmmaV7 : public MatmulKernel {
public:
    MatmulWgmmaV7(int N, int blockDim);
    ~MatmulWgmmaV7() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V7_H
