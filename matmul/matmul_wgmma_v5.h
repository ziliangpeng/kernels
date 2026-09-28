#ifndef MATMUL_WGMMA_V5_H
#define MATMUL_WGMMA_V5_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v5" — v4 with cp.async 16B staging replacing LDG+STS.
// Negative result: 2-stage depth only overlaps the tail of the wgmma —
// asynchrony alone buys nothing; DEPTH is what hides latency.

class MatmulWgmmaV5 : public MatmulKernel {
public:
    MatmulWgmmaV5(int N, int blockDim);
    ~MatmulWgmmaV5() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V5_H
