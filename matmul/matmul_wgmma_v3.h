#ifndef MATMUL_WGMMA_V3_H
#define MATMUL_WGMMA_V3_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v3" — v2 + double-buffered SMEM + wgmma.wait_group 1 lag
// pipeline (load tile k+1 while wgmma k runs). Perf null result: the
// 64x64 tile is HBM-bandwidth-bound, so latency hiding had nothing to
// feed. Also the version that exposed the fake-BUILD-OK failure mode
// (undefined wgmma_wait<1> compiled to nothing; bench ran v2 binary).

class MatmulWgmmaV3 : public MatmulKernel {
public:
    MatmulWgmmaV3(int N, int blockDim);
    ~MatmulWgmmaV3() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V3_H
