#ifndef MATMUL_WGMMA_V6_H
#define MATMUL_WGMMA_V6_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v6" — m64n128k16, 2 warpgroups x 64x128 strips: ONE
// wgmma per K-step per wg (instruction count halved vs v4). B layout
// follows the macro shape: 16 n-atoms x 128B = LBO 2048 (full-128-row
// atoms); A keeps 64-row quadrant layout LBO=1024. 164.83T.

class MatmulWgmmaV6 : public MatmulKernel {
public:
    MatmulWgmmaV6(int N, int blockDim);
    ~MatmulWgmmaV6() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V6_H
