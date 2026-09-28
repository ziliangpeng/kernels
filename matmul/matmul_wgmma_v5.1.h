#ifndef MATMUL_WGMMA_V51_H
#define MATMUL_WGMMA_V51_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v5_1" — v5 with STAGES=4 (32KB SMEM), prologue fills 3
// stages, 3 copies in flight. cp.async.wait_group counting: groups newer
// than tile i are only i+1,i+2 -> depth min(2, tilesLeft). 142.42T.

class MatmulWgmmaV51 : public MatmulKernel {
public:
    MatmulWgmmaV51(int N, int blockDim);
    ~MatmulWgmmaV51() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V51_H
