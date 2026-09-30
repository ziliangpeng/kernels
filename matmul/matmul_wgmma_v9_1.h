#ifndef MATMUL_WGMMA_V9_1_H
#define MATMUL_WGMMA_V9_1_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>

// Kernel "wgmma_v9_1" — v9 (TMA + SWIZZLE_128B) with consumer overlap and a
// 3-stage pipeline.
//
// CTA 128x128 (SMEM: 3 stages x (16KB A + 16KB B) = 96KB). 2 consumer wgs
// run m64n128k16 (v6-verified shape; the swizzle-descriptor walk is covered
// transitively by v9's m64n256 PASS). One commit group per 64-k stage;
// wgmma_wait<1> proves the PREVIOUS stage's group done -> release that
// stage while the current group is still in flight (overlap restored).
//
// Everything else (TMA boxes, swizzle=128B / LBO=1 / SBO=1024 descriptors,
// mbarrier parity math, dynamic-SMEM 1024B alignment) is v9-verified.

class MatmulWgmmaV91 : public MatmulKernel {
public:
    MatmulWgmmaV91(int N, int blockDim);
    ~MatmulWgmmaV91() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
    CUtensorMap tmA, tmB;
    bool mapsReady = false;
    void makeMaps();
};

#endif // MATMUL_WGMMA_V9_1_H
