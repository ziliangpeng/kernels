#ifndef MATMUL_WGMMA_V9_4_H
#define MATMUL_WGMMA_V9_4_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>

// Kernel "wgmma_v9_3" — v9 shape + 3-stage + consumer overlap; TMA (cp.async.bulk.tensor) + SWIZZLE_128B feeding.
//
// CTA 128x256, STAGES=3 (144KB), consumer wait<1> overlap.
// v9.3 adds GROUP_M=8 rasterization swizzle (L2 reuse for B tiles).
// Single-variable vs v9.2: CTA traversal order only. Same warp-specialized skeleton as v8 (producer wg
// + 2 consumer m64n256k16 wgs), but the producer issues ONE TMA instruction
// per operand tile per stage (vs 768 per-thread 16B cp.async in v8) and TMA
// writes the ASYNC proxy directly — mbarrier complete_tx orders it against
// wgmma reads, so no fence.proxy.async and no per-thread waits.
//
// Descriptor parameters (sweep-verified, wgmma_tma_sweep.cu, 2026-09-29):
//   swizzle mode = 128B (desc bits 63-62 = 1), LBO = 1 (assumed),
//   SBO = 1024 (8 swizzle rows x 128B).
// TMA box: {64 k = 128B, 64 rows} per operand per stage. A tile = 2 boxes
// (128 rows), B tile = 4 boxes (256 rows) — each stage: 6 TMA instructions.
// k16 slice j (j=0..3) descriptor start = stage base + j*32 bytes (the
// sweep verified slices via the same advance).
//
// SMEM budget: A 3 stages x 8KB + B 3 x 16KB = 72KB -> needs the 100KB
// smem carveout (cudaFuncSetAttribute dynamic SMEM).

class MatmulWgmmaV94 : public MatmulKernel {
public:
    MatmulWgmmaV94(int N, int blockDim);
    ~MatmulWgmmaV94() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;   // B transposed to [N][K] K-major for TMA boxes
    CUtensorMap tmA, tmB;
    bool mapsReady = false;
    void makeMaps();
};

#endif // MATMUL_WGMMA_V9_4_H
