#ifndef MATMUL_WGMMA_V8_H
#define MATMUL_WGMMA_V8_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cstdint>

// Kernel "wgmma_v8" — warp specialization + mbarrier pipeline (no TMA).
//
// CTA 128x256, STAGES=3, 384 threads = 3 warpgroups:
//   - wg0 (producer): issues the v7 per-thread 16B cp.async copies only.
//   - wg1, wg2 (consumers): each runs m64n256k16 (the v7-verified macro,
//     layouts, and epilogue) on its 64-row x 256-col output strip.
//
// Synchronization: mbarriers, NO __syncthreads in steady state.
//   full[s] (count 128): producer threads arrive after their cp.async
//     wait_group<0> + fence.proxy.async for stage s; consumers try_wait it.
//   free[s] (count 2): one elected thread per consumer wg arrives after its
//     wgmma_wait<1> proves the PREVIOUS wgmma (reading stage s) completed;
//     producer try_waits it before reusing stage s.
//   Producer may run up to STAGES-1 stages ahead; consumers keep 1 wgmma
//   group in flight each (same as v7).
//
// Why not TMA here: cp.async.bulk.tensor writes DENSE SMEM boxes; our
// no-swizzle K-major interleave (atoms (r/8)*128 + chunk*LBO) is not a
// dense box. Expressing it needs TMA SWIZZLE modes (32/64/128B), which
// re-derive the whole operand layout — its own rung (v9). v8 isolates the
// warp-specialization variable with zero layout risk.

class MatmulWgmmaV8 : public MatmulKernel {
public:
    MatmulWgmmaV8(int N, int blockDim);
    ~MatmulWgmmaV8() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;
};

#endif // MATMUL_WGMMA_V8_H
