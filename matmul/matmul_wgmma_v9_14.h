#ifndef MATMUL_WGMMA_V9_14_H
#define MATMUL_WGMMA_V9_14_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>

// Kernel "wgmma_v9_14" — the multicast redo in the CORRECT geometry.
//
// NCU evidence (NCU-PROFILING-2026-10-02.md): at 128x256 the L2 is only
// 58-68% utilized — not the bottleneck — so multicast saved nothing
// (v9_12/v9_13 flat). At 128x128 the per-FLOP L2 read traffic doubles
// (AI 64 vs 85) — that IS the disease multicast treats. CUTLASS evidence:
// the 743.2T best @4096 is 128x128x64, cluster 2x1, 6 stages.
//
// v9.14 = v9.13's proven multicast machinery transplanted into 128x128:
//   - CTA tile 128x128, 384 threads (1 producer WG + 2 consumer WGs),
//     m64n128k16 consumers (each WG owns m64 of the 128 rows).
//   - cluster 2 ALONG M: CTA pair (2c, 2c+1) = adjacent blockM, SAME
//     blockN -> they share one B tile (128 cols). B split-issue multicast:
//     rank r issues n-box r (64n x 64k, 8KB) with mask 0x3; both CTAs
//     receive the full 128n x 64k B stage.
//   - STAGES=6 (192KB SMEM): A 6x16KB + B 6x16KB. Same opt-in 227KB cap.
//   - Traffic per CTA per k64 lap: A 16KB unicast + B 8KB issued (16KB
//     delivered to both) = 24KB issued vs 32KB non-cluster. Effective AI
//     back to ~85 FLOP/B, but with 2x the CTAs of the 128x256 design.
//   - Symmetric barrier ledger (from v9.13): full count 2 (my expect_tx
//     A16+B8=24KB + peer remote expect 8KB = 32KB/lap), free count 4
//     (both CTAs' 2 consumer WGs; B stage is shared).
//   - B descriptor SBO: B stage = [128 n-rows][128B] contiguous; the
//     m64n128k16 n-atom-group stride = 8 rows x 128B = 1024 (same atom
//     math as v9.13's m64n256). W914_SBO env overrides for safety sweep.
//   - Epilogue: cluster drain -> stage C 128x128 f32 (64KB) over the
//     drained A region -> 2 TMA stores (box 128x64).

class MatmulWgmmaV914 : public MatmulKernel {
public:
    MatmulWgmmaV914(int N, int blockDim);
    ~MatmulWgmmaV914() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;   // B transposed to [N][K] K-contiguous for TMA boxes
    CUtensorMap tmA, tmB;
    bool mapsReady = false;
    void makeMaps();
};

#endif // MATMUL_WGMMA_V9_14_H
