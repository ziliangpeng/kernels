#ifndef MATMUL_WGMMA_V9_15_H
#define MATMUL_WGMMA_V9_15_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>

// Kernel "wgmma_v9_15" — the 256x128 TALL TILE: fix v9_14's issue limit.
//
// v9_14 NCU diagnosis: at 128x128 with 2 consumer WGs (2 warps/scheduler),
// tensor 69% / mem 71% — issue-limited; m64n128 halves per-instruction FLOP
// and 2 warps cannot hide the issue path.
//
// v9_15 keeps the multicast machinery but doubles the tile along M:
//   - CTA tile 256x128, 640 threads: 1 producer WG + 4 consumer WGs,
//     each consumer WG m64n128k16 (4 WGs x m64 = 256 rows). Consumer warps
//     per scheduler: 4 (vs v9_14's 2) — the hiding-capacity fix.
//   - STAGES=4 (192KB): A [4][256x128B] 32KB/stage + B [4][128x128B]
//     16KB/stage. Same depth as the v9_7/v9_11 champion family.
//   - cluster 2 along M (pair shares the 128-col B tile); B split-issue
//     multicast unchanged (rank r issues n-box r, mask 0x3).
//   - L2-read arithmetic intensity per cluster: 2x32KB(A) + 16KB(B shared)
//     = 80KB per 8.39M FLOP = 104.6 FLOP/B — 20% less L2 read traffic per
//     FLOP than the 128x256 no-cluster design (87.4).
//   - Ledger: full count 2 (my expect_tx A32+B8=40KB local + peer remote
//     8KB = 48KB/lap); free count 8 (both CTAs' 4 consumer WGs, ct==0 of
//     each arrives local + remote — the remote arrive protects the peer's
//     next multicast into my stage).
//   - Epilogue: cluster drain -> stage C 256x128 f32 (128KB) over the
//     drained pipeline region -> 4 TMA stores (box 128n x 64m).
//
// Note (honest framing): per-WG instruction density is still m64n128
// (262K FLOP/wgmma) — the tall tile restores work-per-SM-per-lap and warp
// hiding, not per-instruction FLOP.

class MatmulWgmmaV915 : public MatmulKernel {
public:
    MatmulWgmmaV915(int N, int blockDim);
    ~MatmulWgmmaV915() override;

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

#endif // MATMUL_WGMMA_V9_15_H
