#ifndef MATMUL_WGMMA_V9_13_H
#define MATMUL_WGMMA_V9_13_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>

// Kernel "wgmma_v9_13" — v9.12 fixed per Fable follow-up review:
// symmetric split-issue 2-CTA multicast, 4-stage depth restored.
//
// v9.12 verdict: PASS but -13% vs v9.7 (320.0/395.4/443.1). Fable's read:
// direction right (multicast saves L2 READ bandwidth — the actual roofline),
// implementation lost to (a) 4→3 stage regression and (b) leader-serialized
// TMA issue. v9.13 fixes:
//   1. STAGES=4 kept (192KB): epilogue staging overlays the DRAINED
//      pipeline region instead of owning separate SMEM.
//   2. SPLIT-ISSUE: each CTA's producer issues its own A (2 boxes, unicast)
//      + HALF of B (2 boxes, multicast mask 0x3). Issue load halves; the
//      follower never queues behind the leader.
//   3. SYMMETRIC accounting: every CTA's full_bar count 2 (own expect_tx
//      32KB = A 16KB + own B-half 16KB; peer's remote expect_tx 16KB =
//      peer's B-half). Every free_bar count 4 (both CTAs' consumers arrive
//      own + remote). No rank-dependent counts.
//   4. Epilogue race guard (latent in v9.12 too): the peer's producer may
//      still fire tail multicasts into my B_s while I stage C over it.
//      cluster.sync() drains BOTH CTAs' mainloops before staging begins.
//
// Same skeleton otherwise: CTA 128x256, band swizzle G=8 on cluster id,
// m64n256k16 consumers, TMA epilogue via SMEM staging + bulk store.

class MatmulWgmmaV913 : public MatmulKernel {
public:
    MatmulWgmmaV913(int N, int blockDim);
    ~MatmulWgmmaV913() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    __half *d_A16;
    __half *d_B16;
    __half *d_Bt16;   // B transposed to [N][K] K-major for TMA boxes
    CUtensorMap tmA, tmB, tmC;
    bool mapsReady = false;
    void makeMaps();
};

#endif // MATMUL_WGMMA_V9_13_H
