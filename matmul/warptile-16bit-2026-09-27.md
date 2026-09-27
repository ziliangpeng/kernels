# Warptile FP16 — Rung 10

**Date**: 2026-09-27
**Kernel**: `matmul/matmul_warptile_typed.cu` (method `warptile_f16`)
**GPU**: H100 80GB HBM3 (gcp5), node gcp5-h100-0-12, job 219296
**Config**: strict mirror of the FP32 baseline (BM=128, BN=128, BK=16, WM=64, WN=64, TM=8, TN=4, 128 threads).
**Verification**: max rel err 5.08e-05 — PASS.

## Results (N=4096, same session/node)

| Method | TFLOPS | note |
|---|---:|---|
| warptile FP32 | 28.14 | baseline (matches 28.13 established) |
| **warptile_f16** | **29.87** | **1.062x — FP16 wins by 6.2%** |
| vec_autotune_f16 winner (context) | 36.50 | current FP16 ladder top |

## Findings

1. **Second dtype win, and the cleanest one**: hard-coded vs hard-coded,
   default configs, same binary — no template tax, no sweep methodology. FP16
   beats FP32 by 6.2% at the warp-tiling rung.
2. Mechanism (consistent with the ladder story): the warptile rung is
   SMEM-heavy (BK=16 deep tiles); FP16 halves SMEM bytes and footprint
   (16KB→8KB per block), doubling potential block residency and cutting SMEM
   read traffic on the regA/regB path. The cvt tax (~1 cvt per 8 FLOP at this
   config's reuse depth) is smaller than the SMEM win.
3. Note the default config is NOT the warptile optimum for either dtype
   (FP32 tuned winner: 39.07T at (128,128,16,16,4,64,32)); this row measures
   the default-config dtype delta, per ladder convention.

## Ladder state (FP16 column, default configs unless noted)

naive 0.89x → coalesced 0.92x → smem 1.04x → 1D 0.98x → 2D 0.97x →
vectorized 1.003x → **warptile 1.062x** | tuned tops: 2D f16 33.12T,
vec f16 **36.47T** (ABAB-confirmed)
