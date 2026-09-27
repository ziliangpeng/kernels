# Vectorized FP16 — Rung 6

**Date**: 2026-09-27
**Kernel**: `matmul/matmul_vectorized_typed.cu` (method `vectorized_f16`)
**GPU**: H100 80GB HBM3 (gcp5), node gcp5-h100-0-4, job 219289 (single session, all four measurements)
**Config**: mirror of FP32 vectorized baseline (BM=128, BN=128, BK=8, TM=TN=8); dtype-scaled 16-byte loads (float4 → uint4/half8).
**Verification**: max rel err 5.66e-05 — PASS.

## Results (N=4096, same session/node)

| Method | TFLOPS | note |
|---|---:|---|
| 2d_blocktile_f16 (prev rung) | 21.51 | baseline re-check |
| **vectorized_f16** | **32.83** | **+53% over 2D FP16** |
| vectorized FP32 | 32.73 | same-session control |
| **FP16/FP32 ratio** | **1.003x** | first ≥1.0 at a default config since smem |

## What the data says

1. **Vectorized loads are worth +53% for FP16** (21.5 → 32.8T) — the largest
   single-rung FP16 gain in the ladder. 8-half loads cut load instructions 8x
   AND halve GMEM bytes; both channels pay out here.
2. **But the FP16/FP32 ratio only reaches 1.003x, not the "significant >1"
   I predicted.** The prediction is half-wrong (magnitude), and the reason is
   Fable's conversion-pipe ceiling: with cvt:FMA ~= 1:1 inside the compute
   loop and the cvt pipe at ~1/8 of FMA throughput, the kernel is
   conversion-bound once loads stop being the bottleneck. Vectorization
   removed the load bottleneck; what remains is cvt. FP32 has no cvt at all,
   so both dtypes converge to the same ~32.7T ceiling set by instruction
   issue — hence ratio 1.0.
3. vectorized_f16 (32.83T, hard-coded build) essentially matches the 2D-tuned
   FP16 winner (33.12T, template build carrying the ~7.5% dispatch tax) —
   a hard-coded rebuild of that winner config would lead, per the ABAB
   protocol; both need a same-session ABAB to rank cleanly.

## Ladder state (FP16 column)

| Rung | FP16 TFLOPS | FP16/FP32 |
|---|---:|---:|
| naive | 4.70 | 0.89x |
| coalesced | 5.26 | 0.92x |
| smem | 9.36 | 1.04x |
| 1d_blocktile | 17.23 | 0.98x |
| 2d_blocktile | 21.47 | 0.97x |
| 2d tuned (template) | 33.12 | 0.98x (vs FP32 winner) |
| **vectorized** | **32.83** | **1.003x** |

The scalar-FMA FP16 ladder ends here: every remaining gap to cuBLAS FP16
(728.7T) is behind the cvt pipe and the missing Tensor Core. Next structural
step is WGMMA (FP16 in, FP32 accum — no cvt at all, mma.sync instructions).
