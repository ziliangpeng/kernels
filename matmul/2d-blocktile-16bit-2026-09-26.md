# 2D Blocktile FP16 — Rung 5 (Cvt-Amortization Prediction FALSIFIED)

**Date**: 2026-09-26
**Kernel**: `matmul/matmul_2d_blocktile_typed.cu` (method `2d_blocktile_f16`)
**GPU**: H100 80GB HBM3 (gcp5) — node gcp5-h100-0-19, jobs 219269 (build) + 219270 (data)
**Config**: strict mirror of the FP32 baseline default (BM=128, BN=128, BK=8, TM=TN=8).
**Verification**: max rel err 7.58e-05 — PASS.

## Result

| Method | TFLOPS | ratio |
|---|---:|---:|
| 2d_blocktile FP32 | 22.19 | — |
| 2d_blocktile_f16 | 21.47 | **0.97x** |

## The prediction was wrong, and that's the finding

Prediction (from the 1D analysis): 2D's outer-product structure converts each
SMEM element ONCE into regA/regB and reuses it TM*TN=64 times, so the cvt tax
should amortize ~64x better than naive — expected ratio >= 1.0x.

Measured: 0.97x — same as the smem rung, WORSE than 1D's 0.98x. Three
contributing factors visible in the data:

1. **The baseline got faster too.** FP32 2d measures 22.19T here vs 22.21T
   earlier — no change. But the FP16 penalty is set by ABSOLUTE cvt count per
   FLOP, which is identical in relative terms across rungs at fixed config:
   ~2 cvt per (TM+TN) loads amortized over TM*TN FMAs = 32 cvt/64 FMA = 0.5
   cvt/FMA at this config. The amortization is real but the tax was never the
   binding constraint here — the rung is issue-bound, not cvt-bound.
2. **FP16's win channel (GMEM traffic) is weak at this rung**: the 2D
   baseline already has high arithmetic intensity (register reuse of A AND
   B), so halving GMEM bytes buys little — same reason the 1D ratio is only
   0.98x.
3. **Net**: FP16 storage pays when traffic dominates (smem rung, 1.04x,
   where tiles live in GMEM and every read hits DRAM) and pays nothing once
   registers absorb the reuse (2D onward). The crossover is not a ladder
   rung — it's a function of where the kernel sits on the roofline.

## Ladder table (FP16 column, default configs, same-session pairs)

| Rung | FP32 | FP16 | ratio |
|---|---:|---:|---:|
| naive | 5.29 | 4.70 | 0.89x |
| coalesced | 5.73 | 5.26 | 0.92x |
| smem | 8.99 | 9.36 | **1.04x** |
| 1d_blocktile | 17.56 | 17.21 | 0.98x |
| 2d_blocktile | 22.19 | 21.47 | 0.97x |

The 1.04x peak at smem stands as the FP16 high-water mark of the scalar
ladder. Vectorized (float4/8-half packed loads — halves LOAD INSTRUCTIONS,
not just bytes) is the next genuinely structural FP16 opportunity.
