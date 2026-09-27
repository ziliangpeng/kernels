# Warptile FP16 Autotune — Rung 10a

**Date**: 2026-09-27
**Kernel**: `matmul/matmul_warptile_tuned.cu` (methods `warp_autotune_f32/f16`)
**GPU**: H100 80GB HBM3 (gcp5), node gcp5-h100-0-54, jobs 219300 (sweep) + 219305 (4-way same-node check)
**Config space**: 136 configs pruned from archive CANDIDATES_W (479) with FP16 priors (BK∈{8,16}, TM∈{8,16}, WM/WN≠128); FP32 winner (128,128,16,16,4,64,32) retained as anchor.

## Results

| | config | TFLOPS |
|---|---|---:|
| **FP16 winner** | (128,128,16,8,4,WM=32,WN=64) | **32.40** |
| FP16 default (warptile_f16) | (128,128,16,64,64,8,4) | 29.84 |
| FP16 vec winner (same node) | (128,64,8,16,8) | 36.49 |
| FP32 default (same node) | (128,128,16,64,64,8,4) | 28.13 |

- Tuning gain: +8.6% over the FP16 default config.
- **Same-node ranking: vec FP16 (36.49) > warptile FP16 tuned (32.40)** — the
  FP32-side ordering (warptile_auto 39.07 > vec 34.65) INVERTS for FP16.
- Top-10 configs are one family: TM=8, TN=4, WM=32, WN=64 (31.7-32.4T).
- 21 configs LAUNCH_FAIL (256-wide blocks, regs/SM).
- ABAB cross-check: warptile_f16/warptile_f32 = 1.061x here vs 1.062x on
  h100-0-12 — dtype delta reproduces across nodes.

## Why the ordering inverts (analysis)

The FP32 warptile edge comes from deep register reuse (TM×TN per-thread
accumulator arrays giving high ILP). For FP16, every SMEM read pays a cvt
BEFORE reuse, and the warptile inner loop reads As/Bs per dotIdx — with
TM=8,TN=4 the reuse depth (32 FMA per element pair) is shallower than the
2D/vec rungs (TM×TN=128), so the cvt:FMA ratio is higher exactly where FP32
harvests ILP. Meanwhile vec's 8-half vector loads cut load instructions 8x
AND its TM=16/TN=8 winner amortizes cvt over 128 FMAs. Net: for 16-bit
storage, load vectorization beats deep warp-level register tiling — a
genuinely dtype-dependent algorithmic ranking, not a tuning accident.

## Data

CSV: `matmul/warp-autotune-f16-gcp5-h100-2026-09-27.csv` (136 rows + header).
