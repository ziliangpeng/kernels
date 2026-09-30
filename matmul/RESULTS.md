# GEMM Performance Results — Single Source of Truth

**Repo**: ~/code/kernels · **GPU**: NVIDIA H100 80GB HBM3 (gcp5, SM90, CUDA 12.4)
**Workload**: N×N×N FP32 GEMM, N=4096 (unless noted) · **Timing**: CUDA events, 10 warmup + 100 batched iterations
**Semantics**: FP16/BF16 rows are 16-bit STORAGE + FP32 ACCUMULATION (scalar FMA, no Tensor Core)
**Last updated**: 2026-09-27

> This file is the single place to look for ladder numbers. Worklogs
> (`matmul/*-YYYY-MM-DD.md`) hold the full analysis; this file holds the table.
> Ratios are same-session/same-node only (ABAB protocol). Cross-node numbers
> only compare trends, never percentages.

## Full ladder — all variants × all precisions (TFLOPS, N=4096)

| # | Rung (optimization) | Default config | FP32 | FP16 | FP16/FP32 | BF16 | Best tuned (dtype) |
|---|---|---|---:|---:|---:|---:|---|
| 1 | naive | 32×32 tile | 5.29 | 4.70 | 0.89x | 4.66 | — |
| 2 | coalesced | — | 5.73 | 5.26 | 0.92x | — | — |
| 3 | smem tiling | 32×32 | 8.99 | 9.36 | **1.04x** | — | — |
| 4 | 1D blocktile | 64,64,8,8 | 17.56 | 17.23 | 0.98x | — | — |
| 4a | 1D blocktile **autotuned** | (64,64,4,16) f32 / (32,32,4,8) f16 | 19.36ᵃ | 16.31ᵃ | 0.84x | — | +10% (f32) |
| 5 | 2D blocktile | 128,128,8,8,8 | 22.21 | 21.47 | 0.97x | — | — |
| 5a | 2D blocktile **autotuned** | (128,128,16,16,8) f32 / (128,128,8,16,8) f16 | 33.78ᵇ | 33.12ᵇ | 0.98x | — | +52% (f32) / +54% (f16) |
| 6 | vectorized (16B loads) | 128,128,8,8,8 | 32.73 | 32.83 | **1.003x** | — | — |
| 6a | vectorized **autotuned** | f32:(128,128,16,8,8) / f16:(128,64,8,16,8) | 33.49ᵇ | **36.47**ᵈ | **1.091x** | — | +2% (f32) / +11% (f16) |
| 10 | warptile | 128,128,16,64,64,8,4 | 28.13 | **29.87** | **1.062x** | — | — |
| 10a | warptile **autotuned** | f32:(64,128,8,8,4,32,64) / f16:(128,128,16,8,4,32,64) | 37.71ᵇ | 32.39ᵇ | **0.859x** | — | +34% (f32, pruned space) / +9% (f16) |
| 12 | warptile + dbuf (cp.async) | f32:(128,256,8,64,64,2,8,4) / f16:(128,256,16,64,64,2,8,4) | 37.60ᵇ | 34.16ᵉ | 0.908x | — | +34% (f32) / +2% (f16) |
| 9a | **WMMA fragment API** (naive TC, GMEM-direct fragments) | 16,16,16 | — | 27.55ᶠ | — | 27.56 (bf16) | first TC data point |
| 9c | `wgmma_v2` (m64n64k16, 1 wg, single-buffer) | 64,64,16 | — | 104.34 | — | — | first TC number; the reference |
| 9c | `wgmma_v3` (v2 + dbuf + wait_group 1) | 64,64,16 | — | 100.27 | — | — | null: 64-tile is HBM-bound |
| 9c | `wgmma_v4` (128x128 CTA, 4 wgs, dbuf) | 128,128,16 | — | 135.08 | — | — | one macro x 4 quadrants |
| 9c | `wgmma_v5` (v4 + cp.async, 2-stage) | 128,128,16 | — | 130.57 | — | — | negative: async alone buys nothing |
| 9c | `wgmma_v5_1` (cp.async, 4-stage) | 128,128,16 | — | 142.46 | — | — | depth—not async—hides latency |
| 9c | `wgmma_v6` (m64n128k16, 2 wgs, 4-stage) | 128,128,16 | — | **163.17** | — | — | champ; 71% of HBM cap |
| 9c | `wgmma_v7` (m64n256k16, 128x256 CTA) | 128,256,16 | — | 163.41 | — | — | same-node flat vs v6: NOT bandwidth-bound |
| 9c | `wgmma_v8` (warp spec: producer wg + 2 consumers, mbarrier) | 128,256,16 | — | 150.52 | — | — | PASS but wait<0>-serialized (correctness-first); cross-node |
| 9c | `wgmma_v8_1` (v8 + wgmma overlap, wait<1> release) | 128,256,16 | — | 145.78 | — | — | overlap LOST 5T vs serialized v8 — consumer spin-wait starves SMSPs; warp-spec on this shape is a dead end without TMA |
| 9c | `wgmma_v9` (TMA bulk + SWIZZLE_128B, warp-spec) | 128,256,16 | — | **305.02** | — | — | +87% vs v6; TMA×swizzle = designed pair (sweep-verified SBO=1024); 42% of cuBLAS |
| 9c | `wgmma_v9_1` (v9 + overlap + 3-stage, CTA 128x128) | 128,128,16 | — | 253.15 | — | — | -52T vs v9: shape cost (AI 85→64, m64n128 < m64n256 per-instr) swamped overlap gain; KEY: H100 opt-in SMEM is 227KB not 100KB |
| 9c | `wgmma_v9_2` (v9 shape + 3-stage 144KB + overlap, 227KB opt-in) | 128,256,16 | — | **311.42** | — | — | +6.4T only: shape near-saturated at 2-stage; depth/overlap not the remaining bottleneck (wave quantization / L2 reuse next) |
| 9c | `wgmma_v9_3` (v9.2 + GROUP_M=8 rasterization swizzle) | 128,256,16 | — | 309.49 | — | — | flat @4096 (tail-wave masks it) but +25.6T @N=8192 (452.4T, 59% of cuBLAS) — L2-reuse confirmed at scale |
| 9c | `wgmma_v9_4` (2D 8m×2n rect swizzle) | 128,256,16 | — | FAIL* | — | 428.6 @8192 | INCOMPLETE (ragged-N bijection bug unfixed; 428.6 not a clean datapoint; trend negative but unfalsified) — relabeled per Fable review 2026-09-30 |
| 9c | `wgmma_v9_5` (TMA epilogue: SMEM staging + bulk store) | 128,256,16 | — | 331.56 | 409.6 | 460.9 | +22T@4096 — C write amplification was real |
| 9c | `wgmma_v9_6` (4-stage 192KB) | 128,256,16 | — | 336.67 | 433.4 | 491.9 | +27T@8192 — TMA latency headroom pays |
| 9c | `wgmma_v9_7` (TMA epilogue + 4-stage) | 128,256,16 | — | **367.05** | **460.4** | **514.8** | synergy +57.6 vs +49.3 additive; 67.6% of cuBLAS @8192 |
| 9c | `wgmma_v9_8` (persistent CTAs, runtime G) | 128,256,16 | — | 367.19 | 457.3 | 508.7 | negative: wave-quantization falsified; per-tile pipeline drain > wave savings; G-sweep confirms G=8 |
| 9c | `wgmma_v9_9` (setmaxnreg 40/232 rebalance) | 128,256,16 | — | 368.93 | 461.1 | 511.4 | flat — consumers never register-starved (0 spill, 154 regs); Fable #1 falsified at this design point |
| 9c | `wgmma_v9_10` (persistent cooperative + async epilogue) | 128,256,16 | — | 318.60 | 387.9 | 432.4 | NEGATIVE -13~16%: 1 CTA/SM + serialized epilogue loses v9.7's free CTA-rotation overlap; N=3072 hang unresolved; true ping-pong (per-wg tile streams, 224KB @ 2-stage) is the faithful Fable #2 |
| — | **cuBLAS FP32** | — | 51.93 | — | — | — | — |
| — | **cuBLAS FP16** (FP32 compute) | — | — | 728.7 | — | — | — |
| — | **cuBLAS BF16** (FP32 compute) | — | — | — | — | 469.8ᶜ | — |

ᵃ template+dispatch build — carries ~7.5% codegen tax vs hard-coded (ABAB
experiment, `matmul/abab-tuned-vs-typed-2026-09-27.md`); autotune output should
be re-built hard-coded for production numbers.
ᵇ same template-tax caveat; FP32 2D winner (128,128,16,16,8) was independently
confirmed against the archive hard-coded auto (33.80T, 0.06% match).
ᶜ node h100-0-4; archive reference 493.6T (cross-node ±5%).
ᵈ ABAB-confirmed hard-coded production number (job 219295): B/A=1.0000 vs
dispatch path — vec template carries NO dispatch tax (the 1D rung's 7.5% tax
was 1D-specific); sweep value 36.53T reproduces to 0.2%.
ᵉ same-tool ratios (both from our 136-config template sweep, same node):
winner/winner 0.859x; per-config systematic — TM=8 family 0.835-0.859x,
TM=16 family 0.76-0.79x (cvt tax grows with accumulator depth). The archive
hard-coded warptile_auto (39.07T full-space) retains the FP32 crown: our
pruned-space template FP32 winner is 37.71T — the warptile template has a
codegen gap vs hard-coded that the vec template does not (open item).

### Simon Boehm numbering map (why rows jump: 7 absorbed, 8→12, 9→10)

| Simon # | Optimization | This repo |
|---|---|---|
| 0-6 | naive … vectorized | rungs 1-6 |
| 7 | bank-conflict elimination (SMEM padding) | built INTO 2D/vectorized (`As[BM][BK+1]` padding), not a separate row |
| 8 | cp.async double buffering | rung 12 (dbuf) |
| 9 | warptile (register-level warp tiling) | rungs 10/10a |
| 9a/b/c | WMMA / PTX MMA / CUTLASS | future WGMMA rungs; cuBLAS FP16 is the library ceiling |
| 11-12 | Strassen / block-recursive | not done — algorithmic FLOP-count variants, orthogonal to the memory-hierarchy teaching ladder |

## Reading the table

- **FP32 column**: scalar-FMA ceiling is warptile_auto 39.07T = 75.2% of
  cuBLAS FP32. The ladder teaches tiling → SMEM → vectorization → warp tiling
  → double buffering.
- **FP16 ordering inverts at the top**: for FP32, warptile (39.07) > vec (34.65); for FP16, vec (36.49) > warptile tuned (32.40) — deep register tiling pays cvt on every SMEM read at shallow reuse depth, so 16-bit favors load vectorization (matmul/warp-autotune-2026-09-27.md).
- **FP16 column**: 16-bit storage wins where GMEM traffic dominates (smem rung
  1.04x) and pays a cvt tax where issue-bound; vectorized rung reaches parity
  (1.003x) but no scalar kernel can beat ~33T — the cvt pipe (cvt:FMA ≈ 1:1,
  throughput 1:8 vs FMA) is the hard ceiling. cuBLAS FP16 (728.7T, Tensor
  Core, no cvt) is 22x above the scalar-FMA FP16 ladder.
- **% of cuBLAS (FP32)**: naive 10.2% → 1D 33.8% → 2D tuned 65.1% →
  vectorized 63.0% → warptile_auto 75.2% → dbuf 72.4%.

## Config winners (autotune)

| Kernel | FP32 winner | FP16 winner |
|---|---|---|
| 1D (13 cfgs) | (128,128,4,32) 19.36T | (32,32,4,8) 16.31T |
| 2D (19 cfgs) | (128,128,16,16,8) 33.78T | (128,128,8,16,8) 33.12T |
| vectorized (16 cfgs) | (128,128,16,8,8) 33.49T | (128,64,8,16,8) 36.53T |
| warptile (136 pruned cfgs, template) | (64,128,8,8,4,32,64) 37.71T | (128,128,16,8,4,32,64) 32.39T |
| warptile dbuf (1360 cfgs) | (128,256,8,64,64,2,8,4) 37.60T | — |

Notable: FP32 and FP16 optima DIVERGE (dtype-dependent optima). FP16 prefers
shallow BK + big TM (SMEM bytes halve → more K-rounds affordable); FP32 prefers
deep BK.

## Provenance

| Number | Job | Node | Worklog |
|---|---|---|---|
| FP32 defaults + cuBLAS FP32 | 219232/219237 | h100-0-15 | full-ladder-rebench-v2-2026-09-26.md |
| FP16 rungs 1-4 | 219237 | h100-0-15 | same |
| 1D autotune f32/f16 | 219241→219273 | h100-0-19 | 1d-autotune-2026-09-26.md |
| 2D autotune f32 | 219279 | h100-0-19 | 2d-autotune-2026-09-27.md |
| 2D autotune f16 + vectorized_f16 | 219280/219289 | h100-0-19 / h100-0-4 | 2d-autotune / vectorized-16bit worklogs |
| warptile_auto / dbuf | 219256 | h100-0-19 | autotune-port-2026-09-26.md |
| vec autotune f32/f16 | 219293 | h100-0-0 | vec-autotune-2026-09-27.md |
| warptile autotune f16/f32 | 219300/219305/219306/219307 | h100-0-54 | warp-autotune-2026-09-27.md |
| cuBLAS FP16/BF16 | 219286 | h100-0-4 | 2d-autotune-2026-09-27.md addendum |

Sweep CSVs live in `matmul/`: `1d-autotune-f32/f16-gcp5-h100-2026-09-26.csv`,
`2d-autotune-f32-f16-gcp5-h100-2026-09-27.csv`, `dbuf-sweep-gcp5-h100-2026-09-26.csv`.

## Method notes (why numbers are trustworthy)

1. Same-session/same-node ratios only; cross-node comparisons are trend-only
   (measured node-to-node FP16 variance ~5-7% on identical binaries).
2. Template/dispatch autotune builds carry a measured ~7.5% codegen tax vs
   hard-coded builds (ABAB ×7, clock pinned, batched==single event timing).
3. Sweeps check `cudaGetLastError` after warmup — a sweep that doesn't can
   "discover" impossible configs from empty-queue timing (205,796 TFLOPS bug,
   fixed 2026-09-27).
4. Build flags matter: `-dc` cost 30-60% on this kernel set
   (docs/kb/nvcc-dc-perf-cliff.md). All numbers above are no-dc builds.
5. WGMMA v2-v7 ladder re-measured 2026-09-29 on ONE node (h100-0-3), one
   process per method, each version its own single-instance TU kernel
   (`wgmma_v2`..`wgmma_v7`) — cleanest measurement of the family; v6 163.17
   vs v7 163.41 confirms the "bigger tile gains nothing" result same-node.
   Earlier cross-node numbers (104.36/135.82/142.42/164.83/162.73) are
   superseded by the same-node rows.

## Methodology note: H100 non-Tensor FP16 rate (verified 2026-09-28, GH100 whitepaper)

Official peaks (SXM5): FP32 non-Tensor 66.9T, **FP16 non-Tensor 133.8T = 2x FP32**
(BF16 non-Tensor also 133.8T). The 2x belongs to the half2-packed path (HFMA2),
whose accumulator is half2 — i.e. it requires FP16 ACCUMULATION. Our ladder's
semantics (FP16 storage, cvt, FP32 accumulation) cannot use packed math at all,
so the scalar-path ceiling stands as measured. Correction to our earlier claim
that "H100 CUDA-core FP16 rate = FP32 rate": the rate is 2x, but it is locked
behind fp16-accumulator numerics (K=4096 sequential fp16 accumulation: worst-case
rel err ~n·eps ≈ 200%, RMS ~3%). A hybrid rung (fp16 partial sums every ~64
steps + fp32 flush, RMS ~0.4%) could partially unlock the packed rate — open
experiment. Sources: NVIDIA H100 whitepaper architecture table (gtc22-whitepaper-hopper.pdf,
PB-11133-001, techpowerup GH100 PDF — all consistent).
