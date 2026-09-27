# 1D Blocktile FP16 — Rung 4 of the 16-bit Ladder (Crossover Confirmed)

**Date**: 2026-09-26
**Kernel**: `matmul/matmul_1d_blocktile_typed.cu` (method `1d_blocktile_f16`)
**GPU**: H100 80GB HBM3 (a-h100-cluster, SM90) — Slurm jobs 219203 (build), 219204/219205 (data)
**Semantics**: FP16 storage (GMEM + SMEM tiles) + FP32 accumulation, default
config BM=64 BN=64 BK=8 TM=8 (NOT autotuned — matches `1d_blocktile`).
**Verification**: max rel err 6.53e-05 — PASS (threshold 5e-3).

## Results (square GEMM, TFLOPS, same session/binary)

| N | 1d_blocktile FP32 | 1d_blocktile_f16 | ratio |
|---:|---:|---:|---:|
| 512 | 4.72 | 4.72 | 1.002x |
| 1024 | 11.11 | 11.38 | 1.024x |
| 2048 | 12.14 | 12.36 | 1.018x |
| 4096 | 12.51 | 12.62 | 1.009x |

## The crossover has happened

Penalty/ratio trajectory by rung (FP16/FP32, same-session pairs):

| Rung | ratio | Register reuse | Why |
|---|---:|---|---|
| 1 naive | 0.67–0.85x | none | pure cvt tax, traffic saving worthless (issue/latency bound) |
| 2 coalesced | 0.84–0.89x | none | same + coalesced loads (still issue bound) |
| 3 smem | 0.97x | none | GMEM traffic halves, starts paying |
| **4 1d blocktile** | **1.009–1.024x** | **TM=8 (B reused across 8 FMAs)** | **AI doubled; halved GMEM traffic now outpays the cvt tax** |

Mechanism: each thread computes TM=8 outputs, so per output element the
GMEM traffic is amortized 8× vs naive. The kernel leaves the
issue/latency-bound regime and enters the traffic-sensitive regime — exactly
where halving bytes with FP16 storage wins. The margin is small (~1-2%)
because the default config is far from the bandwidth roof (12.5T vs ~26.8T
roof at this AI), but it is consistently ≥1.0x at every N.

## Context vs autotuned FP32 (historical, pi1 node)

The FP32 1D rung autotunes to 19.3T with (BM,BK,TM)=(64,4,16) — vs 12.5T for
the default config. The 16-bit variant here uses the DEFAULT config for
rung-apples-to-rung-apples comparison; an FP16-aware autotune pass (16-bit
SMEM halves tile cost → different optimum) is future work.

## Implementation notes

- Same structure as `matmul_1d_blocktile.cu`; FP16 tiles in SMEM (A tile
  64×8×2B=1KB, B tile 8×64×2B=1KB); hot-path cvt on SMEM read (2 cvt per
  TM-group of 8 FMAs — amortized 16× better than smem rung's 2 cvt/FMA).
- B reuse: `tmpB` converted ONCE per dotIdx, reused across all TM=8 FMAs —
  the 1D rung's register-reuse structure also amortizes the cvt.
