# Full Ladder Re-Benchmark — Single Node, Single Session

> **CORRECTION (2026-09-26, later)**: these numbers were measured with the
> `-dc` (relocatable device code) build, which cost 30-60% on this kernel
> set. Both FP32 baselines and FP16 numbers here are depressed; corrected
> same-session numbers are in `full-ladder-rebench-v2-2026-09-26.md`.
> The by-rung RATIO trend is still directionally valid but the absolute
> values and the exact crossover rung shifted (crossover is at smem, not 1d).


**Date**: 2026-09-26
**GPU**: H100 80GB HBM3 (gcp5, SM90) — node gcp5-h100-0-56, Slurm job 219225
**Purpose**: one clean same-node same-session table for the whole ladder
(replaces mixed pi1-historical + gcp5 numbers), after discovering the old
autotune code never migrated to this repo.

## FP32 ladder + 16-bit rungs (N=4096, TFLOPS)

| Method | TFLOPS | % cuBLAS |
|---|---:|---:|
| naive | 5.28 | 10.2% |
| coalesced | 5.70 | 11.0% |
| smem | 7.02 | 13.5% |
| 1d_blocktile (default 64,64,8,8) | 12.51 | 24.1% |
| 1d tuned winner (128,128,4,32) | 13.34 | 25.7% |
| 2d_blocktile (default) | 18.45 | 35.5% |
| vectorized (default) | 20.49 | 39.4% |
| warptile | 26.88 | 51.7% |
| warptile_dbuf (autotuned default 128,256,8,...) | 37.50 | 72.2% |
| **cuBLAS FP32** | **51.94** | 100% |

## 16-bit (FP16 storage + FP32 acc; naive_bf16 kept for reference)

| Method | TFLOPS | vs FP32 same-session |
|---|---:|---:|
| naive_f16 | 4.71 | 0.89x |
| naive_bf16 | 4.66 | 0.88x |
| coalesced_f16 | 5.25 | 0.92x |
| smem_f16 | 6.88 | 0.98x |
| 1d_blocktile_f16 (default) | 12.62 | 1.009x |
| 1d tuned FP16 winner (128,128,4,32) | 13.24 | 0.992x |

## Observations

1. Same-session ratios hold the by-rung crossover story: 0.89 → 0.92 → 0.98 →
   1.009x (default configs). The tuned 1D configs sit at parity (0.992x).
2. dbuf at 37.50T with the autotuned default config confirms the earlier
   sweep result (37.60T winner) within node noise.
3. Old-repo historical numbers (1D auto 19.3T, 2D auto 34.0T, vectorized auto
   34.8T) remain unverified pending the archive-cudakernels rebuild
   (jobs 219226/219227) — the doc numbers may reflect a different harness
   (per-iter median vs batched mean) or build flags.
4. Note the old-repo DEFAULT 2d/vectorized numbers (22.4/32.9T on pi1) also
   exceed today's same-config numbers (18.45/20.49T) — consistent with a
   systematic harness/flag difference, not just autotune code.

## Data provenance
- Job 219225, node gcp5-h100-0-56, 100-iter batched event timing per method,
  standard harness. Autotune sweeps: job's own in-process sweeps (13 configs,
  3 warmup + 100-iter each).
