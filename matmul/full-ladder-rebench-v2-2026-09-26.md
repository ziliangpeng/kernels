# Full Ladder Re-Bench v2 — Corrected Build (no -dc), Single Session

**Date**: 2026-09-26
**GPU**: H100 80GB HBM3 (gcp5) — node gcp5-h100-0-15, Slurm jobs 219232/219237
**Build**: plain `-c` per-TU compile, NO relocatable device code (see
[docs/kb/nvcc-dc-perf-cliff.md](../docs/kb/nvcc-dc-perf-cliff.md) — the
previous parallel build used `-dc`, costing 30-60% on this kernel set and
tainting all 16-bit ratios measured before this fix).

## FP32 ladder (N=4096, TFLOPS)

| Method | TFLOPS | % cuBLAS | Old-repo reference |
|---|---:|---:|---:|
| naive | 5.29 | 10.2% | 5.3 ✓ |
| coalesced | 5.73 | 11.0% | 5.7 ✓ |
| smem | 8.99 | 17.3% | 9.0 ✓ |
| 1d_blocktile (default) | 17.56 | 33.8% | 17.6 ✓ |
| 1d tuned winner (64,64,4,16) | 19.38 | 37.3% | 19.3 ✓ |
| 2d_blocktile (default) | 22.21 | 42.8% | 22.4 ✓ |
| vectorized (default) | 32.71 | 63.0% | 32.9 ✓ |
| warptile | 28.13 | 54.2% | 28.1 ✓ |
| warptile_dbuf (autotuned default) | 37.59 | 72.4% | 37.6 ✓ |
| cuBLAS FP32 | 51.93 | 100% | 51.9 ✓ |

**All historical numbers reproduce within noise.** The archive-repo
autotune numbers also reproduce on this cluster: 2d_auto 34.1, vectorized_auto
34.9, warptile_auto 39.3 (job 219227, different node).

## 16-bit rungs (FP16 storage + FP32 acc), corrected

| Method | TF32→ | TFLOPS | vs FP32 same-session |
|---|---:|---:|---:|
| naive_f16 | | 4.70 | 0.89x |
| coalesced_f16 | | 5.26 | 0.92x |
| smem_f16 | | 9.36 | **1.04x** |
| 1d_blocktile_f16 (default 64,64,8,8) | | 17.21 | 0.98x |
| 1d tuned FP16 winner (32,32,4,8) | | 16.30 | 0.84x vs tuned FP32 |

**The crossover story is revised by the corrected build**: with the untainted
baselines, FP16 crosses 1.0x at the SMEM rung (1.04x, the most GMEM-traffic-
heavy tiled rung) and falls back to 0.98x at 1D (higher arithmetic intensity —
traffic halving matters less, cvt tax still paid). Previous "crossover at 1D
(1.009x)" was an artifact of the -dc-slowed FP32 baseline.

## Open items

1. **FP16 sweep anomaly**: the in-process sweep picked (32,32,4,8) at 16.30T
   as FP16 winner while the default config (64,64,8,8) measures 17.21T in the
   same binary — the sweep's own numbers must contain (64,64,8,8) < 16.30,
   contradicting the benchmark path. Suspect: sweep timing methodology (3
   warmup + 100-iter back-to-back) vs benchmark path (10 warmup, separate
   timing). Full sweep CSV needed (previous run only kept the BEST line).
2. **warptile_auto 39.3T** (archive repo, gcp5) exceeds our dbuf winner 37.59T
   cross-node — needs a same-session comparison before trusting; if real,
   the tuned warptile optimum rivals double buffering.
3. Auto classes (1d/2d/vectorized/warptile) still live only in
   archive-cudakernels; port decision pending.
