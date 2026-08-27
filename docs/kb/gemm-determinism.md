# GEMM Determinism

Tested on: H100 (CUDA 12.6, PyTorch 2.7.1) + MI325X (ROCm 7.14, PyTorch 2.13)
Date: 2026-08-27

## Run-to-run determinism

**All GEMM operations are run-to-run deterministic on both H100 and MI325X.**

Tested: `torch.mm`, `torch.matmul`, `torch.addmm`, `torch.bmm`, `torch.einsum` — 100-200 iterations each, identical checksum every run.

Root cause: standard cuBLAS/rocBLAS GEMM has no `atomicAdd`. Each output cell `C[i][j]` is computed by one thread with a sequential K-loop accumulator. No cross-thread write conflicts.

## Cross-batch non-determinism (batch non-invariance)

Same input produces different output under different batch sizes (M or B dimension). The library selects different algorithms for different M/B values → different accumulation order → different rounding → different results.

### H100 (cuBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ 0.0003% | ❌ 0.04% (only M=16) | ❌ 0.35% (only M=16) |
| `torch.bmm` (vary B) | ❌ 0.0003% | ✅ 0.0000% | ✅ 0.0000% |
| Manual loop (fixed algo) | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% |

cuBLAS `GemmStridedBatched` (used by `torch.bmm`) uses a **fixed kernel for all B values** when inner dimensions (M, K, N) are constant → FP16/BF16 bmm bit-exact.

FP32 bmm still has 0.0003% — FP32 has larger algorithm space (software tiling).

### MI325X (rocBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ 0.0003% | ❌ 0.04% (M≥128) | ❌ 0.17~0.35% (M≥256) |
| `torch.bmm` (vary B) | ❌ 0.0003% | ❌ 0.0434% | ❌ 0.17~0.35% |
| Manual loop (fixed algo) | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% |

rocBLAS does NOT use a fixed kernel for `bmm` across different B values. Algorithm switching thresholds:
- FP32: switches at nearly every M
- FP16: switches at M=128
- BF16: switches at M=256

### TF32 effect (H100 only)

TF32 amplifies cross-batch diff by ~630×:
- TF32 OFF: FP32 diff = 0.0001%
- TF32 ON: FP32 diff = 0.07%

TF32 uses 19-bit mantissa (vs FP32's 23-bit) → larger rounding differences per algorithm switch.

### Cross-vendor bit-exactness

H100 and MI325X produce **completely different checksums** for the same input + weight + dtype. cuBLAS and rocBLAS use different tiling, different K-loop unrolling, different MFMA/Tensor Core internal rounding. Cross-vendor bit-exact GEMM does not exist.

## Why FP16/BF16 are more deterministic than FP32 (hypothesis)

FP16/BF16 GEMM uses Tensor Core / Matrix Core with hardware-fixed tile sizes (16×16×16). This constrains the algorithm space — fewer variants → fewer cross-batch switches.

FP32 GEMM uses CUDA cores with software-selected tiling → more variants → more cross-batch variation.

**Not fully proven (Sol review):** H100 FP32 may use TF32 Tensor Core. Need to query algorithm IDs and disassemble kernels to confirm.

## Test scripts

- `determinism/run_tests.py` — v1 single-run (5 ops, 3 dtypes)
- `determinism/run_tests_v2.py` — v2 multi-run pairwise (200 iterations)
- `determinism/cross_batch_gemm.py` — cross-batch GEMM (mm, vary M)
- `determinism/cross_batch_gemm_v2.py` — multi-run cross-batch GEMM
- `determinism/batched_gemm.py` — batched GEMM (bmm, vary B)
- `determinism/fixed_algo_gemm.py` — 4 approaches to batch invariance
