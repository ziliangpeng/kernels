# GEMM Determinism

Tested on: H100 (CUDA 12.6, PyTorch 2.7.1) + MI325X (ROCm 7.14, PyTorch 2.13)
Date: 2026-08-27

## Run-to-run determinism

**All GEMM operations tested are run-to-run deterministic on both H100 and MI325X.**

Tested: `torch.mm`, `torch.matmul`, `torch.addmm`, `torch.bmm`, `torch.einsum` — 100-200 iterations each, 1 unique checksum (identical every run).

Root cause: standard cuBLAS/rocBLAS GEMM has no `atomicAdd`. Each output cell `C[i][j]` is computed by one thread with a sequential K-loop accumulator. No cross-thread write conflicts.

## Cross-batch non-determinism (batch non-invariance)

Same input produces different output under different batch sizes (M or B dimension). The library appears to select different algorithms for different M/B values → different accumulation order → different rounding → different results.

**Note:** "Algorithm switching" is our most reasonable hypothesis but is NOT directly verified. We have not queried cuBLASLt/rocBLAS solution IDs or disassembled kernels. The manual-loop experiment (below) strongly supports this hypothesis but does not constitute proof. The most valuable next experiment is recording the actual backend, kernel name, and solution ID per shape.

### H100 (cuBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ not exact | ❌ not exact (only M=16) | ❌ not exact (only M=16) |
| `torch.bmm` (vary B) | ❌ not exact | ✅ bit-exact | ✅ bit-exact |
| Manual loop (fixed shape) | ✅ bit-exact | ✅ bit-exact | ✅ bit-exact |

Relative diff (max\|a-b\| / max\|a\|): FP32 ~0.0003%, FP16 ~0.04%, BF16 ~0.35%.

Bit-exactness verified via checksum (`torch.equal` / MD5), not just percentage formatting.

cuBLAS `GemmStridedBatched` (used by `torch.bmm`) appears to use a **fixed kernel for all B values** when inner dimensions (M, K, N) are constant → FP16/BF16 bmm bit-exact. FP32 bmm still shows diff — FP32 may have a larger algorithm space (software tiling).

**Caveat (Sol review):** FP32 on H100 may actually use TF32 Tensor Core depending on `torch.backends.cuda.matmul.allow_tf32` setting, not pure CUDA-core software tiling. Need to verify actual compute type.

### MI325X (rocBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ not exact | ❌ not exact (M≥128) | ❌ not exact (M≥256) |
| `torch.bmm` (vary B) | ❌ not exact | ❌ not exact (B≥2) | ❌ not exact (B≥2) |
| Manual loop (fixed shape) | ✅ bit-exact | ✅ bit-exact | ✅ bit-exact |

Relative diff: FP32 ~0.0003%, FP16 ~0.0434%, BF16 ~0.17~0.35%.

rocBLAS appears to NOT use a fixed kernel for `bmm` across different B values (unlike cuBLAS). Algorithm switching thresholds (hypothesized from checksum changes):
- FP32: switches at nearly every M
- FP16: switches at B=128 (for bmm) / M=128 (for mm)
- BF16: switches at B=256 (for bmm) / M=256 (for mm)

### TF32 effect (H100 only)

TF32 amplifies cross-batch diff significantly:
- TF32 OFF: FP32 relative diff ~0.0001%
- TF32 ON: FP32 relative diff ~0.07%

TF32 uses 19-bit mantissa (vs FP32's 23-bit). Check `torch.backends.cuda.matmul.allow_tf32` value when reproducing.

**Also check (Sol review):** `torch.backends.cuda.allow_fp16_reduced_precision_reduction` and `torch.backends.cuda.allow_bf16_reduced_precision_reduction` — these flags affect reduction precision and should be recorded.

### Cross-vendor bit-exactness

H100 and MI325X produce **completely different checksums** for the same input + weight + dtype. cuBLAS and rocBLAS use different tiling, different K-loop unrolling, different MFMA/Tensor Core internal rounding. Cross-vendor bit-exact GEMM does not exist.

## Why FP16/BF16 are more deterministic than FP32 (hypothesis)

**Hypothesis (not directly proven):** FP16/BF16 GEMM uses Tensor Core / Matrix Core with hardware-fixed tile sizes (16×16×16). This constrains the algorithm space — fewer variants → fewer cross-batch switches. FP32 GEMM uses software-selected tiling → more variants → more cross-batch variation.

**Counterpoints (Sol review):**
1. H100 FP32 may use TF32 Tensor Core (not pure CUDA core) depending on math mode
2. MI325X also has FP32 MFMA path — FP32 is not necessarily pure software/SIMD
3. Even with Tensor Core, cuBLAS/rocBLAS can still vary CTA tile, warp tile, split-K, edge-tile handling, and epilogue

**To prove:** Query algorithm IDs per shape, disassemble kernels (SASS for H100, ISA for MI325X), force single algorithm and verify cross-batch diff disappears.

## Test scripts

- `determinism/run_tests.py` — v1 single-run (5 ops, 3 dtypes)
- `determinism/run_tests_v2.py` — v2 multi-run pairwise (200 iterations)
- `determinism/cross_batch_gemm.py` — cross-batch GEMM (mm, vary M)
- `determinism/cross_batch_gemm_v2.py` — multi-run cross-batch GEMM
- `determinism/batched_gemm.py` — batched GEMM (bmm, vary B)
- `determinism/fixed_algo_gemm.py` — 4 approaches to batch invariance
