# Kernel Learnings

Empirical findings from hands-on kernel testing on NVIDIA H100 and AMD MI325X.

## GEMM Determinism

### Run-to-run determinism

**All GEMM operations are run-to-run deterministic on both H100 and MI325X.**

Tested: `torch.mm`, `torch.matmul`, `torch.addmm`, `torch.bmm`, `torch.einsum` — 100-200 iterations each, identical checksum every run.

This means: same input, same shape, same GPU, run N times → bit-exact identical output. No probabilistic non-determinism in standard GEMM paths.

Root cause: standard cuBLAS/rocBLAS GEMM has no `atomicAdd`. Each output cell `C[i][javb]` is computed by one thread with a sequential K-loop accumulator. No cross-thread write conflicts.

### Cross-batch non-determinism (batch non-invariance)

**Same input produces different output under different batch sizes (M or B dimension).**

This is NOT run-to-run non-determinism. It is cross-batch: the library selects different algorithms for different M/B values, and different algorithms have different accumulation orders → different rounding → different results.

#### H100 (cuBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ 0.0003% | ❌ 0.04% (only M=16) | ❌ 0.35% (only M=16) |
| `torch.bmm` (vary B) | ❌ 0.0003% | ✅ 0.0000% | ✅ 0.0000% |
| Manual loop (fixed algo) | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% |

Key finding: cuBLAS `GemmStridedBatched` (used by `torch.bmm`) uses a **fixed kernel for all B values** when inner dimensions (M, K, N) are held constant. This makes FP16/BF16 bmm bit-exact across batch sizes.

FP32 bmm still has 0.0003% diff because cuBLAS has a larger algorithm space for FP32 (software tiling, no fixed hardware tile).

#### MI325X (rocBLAS)

| Op | FP32 | FP16 | BF16 |
|---|---|---|---|
| `torch.mm` (vary M) | ❌ 0.0003% | ❌ 0.04% (M≥128) | ❌ 0.17~0.35% (M≥256) |
| `torch.bmm` (vary B) | ❌ 0.0003% | ❌ 0.0434% | ❌ 0.17~0.35% |
| Manual loop (fixed algo) | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% |

Key finding: rocBLAS does NOT use a fixed kernel for `bmm` across different B values, unlike cuBLAS. Algorithm switching happens at different B thresholds for each dtype:
- FP32: switches at nearly every M
- FP16: switches at M=128
- BF16: switches at M=256

#### TF32 effect (H100 only)

TF32 amplifies cross-batch non-determinism by ~630×:
- TF32 OFF: FP32 diff = 0.0001%
- TF32 ON: FP32 diff = 0.07%

This is because TF32 uses 19-bit mantissa (vs FP32's 23-bit), making different algorithms' rounding differences much larger.

### Why FP16/BF16 are more deterministic than FP32 (on H100)

FP16/BF16 GEMM uses Tensor Core / Matrix Core with **hardware-fixed tile sizes** (16×16×16). The hardware tile size constrains the algorithm space — fewer algorithm variants means fewer cross-batch switches.

FP32 GEMM uses CUDA cores with software-selected tiling. cuBLAS/rocBLAS can choose different tile sizes, K-loop unrolling, and split-K factors per shape → more cross-batch variation.

**Caveat (from Sol review):** This is a plausible hypothesis but not fully proven. H100 FP32 may actually use TF32 Tensor Core (not pure CUDA core). To prove definitively, need to:
1. Query cuBLASLt/rocBLAS algorithm ID per shape
2. Disassemble kernels (SASS/ISA) for HMMA/WGMMA/V_MFMA instructions
3. Force a single algorithm and verify cross-batch diff disappears

### Batch invariance solution

**Manual loop (fixed algorithm) achieves 0.0000% diff on both GPUs.**

Running B separate `torch.mm` calls (always same shape) forces the library to use the same algorithm every time → bit-exact.

This is a proof of concept, not a production solution — B separate calls are B× slower than one `torch.bmm`. A real batch-invariant kernel would use a single fixed-tiling batched kernel.

Existing solutions:
- **TML `batch_invariant_ops`**: NVIDIA-only, uses `torch.Library` to swap kernels
- **vLLM `VLLM_BATCH_INVARIANT=1`**: loads TML library, NVIDIA-only
- **AMD**: no solution exists. rocBLAS has `rocblas_atomics_not_allowed=1` but this only disables atomics, doesn't fix tiling switches

## FP16/BF16 GEMM Internal Pipeline

All frameworks use **mixed precision** for FP16/BF16 GEMM:

```
Input:  A [M, K] — FP16/BF16
        B [K, N] — FP16/BF16
Step 1: Load tiles to shared memory (FP16/BF16)
Step 2: Multiply A × B → FP32 accumulator (hardware register)
Step 3: Accumulate across K tiles in FP32
Step 4: Cast FP32 → FP16/BF16 → write output
Output: C [M, N] — FP16/BF16
```

Nobody uses FP16 as accumulator — FP16 max is 65504, K=4096 accumulation overflows.

### FP16 vs BF16 tradeoffs

| | FP16 | BF16 |
|---|---|---|
| Exponent | 5 bit (range ±65504) | 8 bit (range ±3.4×10³⁸, same as FP32) |
| Mantissa | 10 bit (~3.7 decimal digits) | 7 bit (~2.4 decimal digits) |
| Training | Risk of overflow/underflow | Safe (same range as FP32) |
| Inference | OK if values are bounded | Safer for large activations |
| Cross-batch diff | Smaller (more mantissa bits) | Larger (fewer mantissa bits) |

BF16 has larger cross-batch diff because 7-bit mantissa means each algorithm switch causes larger relative rounding error.

### FP8 exception (from Sol review)

FP8 GEMM on H100 can use FP16 accumulator (not FP32) for "fast accumulation" mode. This is the one case where sub-FP32 accumulation is used in practice. Internal partial accumulator may be even narrower. This trades accuracy for ~2× throughput.

## Other Op Determinism

### Softmax
- Run-to-run: ✅ stable
- Cross-impl: ❌ not bit-exact (naive vs online vs torch) — reduction order differs
- Batch=1 FP32: ✅ bit-exact (no parallel reduction divergence)

### Reduction (sum)
- Run-to-run: ✅ stable
- Cross-impl: ❌ FP16 diff = 14.5 (sequential accumulation overflow, not run-to-run)
- FP16 naive sum of 100K N(0,1) values: accumulation rounding error ~14.5
- `torch.sum` internally upcasts to FP32 → bit-exact with `torch_stable`

### RMSNorm
- Run-to-run: ✅ stable
- Cross-impl (naive vs fused): ✅ bit-exact (same reduction order via `.mean()`)
- Cross-impl (fp32_upcast on FP16 input): ❌ diff — FP32 reduction vs FP16 reduction

### LayerNorm
- Run-to-run: ✅ stable
- Cross-impl (naive vs torch): ❌ diff — cuDNN uses different algorithm
- Larger diff than RMSNorm because mean + variance has more reduction steps

### GEMM (mm/matmul/addmm)
- Run-to-run: ✅ stable
- Cross-impl: ✅ bit-exact — all three dispatch to same cuBLAS/rocBLAS kernel

## Hardware Details

### NVIDIA H100
- CUDA 12.6, PyTorch 2.7.1
- Tensor Core tile: 16×16×16 (wmma)
- TF32: 19-bit mantissa (vs FP32 23-bit)
- cuBLAS `GemmStridedBatched`: fixed kernel for all B (FP16/BF16)

### AMD MI325X
- ROCm 7.14, PyTorch 2.13.0
- MFMA tile: 16×16×16 and 32×32×16
- No TF32 equivalent
- rocBLAS `gemmStridedBatched`: different kernels for different B (all dtypes)

## Tools & References

- [TML batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — NVIDIA-only batch invariance library
- [vLLM Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) — `VLLM_BATCH_INVARIANT=1`
- [Differentiated: Sliding-Window Prefix Caches Need Physical Proof](https://www.differentiated.io/blog/sliding-window-prefix-caches-need-physical-proof) — cache hit correctness
- SGLang RFC #34562 — SWA receptive-field pyramid (separate topic, see SWA notes)

## Test Scripts

All test scripts in `kernels/determinism/`:
- `run_tests.py` — v1 single-run determinism test (5 ops, 3 dtypes)
- `run_tests_v2.py` — v2 multi-run pairwise (200 iterations)
- `cross_batch_gemm.py` — cross-batch GEMM (mm, vary M)
- `cross_batch_gemm_v2.py` — multi-run cross-batch GEMM
- `batched_gemm.py` — batched GEMM (bmm, vary B)
- `fixed_algo_gemm.py` — 4 approaches to achieve batch invariance
