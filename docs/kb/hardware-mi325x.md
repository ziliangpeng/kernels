# AMD MI325X

## Specs
- GPU: AMD Radeon Graphics (MI325X)
- ROCm: 7.14 (HIP 7.14.60850)
- PyTorch: 2.13.0+rocm7.14.0
- 256GB HBM3e (per node, 8 GPUs)

## Matrix Core (MFMA)
- Supports multiple tile sizes: 16×16×16, 32×32×16
- FP16/BF16 input → FP32 accumulator (mixed precision, same as Tensor Core)
- Instruction: `__mfma_f32_16x16x16_f16`
- More tile size options than Tensor Core → larger algorithm space → more cross-batch switching

## rocBLAS
- Standard GEMM: `rocblas_gemm_ex`
- Batched GEMM: `rocblas_gemm_strided_batched_ex` — does NOT use fixed kernel for all B (unlike cuBLAS)
- hipBLASLt: heuristic algorithm selection, equivalent to cuBLASLt
- `rocblas_atomics_not_allowed=1`: disables atomic operations (partial determinism, does NOT fix tiling switches)
- `ROCBLAS_DEFAULT_ATOMICS_MODE=0`: alternative env var for same effect

## Algorithm selection
- rocBLAS selects algorithm based on B, M, N, K, dtype
- More aggressive switching than cuBLAS — different B values trigger different kernels even for bmm
- Algorithm switching thresholds (our measurements):
  - FP32: switches at nearly every M
  - FP16: switches at M=128
  - BF16: switches at M=256
- Cross-vendor bit-exactness with H100: does not exist (different tiling, different internal rounding)

## Key differences vs H100
| | H100 (cuBLAS) | MI325X (rocBLAS) |
|---|---|---|
| bmm FP16 cross-batch | ✅ bit-exact | ❌ 0.0434% |
| bmm BF16 cross-batch | ✅ bit-exact | ❌ 0.17~0.35% |
| bmm FP32 cross-batch | ❌ 0.0003% | ❌ 0.0003% |
| Batch invariance solution | TML batch_invariant_ops | None |
| TF32 | Yes (amplifies diff) | No equivalent |

## No batch invariance solution
- No equivalent to TML `batch_invariant_ops` exists for ROCm
- vLLM `VLLM_BATCH_INVARIANT=1` only works on NVIDIA
- Writing an AMD version is technically straightforward (HIP kernel with fixed tiling) but nobody has done it
