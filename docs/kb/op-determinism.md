# Op Determinism (Non-GEMM)

Tested on: H100 (CUDA 12.6, PyTorch 2.7.1) + MI325X (ROCm 7.14, PyTorch 2.13)
Date: 2026-08-27

## Summary

| Op | Run-to-run | Cross-impl | Root cause of cross-impl diff |
|---|---|---|---|
| Softmax (batch>1) | ✅ stable | ❌ 1.86e-9 (FP32) | reduction order (naive vs online vs cuDNN) |
| Softmax (batch=1) | ✅ stable | ✅ bit-exact (FP32) | no parallel reduction divergence with 1 row |
| Reduction (sum) | ✅ stable | ❌ 7.93e-4 (FP32), 14.5 (FP16) | sequential vs parallel tree reduction |
| GEMM (mm/matmul/addmm) | ✅ stable | ✅ bit-exact | same cuBLAS/rocBLAS kernel |
| RMSNorm (naive/fused) | ✅ stable | ✅ bit-exact | same reduction order (.mean()) |
| RMSNorm (fp32_upcast) | ✅ stable | ❌ 7.81e-3 (FP16) | FP32 reduction vs FP16 reduction |
| LayerNorm (naive/torch) | ✅ stable | ❌ 1.91e-6 (FP32) | cuDNN different algorithm |

## Softmax

### Implementations tested
- **Naive**: two-pass (max → exp → sum → normalize)
- **Online**: single-pass rescaling (max update + rescale)
- **Torch**: `F.softmax` (cuDNN/FlashInfer backend)

### Findings
- All implementations are run-to-run stable
- Naive vs online: not bit-exact (different reduction order in the sum step)
- Naive vs torch: not bit-exact (cuDNN uses warp-level parallel reduction)
- **Batch=1 FP32**: all three bit-exact (single row → no cross-row parallel divergence)

## Reduction (sum)

### Implementations tested
- **Naive**: sequential loop `for i in range(N): result += x[i]`
- **Torch**: `x.sum()` (parallel tree reduction)
- **Torch stable**: `x.float().sum().to(x.dtype)` (upcast to FP32)

### Findings
- FP32: naive vs torch diff = 7.93e-4 (sequential vs tree reduction order)
- FP16: naive vs torch diff = **14.5** — not overflow but accumulation rounding. 100K FP16 additions, ULP at sum~5000 is ~0.5, errors accumulate to ~14.5
- `torch.sum` internally upcasts to FP32 → bit-exact with `torch_stable`

### Key insight
FP16 reduction error is O(N) for sequential, O(log N) for tree. With N=100K:
- Sequential: ~14.5 error
- Tree (torch): ~0 error (FP32 accumulator internally)

## RMSNorm

### Implementations tested
- **Naive**: `(x ** 2).mean()` → `x * rsqrt(ms + eps) * weight`
- **Fused**: `x.pow(2).mean()` → `x * rsqrt(var + eps) * weight`
- **FP32 upcast**: cast to FP32, compute, cast back

### Findings
- Naive vs fused: bit-exact (both use `.mean()` → same reduction order)
- FP32 upcast on FP16 input: not bit-exact (FP32 reduction has higher precision than native FP16)

## LayerNorm

### Implementations tested
- **Naive**: manual mean + variance + normalize
- **Torch**: `F.layer_norm` (cuDNN backend)

### Findings
- Not bit-exact: cuDNN uses different algorithm with different reduction order
- Larger diff than RMSNorm (mean + variance = 2 reduction steps vs RMSNorm's 1)
- FP16 diff (1.56e-2) >> FP32 diff (1.91e-6) — fewer mantissa bits amplify rounding

## Initialization

All tests use `torch.randn` (standard normal, N(0,1)) with fixed seed via `torch.Generator`. Same seed → same input tensor every run.
