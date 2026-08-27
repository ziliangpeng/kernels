# Op Determinism (Non-GEMM)

Tested on: H100 (CUDA 12.6, PyTorch 2.7.1) + MI325X (ROCm 7.14, PyTorch 2.13)
Date: 2026-08-27

## Summary

| Op | Run-to-run | Cross-impl | Root cause of cross-impl diff |
|---|---|---|---|
| Softmax (batch>1) | ✅ stable | ❌ not exact (FP32) | reduction order differs |
| Softmax (batch=1) | ✅ stable | ✅ bit-exact (FP32) | see note below |
| Reduction (sum) | ✅ stable | ❌ not exact | sequential vs parallel tree reduction |
| GEMM (mm/matmul/addmm) | ✅ stable | ✅ bit-exact | same underlying library call (not proven same kernel) |
| RMSNorm (naive/fused) | ✅ stable | ✅ bit-exact | same PyTorch expression (not independent kernel impls) |
| RMSNorm (fp32_upcast) | ✅ stable | ❌ not exact (FP16) | FP32 reduction vs native FP16 reduction |
| LayerNorm (naive/torch) | ✅ stable | ❌ not exact | different backend algorithm (not confirmed cuDNN) |

**Note (Sol review):** Bit-exactness is verified via checksum. "0.0000%" in percentage formatting does NOT prove bit-exact — must use `torch.equal` or checksum comparison.

## Softmax

### Implementations tested
- **Naive**: two-pass (max → exp → sum → normalize)
- **Online**: single-pass rescaling (max update + rescale)
- **Torch**: `F.softmax` (backend not confirmed — could be cuDNN, FlashInfer, or native)

### Findings
- All implementations are run-to-run stable
- Naive vs online: not bit-exact (different reduction order in the sum step)
- Naive vs torch: not bit-exact (different internal algorithm)
- **Batch=1 FP32**: all three bit-exact

**Note (Sol review):** Batch=1 does NOT mean "no parallel reduction." The reduction along dim=4096 is still parallelized across threads. The bit-exactness at batch=1 is likely because all three implementations happen to use the same reduction topology for a single row, not because there is no parallelism.

## Reduction (sum)

### Implementations tested
- **Naive**: sequential loop `for i in range(N): result += x[i]`
- **Torch**: `x.sum()` (parallel tree reduction, internally upcasts FP16→FP32)
- **Torch stable**: `x.float().sum().to(x.dtype)` (explicit upcast)

### Findings
- FP32: naive vs torch diff = 7.93e-4 (sequential vs tree reduction order)
- FP16: naive vs torch diff = **14.5** — this is **accumulation rounding error**, NOT overflow. 100K N(0,1) values have partial sums well below 65504. The error comes from FP16 ULP at large partial sums (~5000) being ~0.5, with 4096+ additions accumulating error.
- `torch.sum` internally upcasts to FP32 → bit-exact with `torch_stable`

## RMSNorm

### Implementations tested
- **Naive**: `(x ** 2).mean()` → `x * rsqrt(ms + eps) * weight`
- **Fused**: `x.pow(2).mean()` → `x * rsqrt(var + eps) * weight`
- **FP32 upcast**: cast to FP32, compute, cast back

### Findings
- Naive vs fused: bit-exact — **but these are essentially the same PyTorch expression**, not two independent kernel implementations. This is not a strong test of cross-kernel determinism.
- FP32 upcast on FP16 input: not bit-exact (FP32 reduction has higher precision than native FP16)

## LayerNorm

### Implementations tested
- **Naive**: manual mean + variance + normalize
- **Torch**: `F.layer_norm` (backend not confirmed — described as "cuDNN" but not profiled)

### Findings
- Not bit-exact: `F.layer_norm` uses a different algorithm with different reduction order
- Larger diff than RMSNorm (mean + variance = 2 reduction steps vs RMSNorm's 1)
- FP16 diff (1.56e-2) >> FP32 diff (1.91e-6) — fewer mantissa bits amplify rounding

**Note (Sol review):** Do not claim "cuDNN algorithm" without profiler evidence. `F.layer_norm` may dispatch to cuDNN, a native PyTorch kernel, or another backend depending on PyTorch version and configuration.

## Initialization

All tests use `torch.randn` (standard normal, N(0,1)) with fixed seed via `torch.Generator`. Same seed → same input tensor every run.
