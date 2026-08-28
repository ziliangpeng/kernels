# FP8 GEMM: Accuracy, Scaling Modes, and Real-World Impact

**Experiment dates**: 2026-08-27/28 · **GPUs**: NVIDIA H100 80GB (gcp5), AMD MI325X (amd2 dev pod)
**Scripts**: `determinism/fp8_gemm_test.py`, `determinism/fp8_scaling_compare.py`, `determinism/fp8_blockwise_aiter.py`, `determinism/fp8_deepgemm_test.py`
**Results**: `determinism/results_fp8_*.json`

## The headline numbers (H100 + MI325X, identical on both)

| Format | Frobenius rel err vs FP32 | mantissa bits | Ratio vs FP16 |
|---|---|---|---|
| FP16 | 0.036% | 10 | 1× |
| BF16 | 0.288% | 7 | 8× |
| **FP8 e4m3** | **3.75%** | 3 | **104×** |

- Error is **identical to 3 decimals on H100 and MI325X** — accuracy is decided by
  format + library behavior (FP32 accumulate), not vendor.
- Error is **independent of M and K** (M=1 decode to M=4096 prefill, K=4096 to
  16384). The intuition "big K accumulates error" is wrong: cast rounding of the
  inputs dominates; the library's internal FP32 accumulation error is negligible.
- Blockwise (1×128 act + 128×128 weight) on random N(0,1) input: 3.75% → 3.68%,
  only ~2% improvement. Normal distribution has no outliers, so fine-grained
  scales have nothing to isolate. (See "real distributions" below.)

## FP8 speed (hardware confirmed via 2× vs FP16)

| GPU | FP16 | FP8 | ratio |
|---|---|---|---|
| H100 | 759 TFLOPS | 1,444 TFLOPS | 1.9× |
| MI325X | 553 TFLOPS | 1,149 TFLOPS | 2.08× |

The 2× speedup proves hardware matrix unit execution (Tensor Core / MFMA),
not CUDA-core emulation.

## Scaling modes — full support matrix (all tested)

| Mode | Block | Scale dtype | H100 (torch 2.7/2.8+cu12.9) | MI325X (torch 2.13+rocm7.14) |
|---|---|---|---|---|
| **TensorWise** | whole tensor | FP32 | ✅ | ✅ |
| **RowWise** | per row/col | FP32 | ✅ | ✅ |
| **BlockWise 1×128** | 128 elems | FP32 | ❌ PyTorch `_scaled_mm` dispatch not wired | ❌ hipBLASLt "requires CUDA 12.9" |
| **BlockWise 128×128** | 128² tile | FP32 | ❌ same | ❌ same |
| **MXFP8 (1×32)** | 32 elems | **e8m0** | ❌ cuBLASLt sm90 has no heuristic (Blackwell sm100 only) | ❌ hardware gfx950 only |
| **NVFP4 (1×16)** | 16 elems | e4m3 | ❌ Blackwell only | ❌ gfx950 only |
| **Blockwise via aiter `gemm_a8w8_blockscale`** | 1×128/128×128 | FP32 | N/A | ✅ **works**, fro=3.68% |
| **Blockwise via DeepGEMM** | 1×128/128×128 | FP32 | ✅ **works**, fro=3.70% | N/A |

### Why `_scaled_mm` cannot do FP32 blockwise (the debugging trail)

1. **PyTorch dispatch gap**: the C++ `ScaledBlas.cpp` has
   `is_blockwise_1x128_scaling` checks, but the Python/meta registration
   (`_meta_registrations.py`) only recognizes TensorWise/RowWise/e8m0-blockwise.
   Any FP32 1×128 scale tuple is rejected at the meta gate before reaching
   cuBLASLt. Brute-forcing scale shapes/strides cannot fix this.
2. **MI325X hipBLASLt**: error literally says blockwise "requires CUDA 12.9
   and above" — the ROCm library maps this feature to a newer ABI.
3. **MXFP8 on sm90**: cuBLASLt heuristic for e8m0 block scaling exists only on
   Blackwell (sm100). H100 sm90 gets `CUBLAS_STATUS_NOT_SUPPORTED` at
   `cublasLtMatmulAlgoGetHeuristic`. This is hardware, not software.

**Conclusion: on both platforms, FP32-scale blockwise requires the dedicated
kernels (DeepGEMM on NVIDIA, aiter on AMD) — the built-in `torch._scaled_mm`
path is not usable today (2026-08).**

## FP8 dtype quirks (recorded for future debugging)

| | NVIDIA (H100) | AMD (MI325X) |
|---|---|---|
| FP8 dtype | `torch.float8_e4m3fn` (bias-127) | `torch.float8_e4m3fnuz` (bias-128, no −0) |
| Tensor core instruction | `mma.async.aligned.m16n8k32.e4m3` | `v_mfma_f32_32x32x8_fp8` |
| `_scaled_mm` scales | scalar 1-elem tensor OK | rowwise scale_a (M,1), scale_b (1,N), contiguous |
| `_scaled_mm` dims | 2D only (torch 2.7/2.8); 3D batched from torch 2.13 | 2D only (torch 2.13 ROCm) |
| cuBLASLt/hipBLASLt FP32 blockwise | cublasLt ≥ 12.9 | "requires CUDA 12.9" equivalent |

## DeepGEMM on H100 — working recipe

The 2026-08 DeepGEMM (2.6.1) needs:

```bash
pip install nvidia-cutlass   # provides cutlass_library + cute headers
# JIT nvcc must find cute/ headers — set this or kernel compilation fails:
export CPLUS_INCLUDE_PATH=/opt/conda/lib/python3.11/site-packages/cutlass_library/source/include:$CPLUS_INCLUDE_PATH
pip install -e /path/to/DeepGEMM --no-build-isolation
```

Pitfalls:
- Stale `.so` from a failed first build lives in
  `site-packages/deep_gemm/` (not the repo's `build/`) — must delete before
  rebuild, otherwise the old ABI mismatch (`SymInt::maybe_as_int_slow_path`)
  keeps loading.
- `per_token_cast_to_fp8(x, use_ue8m0=False)` and
  `per_block_cast_to_fp8(x, use_ue8m0=False)` require the explicit flag.
- JIT kernel compile happens on first GEMM call, not at import.

Usage (DeepSeek recipe):

```python
A_q, A_s = deep_gemm.per_token_cast_to_fp8(A, use_ue8m0=False)   # 1×128 blocks
W_q, W_s = deep_gemm.per_block_cast_to_fp8(W, use_ue8m0=False)   # 128×128 blocks
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
deep_gemm.fp8_gemm_nt((A_q, A_s), (W_q, W_s), out)
```

## aiter on MI325X — working recipe

```bash
# NOT the PyPI "aiter" (that's an async-iterator lib). Install from GitHub:
pip install "git+https://github.com/ROCm/aiter.git@main" --no-deps
pip install pybind11   # --no-deps skips it; JIT build needs it
```

- Requires `float8_e4m3fnuz` + `get_hip_quant(QuantType.per_1x128)` for
  activations (Triton path; the HIP path rejects fp32 scales for group quant).
- Weights: manual per-128×128 block quant, then
  `aiter.gemm_a8w8_blockscale(A8, W8, s_a, s_w, dtype=aiter.dtypes.bf16)`.
- First JIT build (`module_aiter_core`, `module_quant`, `module_gemm_a8w8_blockscale`)
  takes ~2 min total; subsequent runs are fast.

## Is 4% per-GEMM error a problem? (community consensus)

**No — and yes, the number is well known.** The reference study is
*"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM
Quantization* (Neural Magic / Red Hat, 2024-11, 500K+ evaluations on
Llama-3.1 8B/70B/405B):

- **W8A8-FP8 is "effectively lossless"** across all model scales — task
  accuracy within evaluation margin of error (e.g. MMLU 74.06 → 73.55,
  ~99.3% recovery; 405B sometimes scores *higher* than BF16).
- Well-tuned **INT8 W8A8** loses 1–3% per task; **INT4 W4A16** is competitive
  with 8-bit.

Why 3.75% per-GEMM noise does not hurt task accuracy:

1. The per-element error is **unstructured noise**, not systematic signal
   distortion. Task metrics are robust to logit noise (temperature sampling
   deliberately adds similar-magnitude noise).
2. **No exponential compounding across layers** — residual stream, LayerNorm,
   and attention softmax operate in BF16/FP32; each layer's FP8 GEMM noise is
   added, not multiplied.
3. Training tolerates it too: DeepSeek-V3 pretrained 671B in FP8 with the
   1×128/128×128 recipe; loss curve matches BF16 baseline.

Where FP8 genuinely struggles (also community consensus):
- **Training with outliers**: TWEO (CVPR 2026) documents per-tensor FP8
  training collapse from 10,000+ extreme outliers; blockwise 1×128 (DeepSeek
  recipe) or outlier suppression is required. MXFP8 (1×32 e8m0) is the
  Blackwell-generation answer.
- **KV cache quantization**: attention is more sensitive; usually e5m2 or kept
  at BF16.
- **Small models**: error compounding is worse below ~8B.

## Blockwise quantization history (context for "who invented what")

- Per-block/per-group quant is old (GPTQ/AWQ group-128, 2022-23).
- OCP **Microscaling (MX)** standard — 1×32 block, e8m0 scale — defined 2023
  by NVIDIA/AMD/µsoft consortium.
- **DeepSeek's contribution** (V3 technical report, 2024-12): first
  production FP8 *pretraining* of a 671B model with the 1×128 activation /
  128×128 weight FP32-scale recipe, plus the open-source DeepGEMM kernel
  library. This made fine-grained FP8 the de-facto standard — NVIDIA TE 2.18
  copied the recipe verbatim, vLLM/SGLang expose it as `fp8_per_block`.
- DeepGEMM exists because cuBLASLt FP32-blockwise support arrived late
  (12.9+) and PyTorch's dispatch is still not wired; aiter's CK kernels are
  the AMD counterpart for the same reason.

## Random-input caveat (methodology note)

All accuracy numbers here use N(0,1) random inputs. Blockwise scaling's real
advantage appears with real model activations, which contain outlier channels
(magnitude 10–100× typical, ~0.1% of channels — LLM.int8(), SmoothQuant,
TWEO). With outliers, per-tensor scale gets pulled up by the outlier and the
rest of the tensor loses effective precision; per-block scales isolate the
damage. Measuring that effect requires real weights/activations — future
experiment.

## Links

- Companion: `gemm-fp32-vs-lowprec-accuracy.md` (FP16/BF16 vs FP32, the 8× ratio)
- Companion: `gemm-determinism.md` (FP8 run-to-run BIT-EXACT 100/100 on both GPUs; batch invariance holds except fold-path drift ~0.3-0.5%)
- Pipeline explanation: `fp8-gemm-hardware.md`, `fp8-gemm-chaining.md`
- Hardware details: `hardware-h100.md`, `hardware-mi325x.md`
- Test scripts: `determinism/fp8_gemm_test.py`, `determinism/fp8_scaling_compare.py`,
  `determinism/fp8_blockwise_aiter.py`, `determinism/fp8_deepgemm_test.py`
