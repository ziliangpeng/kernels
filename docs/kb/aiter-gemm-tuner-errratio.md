# AITER GEMM Tuner: What errRatio Actually Measures

**Source**: ROCm/aiter `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`, `aiter/utility/mp_tuner.py`, `aiter/test_common.py::checkAllclose` (main, 2026-09-02)
**Context**: gemma-4-31b FP8-block tuning sweep (PR ROCm/aiter#5062), reviewed live in the Sep 02 CAI×AMD weekly sync.

## The pipeline

```
tuner tries candidate kernel configs for a given (M, N, K):
  1. reference = run_torch(): dequant FP8 (×block scales) → cast FP32 → F.linear
  2. run candidate kernel on same FP8 inputs
  3. torch.isclose(candidate, ref, rtol=1e-2, atol=1e-2)
  4. errRatio = fraction of elements failing isclose
  5. accept only if errRatio ≤ tol_err_ratio (default 0.05)
```

## Key facts (read from source, not inferred)

- **errRatio is NOT max relative error.** It is the *fraction of elements* failing
  `rtol=1e-2 / atol=1e-2`. Per-element tolerance and population-level fraction are
  two separate gates, both must pass.
- **The reference is FP32 matmul of dequantized FP8 inputs.** Not "another kernel",
  not a majority vote. `run_torch` does `x.to(fp32)` *after* dequant, so reference
  error (~1e-5..1e-4) sits an order of magnitude below the 1e-2 bar → it is
  qualified to arbitrate.
- **All candidates eat identical FP8 inputs**, so input-quantization error cancels
  in the comparison. What differs between candidates is accumulation order +
  intermediate precision only.
- **`_check_catastrophic`** exists separately: a few elements with huge abs delta
  fail the whole candidate even at low errRatio. Guards against "1% of elements
  but they're wildly wrong".
- SplitK changes chunk boundaries → changes partial-sum rounding order → shifts
  the error distribution. Not "wrong math", just a different point in the cloud
  around the exact value. Observed: splitK=3 gave errRatio=0.0242, splitK=2 gave
  0.0078 at ~5% speed cost (675 vs 642 µs, M=1024 N=5376 K=16384) — tuner chose
  the safer one.

## Why 2.4% elements >1% off, and why that's OK

- FP8 e4m3 inputs carry ~3 mantissa bits; the *cast* error dominates (matches my
  earlier measurement: FP8 Frobenius rel err ≈ 3.75%, format-decided, M/K-independent —
  see fp8-gemm-accuracy-scaling-modes.md).
- errRatio here measures the **kernel's extra deviation on top of that**, vs the
  FP32-exact evaluation of the same FP8 inputs. A 2.4% tail is a rounding-order
  signature, not a correctness failure.
- End-to-end gsm8k was flat (±0.02) across all accepted configs.

## The CSV row = lookup + audit trail

```
gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
gfx942,304,1,5376,16384,ck,6,3,33.85,a8w8_blockscale_1x128x128_256x16x64x128_...,5.2,2603.02,0.0032
```

- Runtime only consumes `shape → (kernelId, splitK, libtype)`.
- `us` is functionally needed by the re-tune flow: `min_improvement_pct=3.0` means
  a new candidate must beat the recorded µs by ≥3% to overwrite. Retuning is
  incremental because the old measurements are in the file.
- `tflops/bw` are derived (`calculate()`) — reviewer-facing: M=1 row shows 5.2
  TFLOPS / 2.6TB/s → instantly readable as weight-load-bound.
- `errRatio` is the correctness audit column. Reviewer (yzhou103) challenged the
  0.0242 entry; re-tuned to splitK=2. **Lesson: the tuner default gate (5%) is
  looser than what the reviewer would accept (~1%). Model owner should set the
  errRatio budget, not inherit the tuner default.**

## Related

- [fp8-gemm-accuracy-scaling-modes.md](fp8-gemm-accuracy-scaling-modes.md) — FP8 cast error floor (3.75%), the 1% rtol bar sits inside that world
- [gemm-determinism.md](gemm-determinism.md) — the DSV4 fused-MoE case: same inputs, different accumulation order → 100% drift
- PRs: ROCm/aiter#5062 (tuning sweep), vllm#53273/#53874/#53918 (fusion, same Gemma4 effort)
