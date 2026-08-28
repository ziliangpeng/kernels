# FP8 GEMM Chaining Between Layers

## The question: can we FP8 all the way?

After an FP8 GEMM, the Tensor Core outputs FP32. You can:
- Cast to BF16 (safe, standard)
- Cast to FP8 + new scale factor (saves memory, risks accuracy)

If casting to FP8, the FP8 output matrix + scale factor becomes the input to the next GEMM layer. This is "FP8 chaining."

## Full pipeline with FP8 chaining

```
Layer N GEMM:
  A_fp8 + scale_A  ──┐
  B_fp8 + scale_B  ──┤
                      ↓
  Tensor Core: FP8 × FP8 → FP32 accumulate
                      ↓
  De-scale: FP32 × scale_A × scale_B → true FP32
                      ↓
  Compute new scale: new_scale = max(|result|) / 448
                      ↓
  Re-scale + cast: result_fp8 = cast(result / new_scale)
                      ↓
  A_fp8 + new_scale  ──→ Layer N+1 GEMM input
```

## Epilogue fusion optimization

If output scale is known ahead of time, the kernel can fuse de-scale + re-scale into one epilogue multiplier — no need to materialize the FP32 matrix:

```
// Without fusion (slow):
FP32 accumulator → de-scale → materialize FP32 → compute max → re-scale → cast FP8 → write HBM

// With fusion (fast):
FP32 accumulator → combined_scale = (scale_A × scale_B) / new_output_scale → cast FP8 → write HBM
```

The challenge: computing `new_output_scale` requires knowing the output's max value, which requires a reduction pass. Solutions:
- **Delayed scaling**: use previous step's scale (fast, but may be stale)
- **Current scaling**: compute scale in real-time (accurate, but extra reduction)
- **Static scaling**: pre-computed from calibration (fast, but inflexible)

## What frameworks actually do

Most frameworks do NOT chain FP8 activations between layers. They keep activations in BF16 and only quantize to FP8 at the GEMM entrance:

```
Weight:     FP8 (kept across layers — saves memory)
Activation: BF16 (between layers — preserves precision)
GEMM:       BF16 activation → on-the-fly cast to FP8 → FP8 MMA → FP32 → cast back to BF16
```

This is called **W8A8 mixed precision** — weight 8-bit, activation 8-bit only inside GEMM.

| Framework | Weight | Activation between layers | FP8 chaining? |
|---|---|---|---|
| Transformer Engine | FP8 | BF16 (default) | Default no. Supports `fp8_output=True` for fused paths |
| vLLM | FP8 | BF16 | No — Linear activations are BF16 |

## Why not chain FP8 activations

Sol review identified these risks:

1. **Residual stream accumulation**: FP8 quantization error compounds across many layers through the residual connection
2. **Sensitive operations**: LayerNorm, softmax, router logits are sensitive to small numerical differences
3. **Outlier dominance**: per-tensor scale is dominated by a few large values, reducing effective precision for normal values
4. **Delayed scaling lag**: activation distribution can shift suddenly, making previous step's scale stale
5. **Clipping vs waste**: scale too tight → clipping; scale too loose → wasted FP8 range

## Exception: back-to-back GEMM

If two GEMMs are directly back-to-back with no operation between them, direct FP8 chaining may be better — saves one BF16 materialization, and numerically no worse (both paths require one FP8 quantization). But real Transformers rarely have this pattern.

## Accuracy comparison

| Approach | Activation between layers | Memory | Accuracy |
|---|---|---|---|
| BF16 activation (mainstream) | BF16 | More | Robust |
| FP8 chaining | FP8 + scale | Saves 2× | Risk: error accumulation, outlier sensitivity |
| FP8 epilogue fusion | FP8 + scale | Saves 2× + no FP32 materialize | Same as FP8 chaining |

## Bottom line

FP8 chaining is technically possible but requires careful calibration and model-level accuracy validation. For most production use cases, BF16 activations between layers is the safe default. FP8 chaining is worth considering only for fused back-to-back GEMMs or when memory is the binding constraint.

Sources: NVIDIA Transformer Engine docs (current scaling, delayed scaling), vLLM FP8 W8A8 docs, Sol review.
