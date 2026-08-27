# FP8 and FP4 Training

## FP8 training

FP8 = 8-bit floating point. Two common formats:

| Format | Exponent | Mantissa | Range | Use case |
|---|---|---|---|---|
| E4M3 | 4 bit | 3 bit | ±448 | Forward pass (weights, activations) |
| E5M2 | 5 bit | 2 bit | ±57344 | Backward pass (gradients, larger range) |

### How FP8 training works

FP8 training is NOT "everything in FP8." It's **mixed precision with FP8 compute + FP32 master**:

```
Master weights:    FP32 (same as BF16 training)
Forward:           FP32 weight → cast to FP8 (E4M3) → FP8 GEMM → FP32 accumulator
Backward:          FP32 grad → cast to FP8 (E5M2) → FP8 GEMM → FP32 accumulator
Optimizer update:  FP32 (same as BF16 training)
```

### Key difference from BF16 training

- **BF16 training**: cast weight to BF16, keep it for the whole forward/backward
- **FP8 training**: cast weight to FP8 **per-tile** during GEMM, with **dynamic scaling factors** that adjust per tensor or per tile

The scaling factor is critical because FP8 has very limited range (E4M3 max = 448). Without scaling, activations easily overflow. The framework (Transformer Engine, TorchAO) maintains a running max statistic and computes a scale factor to map the tensor's dynamic range into FP8's tiny range.

### Why two formats?

- **E4M3** (forward): more mantissa bits (3) → better precision for weights/activations. Range ±448 is enough because values are bounded.
- **E5M2** (backward): more exponent bits (5) → larger range ±57344 for gradients, which can be large or small. Less precision (2 mantissa) is OK because gradients are noisy anyway.

### What FP8 actually saves

- **2× memory** for weights and activations (8 bit vs 16 bit)
- **2× compute throughput** on H100 (FP8 Tensor Core is 2× BF16 rate)
- **2× memory bandwidth** (half the bytes per element)

### FP32 master weights still mandatory

Same reason as BF16 training: FP8 ULP at w~1.0 is ~0.125 (E4M3) — way too coarse for weight updates. FP32 master is still the source of truth.

## NVFP4 training (NVIDIA, experimental)

NVFP4 = 4-bit floating point with **microscaling**. Announced with Blackwell (B200).

### Format

NVFP4 is not a single 4-bit format. It's **block-scaled FP4**:

```
Each block of 32 elements shares:
  - One FP8 scale factor
  - 32 × 4-bit values (1 sign + 2 exponent + 1 mantissa = E2M1)
```

The per-block scale factor compensates for FP4's tiny range. Without it, FP4 is useless. With it, the effective dynamic range is much larger.

### Why 4-bit?

- **4× memory savings** vs FP16/BF16
- **4× compute throughput** on Blackwell (FP4 Tensor Core)
- Training a 1T parameter model in BF16 needs ~2TB just for weights. FP4 needs ~500GB.

### Does NVFP4 training work?

As of 2026, NVIDIA claims NVFP4 training works for large models with:
- FP32 master weights (still mandatory)
- Per-block microscaling
- Careful loss scaling
- Stochastic rounding (probabilistic rounding to avoid systematic bias)

But it's **very experimental**. Known issues:
- Very sensitive to scaling factor quality
- Small models may not converge (not enough averaging to absorb noise)
- Only works on Blackwell hardware
- Most frameworks don't support it yet

### The fundamental question

Can you train a model with 4-bit weights when each weight only has 2 mantissa bits (E2M1)?

The bet: **yes, if the model is big enough.** Large models have redundancy — individual weights don't matter much, the ensemble does. 4-bit noise is absorbed by the model's overparameterization, same way BF16 noise is absorbed by SGD.

This is unproven at scale. NVIDIA is betting the company on it.

## Comparison

| | BF16 | FP8 | NVFP4 |
|---|---|---|---|
| Bits per element | 16 | 8 | 4 (+ per-block scale) |
| Mantissa bits | 7 | 2-3 | 1 |
| Master weights | FP32 | FP32 | FP32 |
| Scaling | None | Per-tensor or per-tile dynamic | Per-block microscaling (FP8 scale) |
| Hardware | All modern GPUs | H100+ | Blackwell+ |
| Maturity | Production | Early production | Experimental |
| Memory savings vs FP32 | 2× | 4× | 8× |
| Compute speedup vs FP32 | 2× | 4× | 8×+ |

## The pattern

Every precision reduction follows the same pattern:
1. **Keep FP32 master weights** (non-negotiable for weight updates)
2. **Cast to lower precision for compute** (forward/backward GEMM)
3. **Add scaling to fit dynamic range** (more aggressive scaling needed at lower precision)
4. **Rely on SGD noise / model overparameterization to absorb precision loss**
5. **Verify convergence empirically** (lower precision = more noise = may not converge on small models)

The lower the precision, the more the training relies on:
- Large model size (redundancy absorbs noise)
- Good scaling factors (maximize dynamic range usage)
- Stochastic rounding (avoid systematic bias)
- FP32 master weights (prevent update vanishing)
