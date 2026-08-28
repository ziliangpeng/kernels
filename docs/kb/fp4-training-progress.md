# FP4 Training Progress and Status

Last updated: 2026-08-27

## What is NVFP4?

NVFP4 = 4-bit floating point with **microscaling** (block-scaled). Developed by NVIDIA for Blackwell GPUs.

Format: each block of 32 values shares one FP8 scale factor + 32 × 4-bit values (E2M1: 1 sign + 2 exponent + 1 mantissa). The per-block scale compensates for FP4's tiny range (±6 without scaling).

## What's been proven

### NVIDIA paper (arXiv 2509.25149, September 2025)

First public NVFP4 pretraining:
- **Model size**: 12B parameters
- **Training data**: 10T tokens — longest publicly documented 4-bit training run
- **Result**: training loss and downstream task accuracy **comparable to FP8 baseline**
- **Key technique**: novel approach for stable training — per-block microscaling + stochastic rounding + careful scaling

### Speedup numbers

| Comparison | Speedup | Hardware |
|---|---|---|
| NVFP4 vs FP8 | 1.59-1.73× | Blackwell / GB300 |
| NVFP4 vs BF16 | ~2.6× | Blackwell |
| NVFP4 inference (FLUX.2) | 6.3× | B200 |

### Third-party progress

| Who | What | When |
|---|---|---|
| NVIDIA | 12B model, 10T tokens, NVFP4 pretraining | 2025-09 |
| Quartet II (paper) | Improved NVFP4 pretraining, loss gap vs BF16 reduced 15-25% | 2026-01 |
| NVIDIA + JAX/MaxText | NVFP4 training recipe for Blackwell + Rubin | 2026-06 |
| LMSYS (Miles team) | First end-to-end MXFP8 + NVFP4 **RL training** | 2026-07 |
| Unsloth | NVFP4 inference quantization (4-bit MLP + 8-bit attention) | 2026-08 |

## What's still missing

1. **No 100B+ model trained with NVFP4** — only 12B tested publicly. DeepSeek V3 proved FP8 at 671B scale, but nobody has tried NVFP4 at frontier model scale.

2. **Training stability** — NVIDIA's own paper states: "poses challenges to training stability, convergence, and implementation, notably for large-scale models trained on long token horizons"

3. **MXFP8 is more stable** — LMSYS RL experiments found MXFP8 (block-level scaling FP8) has better convergence tracking than NVFP4, though final accuracy is comparable. MXFP8 may be the safer stepping stone.

4. **Hardware limited** — Only Blackwell (B200/GB300) supports FP4. H100/H200 cannot do hardware FP4.

5. **RL training just started** — LMSYS did the first end-to-end NVFP4 RL in July 2026. Very early.

6. **No third-party frontier lab adoption** — No public evidence that OpenAI, Anthropic, Google, Meta, or xAI use NVFP4 training. They likely use FP8 (H100 cluster), but FP4 requires Blackwell which is still rolling out.

## Timeline

```
2024-12  DeepSeek V3 — FP8 training at scale (671B) — proves FP8 works
2025-09  NVIDIA — NVFP4 pretraining (12B, 10T tokens) — proves FP4 might work
2026-01  Quartet II — improved NVFP4 pretraining method
2026-06  NVIDIA — NVFP4 training recipe for JAX/MaxText (Blackwell + Rubin)
2026-07  LMSYS — first end-to-end NVFP4 RL training
2026-08  Now — NVFP4 training still experimental, not validated at 100B+ scale
```

## Comparison: precision levels in training

| | BF16 | FP8 | MXFP8 | NVFP4 |
|---|---|---|---|---|
| Bits per element | 16 | 8 | 8 (block-scaled) | 4 (block-scaled) |
| Scaling | None | Per-tensor/tile | Per-block (32 elements) | Per-block (32 elements) |
| Master weights | FP32 | FP32 | FP32 | FP32 |
| Hardware | All modern GPUs | H100+ | Blackwell+ | Blackwell+ |
| Maturity | Production | Early production | Early | Experimental |
| Max scale proven | 1T+ params | 671B (DeepSeek V3) | RL (LMSYS) | 12B (NVIDIA) |
| Speedup vs BF16 | 1× (baseline) | 2× | ~2× | ~2.6× |

## NVIDIA's strategy

1. Push hardware support (Blackwell → Rubin) ✅
2. Push framework support (Transformer Engine, JAX/MaxText) ✅
3. Prove it works (12B paper, LMSYS RL) ✅
4. Get large labs to adopt ⏳ (in progress)
5. Become industry standard ❌ (not yet)

Currently at step 3-4. The key question: will NVFP4 work at 100B+ scale? NVIDIA's bet is yes (model overparameterization absorbs 4-bit noise). Unproven.

## Sources

- [arXiv 2509.25149](https://arxiv.org/abs/2509.25149) — NVIDIA NVFP4 pretraining paper
- [Quartet II (arXiv 2601.22813)](https://arxiv.org/html/2601.22813v1) — improved NVFP4 method
- [LMSYS blog](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl) — MXFP8 + NVFP4 RL
- [NVIDIA developer blog](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/) — NVFP4 overview
- [NVIDIA + JAX/MaxText](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/) — training recipe
