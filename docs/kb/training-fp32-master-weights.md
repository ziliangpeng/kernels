# FP32 Master Weights in Training

## Standard mixed precision training

```
Forward/Backward:  BF16 (fast, half memory)
Master weights:    FP32 (accurate)
Gradient:          BF16 → cast to FP32 for optimizer update
Optimizer state:   FP32 (Adam m, v)
```

Each step:
1. Cast FP32 master weight → BF16 copy
2. Forward + backward using BF16 copy (fast GEMM via Tensor Core)
3. Gradient cast back to FP32
4. Optimizer updates FP32 master weight
5. Repeat

## Why FP32 master weights are mandatory (not just best practice)

Weight update: `w_new = w_old - lr * grad`

If `lr=1e-5` and `grad=1e-3`, update = `1e-8`.

- FP32 ULP at w~1.0: ~1.2e-7 → can represent 1e-8 update ✅
- BF16 ULP at w~1.0: ~0.0078 → 1e-8 update rounds to 0 → **weight doesn't change** ❌

Without FP32 master weights, small learning rate updates vanish. This is a numerical requirement, not a convention.

## Does anyone NOT use FP32 master weights?

Rare, and always a conscious tradeoff:

| Approach | Who uses it? | Why | Cost |
|---|---|---|---|
| FP32 master (standard) | Almost everyone | Accurate, stable | 2× weight memory |
| BF16 master (pure BF16) | Very few, memory-constrained | Saves half weight memory | Loses precision, long training may drift |
| FP8 master | Research stage | Extreme memory savings | Loses even more precision |

Pure BF16 master weights don't work well because weight updates vanish (ULP too coarse). This is the same numerical problem as BF16 accumulation in inference — but worse, because training runs for 100K+ steps and small errors compound.
