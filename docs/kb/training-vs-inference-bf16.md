# BF16 vs FP16: Training vs Inference

## Why training uses BF16

### The real reason: gradient overflow/underflow

Training gradients can be very small (1e-7) or very large (1e+3 during loss spikes).

- FP16 range: ±65504 → small gradients underflow to 0 (learning stops), large gradients overflow to NaN (training crashes)
- BF16 range: ±3.4×10³⁸ (same as FP32) → no overflow/underflow → training is stable

This is the **only** real reason training prefers BF16. Not because BF16 is "more accurate" — it's because FP16 can crash training.

### Why BF16's 0.3% cross-batch diff doesn't matter in training

1. **SGD noise dominates**: A batch of 256 samples has ~6% sampling variance. BF16's 0.3% numerical noise is 20× smaller — completely drowned out.

2. **Self-correction across steps**: If step N's gradient is 0.3% too large, the weight overshoots slightly, making step N+1's gradient slightly smaller. Errors cancel over 100K steps.

3. **Gradient descent is inherently noisy**: The optimizer (Adam, SGD) is designed to work with noisy gradients. Adding 0.3% more noise changes nothing.

## Why BF16 is problematic in inference

Inference has none of these protections:

1. **No SGD noise to mask it**: Inference is a deterministic forward pass. BF16's 0.3% is the ONLY noise source.

2. **No self-correction**: Each token is an independent argmax. One wrong token → different continuation → no recovery.

3. **Greedy = winner-take-all**: Top-1 and top-2 logits may differ by only 0.1%. BF16's 0.3% noise (from cross-batch algorithm switching) can flip argmax → completely different output.

4. **MarginGate paper (arXiv 2605.30218) confirmed**: Token flips in BF16 greedy decoding are "sparse, local, and exposed by near-tie logits." The problem is real but rare.

## Why inference still uses BF16 anyway

Most modern models (Gemma 4, LLaMA 3, etc.) are trained in BF16. The model weights are stored in BF16. Inference uses BF16 because:

1. **Matching dtype**: Casting BF16 weights to FP16 introduces its own error
2. **No overflow risk**: Activations can have large values during prefill; BF16 is safer
3. **The problem is rare**: Most tokens don't have near-tie logits, so 0.3% noise doesn't flip them
4. **No better option on AMD**: TML batch_invariant_ops only works on NVIDIA

## Summary

| | Training | Inference |
|---|---|---|
| BF16 range advantage | **Critical** (prevents crash) | Minor (values usually bounded) |
| BF16 precision disadvantage | Irrelevant (masked by SGD noise) | **Visible** (can flip argmax) |
| Self-correction | Yes (errors cancel across steps) | No (each token independent) |
| Preferred dtype | **BF16** | **FP16** (but stuck with BF16 due to model weights) |

## Practical guidance

- **Training**: Always BF16. FP16 will crash on large models.
- **Production inference**: BF16 is fine. Token flips are rare and users don't notice.
- **Eval / A-B replay**: FP16 preferred (smaller worst-case diff). Or use batch-invariant ops (NVIDIA only).
- **Debugging**: Fix batch size to eliminate cross-batch diff. Or use manual loop (fixed algo, 0% diff, but slower).
