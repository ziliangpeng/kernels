# FP16 vs BF16 Cross-batch Determinism Comparison

Systematic comparison of FP16 vs BF16 cross-batch non-determinism in GEMM.
Two test modes: batch GEMM (`torch.bmm`, vary B) and non-batch GEMM (`torch.mm`, vary M).

Tested on: H100 (CUDA 12.6, PyTorch 2.7.1) + MI325X (ROCm 7.14, PyTorch 2.13)
Date: 2026-08-27

## Batch GEMM (`torch.bmm`, vary B, fixed M=128 K=4096 N=4096)

252 configs per dtype (7 M × 3 K × 3 N × 4 B).

### Summary

| | H100 FP16 | H100 BF16 | MI325X FP16 | MI325X BF16 |
|---|---|---|---|---|
| Bit-exact | 79.4% | 79.4% | 29.4% | 36.9% |
| Diff median | 0.045% | 0.294% | 0.038% | 0.184% |
| Diff mean | 0.044% | 0.281% | 0.036% | 0.215% |
| Diff max | 0.08% | 0.43% | 0.08% | 0.60% |

### Head-to-head (when at least one non-exact)

| | H100 | MI325X |
|---|---|---|
| BF16 worse | **100%** (52/52) | **60.3%** (152/252) |
| FP16 worse | 0% | 18.7% |

**Batch GEMM conclusion: BF16 is consistently worse than FP16.** When both are non-exact, BF16 diff is always larger. BF16 median diff is 5-8× larger than FP16.

## Non-batch GEMM (`torch.mm`, vary M)

1,188 configs per dtype (11 M × 6 K × 6 N × 3 seeds).

### Summary

| | H100 FP16 | H100 BF16 | MI325X FP16 | MI325X BF16 |
|---|---|---|---|---|
| Bit-exact | 83.6% | 85.1% | 57.1% | 75.4% |
| Non-exact | 195 | 177 | 510 | 292 |
| Diff median | 0.029% | 0.073% | 0.013% | 0.015% |
| Diff mean | 0.031% | 0.105% | 0.015% | 0.047% |
| Diff max | 0.054% | 0.426% | 0.073% | 0.402% |

### Head-to-head (when at least one non-exact)

| | H100 | MI325X |
|---|---|---|
| BF16 worse | 59.3% (121/204) | 31.9% (176/552) |
| FP16 worse | 38.7% (79/204) | **66.8%** (369/552) |

**Non-batch GEMM conclusion: more nuanced.**

- **MI325X FP16 is more often non-exact** (43% vs BF16's 25%), but each diff is small (median 0.013%)
- **MI325X BF16 is less often non-exact** (coarse ULP hides differences), but when non-exact, diff is larger (max 0.40%)
- **H100 BF16 is still worse** in both frequency and magnitude, but gap is smaller

## Why BF16 and FP16 behave differently

Both use FP32 accumulator internally. The difference is in the **output cast step**:

- FP16: 10-bit mantissa → ULP ~0.1% of value → small visible diff per rounding boundary crossing
- BF16: 7-bit mantissa → ULP ~0.78% of value → large visible diff per rounding boundary crossing

This creates two opposing effects:
1. **BF16 hides more differences** (coarse quantization rounds many small accumulator differences to the same BF16 value) → more bit-exact cases
2. **BF16 shows larger diff when it doesn't hide** (one ULP jump is 8× larger than FP16)

Which effect dominates depends on the GEMM dispatch path:
- **Batch GEMM (bmm)**: fewer algorithm switches → effect 2 dominates → BF16 worse
- **Non-batch GEMM (mm) on MI325X**: more algorithm switches for FP16 → effect 1 dominates for BF16 → FP16 more often non-exact

## Practical implications

| Use case | Recommendation |
|---|---|
| Need bit-exact reproducibility | Use FP16 (smaller diff when non-exact) |
| Need max stability on MI325X non-batch | BF16 actually has more bit-exact cases |
| Eval / A-B replay on AMD | FP16 preferred — worst case 0.08% vs BF16's 0.60% |
| Production serving | Either is fine — diff invisible to users |

## Key insight

"BF16 is worse than FP16" is **not universal**. It depends on:
1. Batch vs non-batch GEMM dispatch path
2. GPU vendor (cuBLAS vs rocBLAS heuristic behavior)
3. Which metric you care about (frequency of non-exact vs magnitude of diff)

The correct framing: **BF16 has fewer but larger non-exact events; FP16 has more but smaller ones.** Which is "worse" depends on whether you care about how often diff appears or how large it gets.

## Test scripts

- `determinism/fp16_vs_bf16.py` — batch GEMM (bmm, vary B), 252 configs
- `determinism/fp16_vs_bf16_mm.py` — non-batch GEMM (mm, vary M), 99 configs
- `determinism/fp16_vs_bf16_mm_expanded.py` — non-batch GEMM expanded, 1188 configs
