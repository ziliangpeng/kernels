# GEMM Mixed Precision Pipeline

## Standard pipeline

All frameworks use mixed precision for FP16/BF16 GEMM:

```
Input:  A [M, K] — FP16/BF16
        B [K, N] — FP16/BF16
Step 1: Load tiles to shared memory (FP16/BF16)
Step 2: Multiply A × B → FP32 accumulator (hardware register)
Step 3: Accumulate across K tiles in FP32
Step 4: Cast FP32 → FP16/BF16 → write output
Output: C [M, N] — FP16/BF16
```

Nobody uses FP16/BF16 as accumulator in standard GEMM. FP16 max = 65504, K=4096 accumulation can overflow for non-centered distributions.

## Why FP32 accumulator

Each `A[i][k] × B[k][j]` product with FP16 inputs needs ~22 bits to represent exactly. FP32 has 24-bit mantissa → barely enough for single product.

Main error source is **summation order across K**, not individual multiply. Different algorithms sum in different orders → different FP32 rounding → different results.

K=4096 with FP16 accumulator: ULP at sum~5000 is ~0.5, 4096 additions → error ~2048. Completely unacceptable.
K=4096 with FP32 accumulator: ULP at sum~64 is ~0.000008, error negligible.

## FP8 exception

H100 supports FP8 GEMM with **FP16 accumulator** (not FP32) in "fast accumulation" mode. Internal partial accumulator may be even narrower than FP16. This is the one case where sub-FP32 accumulation is used in practice. Trades accuracy for ~2× throughput. Only for inference, not training.

**Note (Sol review):** FP32 accumulation is the mainstream safe default, not a universal hardware law. FP16 accumulator, reduced-precision split-K reduction, and FP8 narrower internal fast accumulation all exist in real hardware. Exact behavior depends on hardware, backend, compute type, and math-mode flags.

## FP16 vs BF16

| | FP16 | BF16 |
|---|---|---|
| Exponent | 5 bit (range ±65504) | 8 bit (same range as FP32) |
| Mantissa | 10 bit (~3.7 decimal digits) | 7 bit (~2.4 decimal digits) |
| Training | Overflow/underflow risk | Safe (FP32 range) |
| Output cast | Risk of overflow (sum > 65504 → NaN) | Cannot overflow (same exponent as FP32) |

### Why BF16 has larger observed cross-batch diff (Sol review correction)

Previous explanation ("BF16 amplifies accumulation rounding") was imprecise. The correct explanation:

> BF16's coarser output quantization means small accumulator differences are often hidden if they round to the same BF16 value, but when they cross a rounding boundary the visible difference is at least one relatively large BF16 ULP. This can produce fewer non-exact cases but larger observed max differences than FP16.

The FP32 accumulator's addition order rounding error is the same regardless of output dtype. The difference is in the **output cast step**: BF16's 7-bit mantissa means each ULP is larger, so when a rounding boundary is crossed, the visible diff is larger.

## Hardware implementation

### NVIDIA Tensor Core (wmma)

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a_frag;  // FP16 input
wmma::fragment<wmma::matrix_b, 16, 16, 16, half> b_frag;  // FP16 input
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;  // FP32 accumulator

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // c += A × B (FP32 accumulate)
// Hardware auto-casts FP32 → FP16 on store
```

Tile size 16×16×16 is hardware-fixed. Software cannot choose different tile → fewer algorithm variants → more deterministic (see gemm-determinism.md).

**Caveat (Sol review):** MMA instruction's fixed tile only fixes the *local* reduction structure, not the entire GEMM. CTA tile, warp tile, K-stage count, split-K, edge-tile handling, and epilogue fusion can still vary.

### AMD Matrix Core (MFMA)

`__mfma_f32_16x16x16_f16` instruction. Input: packed FP16. Output: FP32 accumulator. Same mixed precision as Tensor Core.

Supports multiple tile sizes (16×16×16, 32×32×16) → more algorithm variants than Tensor Core → more cross-batch switching on MI325X.

## FP32 → FP16/BF16 output cast

FP32 → BF16: truncate mantissa (23→7 bit), exponent unchanged. **Cannot overflow** (same exponent range).

FP32 → FP16: must check range. If sum > 65504 → overflow → NaN. This is why BF16 is safer for inference.
