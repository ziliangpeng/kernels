# FP32 Reference vs FP16/BF16 GEMM Accuracy

**Experiment date**: 2026-08-27 · **GPUs**: NVIDIA H100 80GB (gcp5), AMD MI325X (amd2 dev pod)
**Script**: `determinism/fp32_vs_lowprec.py` · **Results**: `determinism/results_fp32_vs_lowprec_{h100,mi325x}.json`

## Question

How far does the FP16/BF16 GEMM output drift from the FP32 ground truth?
Intuition says "low precision accumulates error over big K" — is that true?

## Methodology

- One set of seeded FP32 inputs (seed=42), same across all variants.
- **FP32 reference**: `torch.matmul` with `allow_tf32=False` (true FP32 cuBLAS, not TF32).
- **FP16/BF16 runs**: inputs cast down, output cast back to FP32 for comparison.
  This mirrors real inference — low-precision weights/activations, FP32
  accumulation inside the library (cuBLAS / hipBLASLT default).
- Metrics vs FP32 reference: max element diff %, mean %, Frobenius norm
  relative error %, p99 %.
- Shapes: M ∈ {1, 8, 64, 512, 4096} (decode → prefill), K ∈ {4096, 16384}, N=4096.

## Results

| dtype | Frobenius rel err | p99 | max element |
|---|---|---|---|
| FP16 | **0.036%** | 0.098% | 0.04–0.05% |
| BF16 | **0.288%** | 0.78% | 0.28–0.42% |

Identical to 3 decimal places on H100 AND MI325X.

### bf16/fp16 error ratio is exactly 8× on both GPUs

Every shape, both GPUs: 8.0× ± 0.1. Not a coincidence:

- The dominant error source is **input cast rounding**, not accumulation.
- BF16 has 8-bit mantissa vs FP16's 10-bit → cast rounding error is 2² = 4× worse.
- The remaining ~2× comes from position/rounding-mode differences between the
  two formats on the same input distribution.

## Three key insights

### 1. The 8× ratio — cast error dominates, not accumulation

Because libraries accumulate in FP32, the per-partial-sum rounding error is
negligible compared to the rounding already baked into the low-precision
inputs. The output error is essentially `error(cast(A))·B + A·error(cast(B))`
propagated linearly — nothing to do with K-length accumulation drift.

### 2. Error is INDEPENDENT of M and K

M=1 (decode) vs M=4096 (prefill), K=4096 vs 16384: fro% identical at
~0.036% / ~0.288%. The intuition "big K → error accumulates" is wrong here:
FP32 accumulation error stays far below input-cast error at any realistic K.
Measured, not inferred — this is what the data shows.

### 3. Max element diff has a tail (up to 0.42%)

BF16 worst-case element hits 0.42% relative to the global max |C|. These
outliers are what compounds through layers — but after Frobenius
normalization, total error stays tiny.

## Cross-GPU: H100 vs MI325X — numerically identical accuracy

For accuracy purposes the two architectures are indistinguishable (0.036%
vs 0.036%, 0.288% vs 0.288%). Both libraries do the same thing: low-prec
multiply + FP32 accumulate. The differences we found earlier were in
**determinism** (algo selection), not accuracy.

## The BF16 paradox, resolved

BF16 is 8× less accurate than FP16, yet training uses BF16. Why:

1. **Range > precision**: BF16 has FP32-identical exponent range (±3.4e38);
   FP16 caps at ±65504. Overflow destroys training; small rounding doesn't.
2. 0.288% per-GEMM error is well below what backprop noise injects anyway.
3. This is why **inference** often prefers FP16 (accuracy matters more,
   activations are bounded) while **training** prefers BF16 (range matters).

## Pitfall encountered (gcp5 H100)

User-site torch 2.7.1+cu126 bundles its own NCCL, but the Slurm batch
environment resolves system `libnccl.so.2` (2.20.5) which lacks
`ncclCommInitRankScalable` → `ImportError: undefined symbol` on
`import torch`. Fix in sbatch:

```bash
export LD_PRELOAD=/home/ziliang/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export PATH=/usr/local/nvidia/bin:$PATH  # nvidia-smi not in default batch PATH either
```

## Links

- Cross-batch determinism (companion finding: FP16 MORE deterministic than FP32): `gemm-determinism.md`
- Why accuracy-vs-determinism are different axes: `gemm-fp16-vs-bf16.md`
- Mixed precision pipeline: `gemm-mixed-precision.md`
