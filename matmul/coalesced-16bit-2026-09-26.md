# Coalesced FP16 — Rung 2 of the 16-bit Ladder

**Date**: 2026-09-26
**Kernel**: `matmul/matmul_coalesced_typed.cu` (method `coalesced_f16`)
**GPU**: H100 80GB HBM3 (gcp5, SM90) — Slurm jobs 219185 (build) + 219187 (data)
**Semantics**: FP16 storage + FP32 accumulation (16-bit format settled on FP16,
2026-09-26). Same thread mapping as `matmul_coalesced.cu` (32×32 tile, 1D
indexing, coalesced B loads). Conversion lazy/untimed.
**Baseline**: `naive` / `coalesced` / `naive_f16` from the same job/session.
**Timing**: repo-standard 100-iteration batched event timing.

## Results (square GEMM, TFLOPS)

| N | coalesced FP32 | coalesced FP16 | ratio | naive FP32 | naive FP16 |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.28 | 2.02 | 0.89x | 3.17 | 2.14 |
| 512 | 5.85 | 4.95 | 0.85x | 4.68 | 3.97 |
| 768 | 5.66 | 4.78 |  - | 4.98 | 3.98 |

(naive rows from job 219168 same binary/session.)

Verification: `coalesced_f16` max rel err 8.47e-05 — PASS (threshold 5e-3;
pure input quantization error).

## Reading the data

1. **Coalescing gains nothing on the naive rung at N≤768** (5.85 vs 4.68 at
   N=512) — both rungs are latency/issue-bound, not bandwidth-bound; the
   classic +7% coalescing win from Simon's A6000 shows up at larger N where
   bandwidth starts to bind. Same story as naive: 16-bit storage loses
   (0.85–0.89x) because scalar loads still move one element per instruction.
2. **FP16 cross-rung ratio is stable** (~0.85x at every N): the cvt+load
   overhead mechanism is rung-independent, as predicted.
3. No bandwidth-bound rung has been 16-bit'd yet — the >1.0x crossover
   prediction is still untested. First real test: `smem_f16` (rung 3).

## Implementation notes

- Identical structure to `matmul_naive_typed.cu`; converter kernel duplicated
  per-TU (safe: internal linkage, one instantiation each).
- Registered at all three dispatch sites + usage + skip rule (N≥1024 skip in
  all-methods sweep, same as FP32 coalesced).
