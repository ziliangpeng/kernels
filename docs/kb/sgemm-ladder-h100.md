# SGEMM Optimization Ladder on H100 — First-Hand Results

First-hand benchmark data from writing and tuning every rung of the SGEMM ladder by hand.
Full step-by-step derivation, code walkthrough, and analysis: [`matmul/worklog.md`](../../matmul/worklog.md) (750 lines).

- GPU: H100 80GB HBM3, SM90, 132 SMs @ 1.98 GHz
- Node: pi1-h100-11 (idle)
- Shape: N=4096 square FP32 GEMM, median of 100 iterations
- Baseline: cuBLAS FP32 (`CUBLAS_PEDANTIC_MATH`) = **52.2 TFLOPS**

Date: 2026-08-27 (experiments originally run 2026-05-30/31)

## The ladder

| Step | Kernel | TFLOPS | % of cuBLAS | Gain | Key change |
|---|---|---:|---:|---:|---|
| 1 | Naive (16×16) | 5.3 | 10.2% | — | one output per thread, no reuse |
| 2 | Coalesced (1D mapping) | 5.7 | 10.9% | +7% | warp = one row, broadcast + coalesced B |
| 3 | SMEM tiling (32×32) | 9.0 | 17.2% | +59% | block-level tile reuse in shared memory |
| 4 | 1D blocktile (TM=8) | 17.6 | 33.7% | +96% | register reuse of B across 8 outputs |
| 4a | 1D blocktile **autotuned** | **19.3** | 36.9% | +9% | best (BM,BK,TM) = (64,4,16) |
| 5 | 2D blocktile (TM=TN=8) | 22.4 | 42.9% | +27% | outer product — reuse A **and** B in registers |
| 5a | 2D blocktile **autotuned** | **34.0** | 65.2% | +53% | best (BM,BN,BK,TM,TN) = (128,128,16,16,8) |
| 6 | Vectorized (float4 + A-transpose) | 32.9 | 63.0% | +47% | float4 GMEM loads + SMEM A-tile transpose |
| 6a | Vectorized **autotuned** | **34.8** | 66.7% | +6% | same winning config as 5a |

Naive → best: **6.6×**. Best hand-tuned rung reaches ~67% of cuBLAS FP32 — the remaining gap is Tensor Core (WMMA step exists in `matmul/matmul_wmma.cu`, not yet benchmarked in this ladder).

## Arithmetic intensity per rung (the core mechanism)

| Kernel | SMEM reads → madds | madd/read | Reused in registers |
|---|---|---:|---|
| SMEM tiling | 2 → 1 | 0.50 | nothing |
| 1D blocktile | 9 → 8 | 0.89 | B only |
| 2D blocktile | 16 → 64 | 4.00 | A **and** B (outer product) |

Recursive tiling is the unifying idea: **HBM tile → SMEM tile → register tile → (next: Tensor Core fragment)**. Each tile level unlocks reuse at the next level of the memory hierarchy. A "block tile" is literally a tile of a tile.

## First-hand lessons (each verified by measurement, not theory)

1. **Warp-to-row mapping matters.** 16×16 blocks split a warp across two rows → every memory transaction halves its useful payload. Fixing the mapping alone: +7%.

2. **SMEM tiling is the biggest single jump for memory-bound kernels** (+59%). 32× fewer GMEM reads translate directly to throughput.

3. **Register reuse beats everything after that.** 1D blocktile (reuse B) → +96%. 2D blocktile (reuse A too, outer product) → +27% more.

4. **TM and TN are NOT mirror-symmetric** (most surprising finding). Same TM·TN=128, same SMEM, same threads:
   - TM=16, TN=8 → 34.03 T
   - TM=8, TN=16 → 23.96 T (**−30% from a mirror swap**)
   
   Cause: the inner j-loop hoists `regA[i]` out, so its register lifetime spans TN iterations. Short j-loop (TN=8) → clean allocation; long j-loop (TN=16) → register pressure → likely spills. The kernel source is symmetric; the hardware behavior is not. **No theory paper would tell you this.**

5. **BK has a sweet spot, not a monotonic ladder**: BK=8 → 32.4 T, BK=16 → 34.0 T (peak), BK=32 → 22.5 T (collapse). Doubling SMEM per block halves resident blocks per SM. Deeper K-chunk only pays until occupancy saturates.

6. **Bigger block ≠ better once occupancy is saturated.** BM/BN 128→256 loses (34.0 → 31.5 T); 256×256 → register spill, kernel rejected. At 4096² there are already 1024 blocks — plenty for 132 SMs.

7. **The default config is one of the worst.** The siboehm-A100 default (128,128,8,8,8) ranked 8th of 19. Autotune found +52% over it. H100 prefers smaller blocks, bigger thread tiles, deeper BK than A100.

8. **Bottlenecks shift after each fix — optimization is layered.** SMEM bank conflicts existed since 2D blocktile but were invisible while GMEM instruction pressure dominated. Only after `float4` cut memory instructions 4× did the 2-way bank conflict become ~30% of inner-loop latency — and only then did transposing the A tile pay (+47% combined). Prematurely transposing would have added complexity for zero gain.

9. **Autotune finds the best within an algorithm's ceiling; only a new algorithm raises the ceiling.** 1D autotuned (19.3 T) can never reach 2D baseline (22.4 T) — the ceiling is set by register reuse factor, not tuning.

10. **Reproducibility check matters**: winner stable across 3 runs at ±0.2%; winner-vs-runner-up gap +3.6% = 18× the noise floor. Also: an invalid BK=24 candidate was silently writing OOB SMEM — caught in review, fixed with divisibility checks, and the BK=24 datapoint retracted. Invalid configs must be filtered before launch, not after.

## Benchmark methodology

- 100-iteration median per config (kills outliers)
- Checksum + `torch.equal` verification against reference (not just timing)
- 3 independent full-sweep runs to confirm winner stability
- Validity constraints (thread-count divisibility) enforced in the autotuner

## Related KB notes

- [gemm-determinism.md](gemm-determinism.md) — run-to-run and cross-batch determinism of these same GEMM ops
- [gemm-mixed-precision.md](gemm-mixed-precision.md) — why FP32 accumulator + FP16/BF16 output is standard
- [hardware-h100.md](hardware-h100.md) — Tensor Core / cuBLAS baseline details

## Source

- Full experiment log: `~/code/kernels/matmul/worklog.md`
- Kernels: `~/code/kernels/matmul/matmul_{naive,coalesced,smem,1d_blocktile,2d_blocktile,vectorized,warptile,wmma}.cu`
- Autotune details: `~/code/kernels/matmul/autotune.md`
