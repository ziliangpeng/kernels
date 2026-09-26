# Warp-tile GEMM + cp.async Double Buffering — First-Hand Results

**Date**: 2026-09-26
**Kernel**: `matmul/matmul_warptile_dbuf.cu` (method `warptile_dbuf`)
**GPU**: H100 80GB HBM3 (gcp5, SM90, 132 SMs @ 1.98 GHz) — gcp5-h100-0-28
**Baseline**: same-session `warptile`, `vectorized`, `cublas` from the same build
**Shape**: N=4096 square FP32 GEMM, 100-iteration batched timing (repo standard)

This is the repo's implementation of Simon Boehm's rung 11/12 (double buffering,
[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) kernels 11-12). Simon's rung 11
splits threads (half load / half compute); his rung 12 uses cuda::barrier +
cuda::memcpy_async. We implement the CUTLASS-style raw cp.async PTX pipeline
(commit_group / wait_group), which is the canonical pre-TMA pattern on Ampere+.

## What changed vs `warptile` (single-variable A/B)

Exactly one thing: how the GMEM→SMEM tile loads are performed.

| | `warptile` (before) | `warptile_dbuf` (after) |
|---|---|---|
| Load path | synchronous LDG→STS through registers | `cp.async` PTX (bypasses registers) |
| Buffers | 1 SMEM tile per operand | 2 SMEM tiles per operand (ping-pong) |
| Overlap | none — load stalls whole block | next tile loads while current computes |
| B width | 4B per thread | 16B `cp.async.cg` (4B fallback at boundaries/unaligned) |
| OOB | guarded stores of 0.0f | cp.async src-size zero-fill |

Tiling, load mapping, thread placement, compute inner loop, and the epilogue are
identical to `matmul_warptile.cu`. Any perf delta is attributable to the pipeline.

## Results (N=4096, gcp5 H100, 3-run stability ±0.02%)

| Kernel | TFLOPS | vs cuBLAS FP32 | vs warptile |
|---|---:|---:|---:|
| cublas (FP32 pedantic) | 51.9 | 100% | +85% |
| vectorized (rung 6) | 32.7 | 63.0% | +16.3% |
| warptile (rung 10) | 28.1 | 54.2% | — |
| **warptile_dbuf (this work)** | **32.8** | **63.2%** | **+16.5%** |

- warptile 28.14T → warptile_dbuf 32.77T = **+16.4%**, 3 runs each, spread < 0.1T.
- dbuf now matches `vectorized` (32.77 vs 32.70T, +0.2% — within noise): the
  cp.async pipeline recovers exactly what `vectorized` got from float4 GMEM loads.
- Still 63% of cuBLAS: the remaining gap is Tensor-Core territory (WGMMA), out of
  scope for the FP32 ladder.

Shape sweep (single run): 512: 2.37T vs warptile 2.01T (+18%); 2048: 32.68T vs
28.38T (+15.1%); 4096: 32.77T vs 28.13T (+16.5%). Gain is shape-stable.

Correctness matrix (all vs CPU double-precision reference, threshold 1e-4):

| N | Path exercised | Max rel err | Result |
|---:|---|---:|---|
| 512 | aligned, full tiles | 1.4e-06 | PASS |
| 1000 | OOB boundary tiles + K-tail (cp.async src-size zero-fill) | 2.1e-06 | PASS |
| 2048 | aligned, full tiles | 3.4e-06 | PASS |
| 4096 | aligned, full tiles (scale) | 4.6e-06 | PASS |

## Pipeline mechanics (what makes it fast)

```
prologue:  cp.async(tile0 → buf0); commit;              // group G0 in flight
loop i:    cp.async(tile i+1 → buf 1-i); commit;        // group G(i+1) in flight
           cp.async.wait_group 1;                        // G(i) done, G(i+1) still flying
           __syncthreads();                              // block-wide visibility
           compute on buf i;                             // loads for i+1 overlap this
           __syncthreads();                              // buffer-reuse safety
```

- `cp.async.wait_group 1` is the key trick: with two commit-groups in flight it
  completes the CURRENT tile's group while the NEXT tile's copies keep flying.
- Each load of A is 4B (tile is transposed on store — a 16B load cannot land
  transposed, so we keep Simon's scalar-async pattern for A); B uses 16B `.cg`
  (bypasses L1, 128 threads × 4 passes cover the 16×128 tile).
- SMEM cost doubles: 2×(16×128 + 16×128)×4B = 32KB per block (fits trivially).

## Bugs found and fixed (the real lesson)

1. **cuda::init missing on CUDA 12.4** — libcudacxx on this toolchain has no
   `cuda::init(&barrier, count)`; Simon's rung-12 source doesn't compile as-is.
   Switched to raw cp.async PTX (commit/wait_group), which is what CUTLASS does
   pre-TMA anyway. Lesson: the "portable" cuda::barrier API is less portable
   than the PTX it wraps.
2. **B-tile double-offset bug** (max rel err 0.24, caught by --verify at 512):
   kernel pre-advances `B += blockCol * BN` at entry, but the loader also added
   `blockCol * BN` into its column index — every block except column 0 read the
   wrong B columns. compute-sanitizer memcheck reported ZERO errors (wrong-but-
   in-bounds reads); only the CPU-reference checksum caught it. Lesson: memcheck
   can't catch logic bugs; the checksum gate is the gate that matters.
3. **float4 alignment**: fast path guards `(addr & 15) == 0` and falls back to
   4B copies; odd-N matrices stay correct (learned from the repo's earlier
   BK=24 OOB incident — validity belongs in the fast path, not in retraction).

## How to reproduce

```bash
# on gcp5 (or any H100 with CUDA 12.4):
bash build_matmul.sh                  # nvcc direct build, sm_90, -O3
srun --partition=low --gres=gpu:1 ./matmul_bench --method warptile_dbuf -n 4096 --verify
```

## Next steps (not done here)

- Autotune the dbuf config (BM/BN/BK, WNITER, TM/TN) — autotune.md predicts
  +40-50% headroom at the warptile level; the pipeline should shift the optimum.
- Simon rung 11 (thread-split double buffering) as a comparison flavor — likely
  slower than cp.async (half the threads compute), but cheap to try.
- WGMMA (Hopper tensor cores) — the actual path to close the 37% gap to cuBLAS.
