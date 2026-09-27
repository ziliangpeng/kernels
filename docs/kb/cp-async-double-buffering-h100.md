# cp.async Double Buffering on H100 — First-Hand Results

Distilled takeaways from implementing Simon Boehm's double-buffering rung
([SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) kernels 11-12) as
`matmul_warptile_dbuf.cu`. Full experiment log:
[`matmul/warptile-dbuf-2026-09-26.md`](../../matmul/warptile-dbuf-2026-09-26.md).

- GPU: H100 80GB HBM3 (an-h100-node, SM90), CUDA 12.4, same-session baselines
- Shape: N=4096 FP32, 100-iteration batched timing, 3-run stability ±0.02%

## The number that matters

| Kernel | TFLOPS | % cuBLAS FP32 |
|---|---:|---:|
| warptile (sync loads) | 28.1 | 54.2% |
| **warptile + cp.async double buffer** | **32.8** | **63.2%** |
| vectorized (rung 6, for reference) | 32.7 | 63.0% |
| cuBLAS FP32 pedantic | 51.9 | 100% |

**+16.5% from the pipeline alone**, shape-stable (512: +18%, 2048: +15%).
dbuf lands exactly on vectorized — the async pipeline recovers what float4
GMEM loads gave, via overlap instead of width.

## Lessons (each verified by measurement)

1. **The portable API was the portability problem.** `cuda::barrier` +
   `cuda::memcpy_async` (Simon's rung 12) does not compile on CUDA 12.4's
   libcudacxx (`cuda::init` missing). Raw `cp.async` PTX with
   `commit_group`/`wait_group` compiled first try and is what CUTLASS uses
   pre-TMA anyway. When a vendor "portable" wrapper fails, drop to the PTX it
   wraps.

2. **`cp.async.wait_group N` is a pipeline-stage selector, not a fence.** With
   groups [current, next] in flight, `wait_group 1` completes current while
   next keeps flying — the entire double-buffer in one instruction. The
   `__syncthreads()` after it provides block-wide visibility; the one after
   compute provides buffer-reuse safety.

3. **cp.async's src-size operand is an OOB handler for free.** `cp.async ... 4,
   0` zero-fills the destination — boundary tiles need no branches in the
   store path, only a clamped source pointer and a size select.

4. **memcheck is blind to logic bugs; the checksum gate is the gate.** The
   B-tile double-offset bug (blockCol×BN added twice) read wrong-but-in-bounds
   memory: compute-sanitizer reported 0 errors while --verify showed 24% max
   rel error. OOB tools catch address errors, only a reference comparison
   catches *which* data you loaded.

5. **A-transposed loads stay 4B.** A 16B load cannot land transposed in SMEM,
   so A keeps scalar async copies (Simon's choice too); B takes 16B `.cg`.
   Alignment guard `(addr & 15) == 0` with 4B fallback keeps odd-N correct.

## Where this leaves the FP32 ladder

63% of cuBLAS. The remaining 37% is tensor-core territory (WGMMA), not a
CUDA-core tuning problem. Autotuning the dbuf config (BM/BN/BK, TM/TN,
WM/WN) is the one lever left before switching to tensor cores.

## Related

- [sgemm-ladder-h100.md](sgemm-ladder-h100.md) — the ladder through rung 6 (pi1 numbers)
- `matmul/autotune.md` — why warptile-level config space needs autotune
- `matmul/warptile-dbuf-2026-09-26.md` — full worklog, bug post-mortems, repro
