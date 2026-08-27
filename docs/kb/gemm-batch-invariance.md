# GEMM Batch Invariance

Batch invariance = same input, different batch size, output bit-exact (diff = 0.0000%).

## The problem

cuBLAS and rocBLAS select different GEMM algorithms based on M/B dimensions. Different algorithms have different tiling, K-loop unrolling, and accumulation order → different FP rounding → different results for the same input.

This is NOT run-to-run non-determinism (same shape is always stable). It is cross-batch: the same input vector produces different output when computed alone (B=1) vs in a batch (B=2+).

## Measured impact

See [gemm-determinism.md](gemm-determinism.md) for full results.

Worst case: MI325X BF16 bmm, 0.35% relative diff. This is large enough to flip argmax in greedy decoding.

## Solution: fixed algorithm

**Manual loop** — run B separate `torch.mm` calls (always same shape) forces the library to use the same algorithm every time → bit-exact.

| Approach | H100 FP16 | MI325X FP16 | H100 BF16 | MI325X BF16 |
|---|---|---|---|---|
| `torch.bmm` (default) | ✅ 0.0000% | ❌ 0.0434% | ✅ 0.0000% | ❌ 0.17~0.35% |
| Manual loop (fixed algo) | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% | ✅ 0.0000% |

Manual loop is a PoC (B× slower). A real solution needs a single fixed-tiling batched kernel.

## Existing solutions

### TML batch_invariant_ops (NVIDIA only)

GitHub: [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)

Uses `torch.Library` to swap PyTorch kernels with custom batch-invariant ones. Forces fixed tiling + fixed reduction tree for all batch sizes.

- NVIDIA only (compute capability ≥ 8.0)
- No AMD support
- Performance cost: gives up heuristic algorithm selection

### vLLM VLLM_BATCH_INVARIANT

Env var `VLLM_BATCH_INVARIANT=1` loads TML library. NVIDIA only.

Use cases: framework debugging, model debugging, RL training, eval/replay.

### AMD: no solution

rocBLAS has `rocblas_atomics_not_allowed=1` — disables atomic operations but does NOT fix tiling switches. No equivalent to TML library exists for ROCm.

## When batch invariance matters

| Scenario | Need batch invariance? |
|---|---|
| Production serving | ❌ No (diff invisible to users) |
| Training | ❌ No (gradient descent tolerates rounding) |
| Eval / A-B replay | ✅ Yes (different batch → different token) |
| Debugging | ✅ Yes (need to isolate bugs from noise) |
| RL training | ✅ Yes (need reproducible rollouts) |
