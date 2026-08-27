# Kernel Determinism Test Suite

## Goal

Empirically test run-to-run and cross-implementation determinism of common inference kernels on NVIDIA GPU.

## Test methodology

For each op:
1. Fix input tensors (seeded RNG, same dtype)
2. Run kernel N=100 times, record output checksum (sum of all elements + hash)
3. Compare:
   - **Run-to-run**: same kernel, same input, does output change across 100 runs?
   - **Cross-impl**: different implementations of same op, are outputs bit-exact?
4. Toggle determinism flags (`CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`) and re-test
5. Record: which ops are stable, which drift, what's the magnitude of drift (max abs diff)

## Ops to test

| # | Op | Implementations to compare | Key determinism question |
|---|---|---|---|
| 1 | **Softmax** | naive (loop), online (rescaling), warp-shuffle, cuDNN | reduction order → bit-exact? |
| 2 | **GEMM** (FP32) | naive, shared-mem tiled, cuBLAS, cuBLASLt, Triton | algo selection → run-to-run stable? |
| 3 | **GEMM** (BF16) | wmma, cuBLAS bf16, Triton bf16 | mixed accumulation → bit-exact? |
| 4 | **RMSNorm** | naive, fused | reduction order → bit-exact? |
| 5 | **Reduction** (sum) | naive, warp-shuffle, atomic, CUB | associativity → bit-exact? |

## Environment variables to test

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (cuBLAS deterministic workspace)
- `torch.use_deterministic_algorithms(True)`
- Default (no flags) — production mode

## Files

- `run_tests.py` — main test harness, runs all ops, outputs CSV
- `kernels/` — standalone kernel implementations (CUDA + PyTorch reference)
- `results/` — CSV output + analysis
