# Kernel Knowledge Base

Hands-on findings from kernel experiments on NVIDIA H100 and AMD MI325X.

## Index

### GEMM
- [gemm-determinism.md](gemm-determinism.md) — Run-to-run and cross-batch determinism on H100 vs MI325X
- [gemm-batch-invariance.md](gemm-batch-invariance.md) — Batch non-invariance: causes, measurement, solutions
- [gemm-mixed-precision.md](gemm-mixed-precision.md) — FP16/BF16/FP8 internal pipeline and accumulator precision

### Op Determinism
- [op-determinism.md](op-determinism.md) — Softmax, reduction, RMSNorm, LayerNorm determinism

### Hardware
- [hardware-h100.md](hardware-h100.md) — NVIDIA H100 Tensor Core, cuBLAS, TF32
- [hardware-mi325x.md](hardware-mi325x.md) — AMD MI325X MFMA, rocBLAS
