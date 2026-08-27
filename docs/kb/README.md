# Kernel Knowledge Base

Hands-on findings from kernel experiments on NVIDIA H100 and AMD MI325X.

## Index

### GEMM
- [gemm-determinism.md](gemm-determinism.md) — Run-to-run and cross-batch determinism on H100 vs MI325X
- [gemm-batch-invariance.md](gemm-batch-invariance.md) — Batch non-invariance: causes, measurement, solutions
- [gemm-mixed-precision.md](gemm-mixed-precision.md) — FP16/BF16/FP8 internal pipeline and accumulator precision
- [gemm-fp16-vs-bf16.md](gemm-fp16-vs-bf16.md) — FP16 vs BF16 cross-batch diff: systematic comparison (batch + non-batch)

### Op Determinism
- [op-determinism.md](op-determinism.md) — Softmax, reduction, RMSNorm, LayerNorm determinism

### Precision tradeoffs
- [training-vs-inference-bf16.md](training-vs-inference-bf16.md) — Why training uses BF16 but inference prefers FP16

### Hardware
- [hardware-h100.md](hardware-h100.md) — NVIDIA H100 Tensor Core, cuBLAS, TF32
- [hardware-mi325x.md](hardware-mi325x.md) — AMD MI325X MFMA, rocBLAS
