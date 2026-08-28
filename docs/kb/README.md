# Kernel Knowledge Base

Hands-on findings from kernel experiments on NVIDIA H100 and AMD MI325X.

## Index

### GEMM
- [gemm-variant-taxonomy.md](gemm-variant-taxonomy.md) — The 6-dimension framework: memory ladder (D1), K-splitting (D2), shape (D3), precision (D4), hardware instructions (D5), implementation stack (D6)
- [gemm-determinism.md](gemm-determinism.md) — Run-to-run and cross-batch determinism on H100 vs MI325X
- [gemm-batch-invariance.md](gemm-batch-invariance.md) — Batch non-invariance: causes, measurement, solutions
- [gemm-mixed-precision.md](gemm-mixed-precision.md) — FP16/BF16/FP8 internal pipeline and accumulator precision
- [gemm-fp16-vs-bf16.md](gemm-fp16-vs-bf16.md) — FP16 vs BF16 cross-batch diff: systematic comparison (batch + non-batch)
- [gemm-fp32-vs-lowprec-accuracy.md](gemm-fp32-vs-lowprec-accuracy.md) — FP32 reference vs FP16/BF16: 0.036% vs 0.288%, bf16 is 8× worse, error independent of M/K
- [sgemm-ladder-h100.md](sgemm-ladder-h100.md) — Hand-written SGEMM ladder on H100: 5.3→34.8 TFLOPS, autotune lessons

### Beyond GEMM
- [beyond-gemm-kernel-landscape.md](beyond-gemm-kernel-landscape.md) — 7 kernel frontiers: Flash Attention, paged KV, GEMV, SWA, DSA MLA, allocators, fusion
- [paged-attention-tradeoff.md](paged-attention-tradeoff.md) — Non-contiguous KV: coalescing loss vs memory savings (10-15× batch), why the trade is always worth it

### Op Determinism
- [op-determinism.md](op-determinism.md) — Softmax, reduction, RMSNorm, LayerNorm determinism

### Precision tradeoffs
- [training-vs-inference-bf16.md](training-vs-inference-bf16.md) — Why training uses BF16 but inference prefers FP16
- [training-fp32-master-weights.md](training-fp32-master-weights.md) — Why FP32 master weights are mandatory (not convention)
- [fp8-fp4-training.md](fp8-fp4-training.md) — FP8 and NVFP4 training: how it works, microscaling, why 4-bit might work
- [fp4-training-progress.md](fp4-training-progress.md) — NVFP4 training status: who's doing it, what's proven, what's missing (2026-08)
- [fp8-gemm-hardware.md](fp8-gemm-hardware.md) — FP8 GEMM on H100: Tensor Core native, CUDA core emulation, memory path
- [fp8-gemm-chaining.md](fp8-gemm-chaining.md) — FP8 activation chaining between layers: scaling, epilogue fusion, framework behavior
- [fp8-gemm-accuracy-scaling-modes.md](fp8-gemm-accuracy-scaling-modes.md) — FP8 accuracy vs FP32 (3.75%), full scaling-mode support matrix, DeepGEMM/aiter blockwise recipes, why 4% per-GEMM noise ≈ lossless task accuracy

### Hardware
- [hardware-h100.md](hardware-h100.md) — NVIDIA H100 Tensor Core, cuBLAS, TF32
- [hardware-mi325x.md](hardware-mi325x.md) — AMD MI325X MFMA, rocBLAS
