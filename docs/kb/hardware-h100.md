# NVIDIA H100

## Specs
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.6
- PyTorch: 2.7.1+cu126
- 80GB HBM3

## Tensor Core
- Tile size: 16×16×16 (wmma), hardware-fixed
- FP16/BF16 input → FP32 accumulator (mixed precision)
- TF32 mode: 19-bit mantissa (vs FP32 23-bit) — trades precision for throughput
- wmma API: `wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)`

## cuBLAS
- Standard GEMM: `sgemm` (FP32), `hgemm` (FP16), `bgemm` (BF16)
- Batched GEMM: `cublasGemmStridedBatchedEx` — uses fixed kernel for all B when inner dims constant (FP16/BF16 bit-exact)
- cuBLASLt: heuristic algorithm selection, can return multiple algo candidates
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — forces deterministic workspace, but does NOT fix cross-batch algo switching

## TF32
- Enabled by default in PyTorch (`torch.backends.cuda.matmul.allow_tf32 = True` since PyTorch 1.12)
- Amplifies cross-batch non-determinism ~630× (0.0001% → 0.07%)
- Uses Tensor Core with reduced mantissa (19 vs 23 bit)
- Only affects FP32 matmul — FP16/BF16 unaffected

## Algorithm selection
- cuBLAS selects algorithm based on M, N, K, dtype, layout
- Different M → different tiling → different accumulation order → not bit-exact across M
- FP32 has largest algorithm space (software tiling, no fixed hardware tile)
- FP16/BF16 constrained by Tensor Core fixed tile (16×16×16) → fewer variants → more deterministic
