# GEMM Variant Taxonomy — The 6 Dimensions

All GEMM variants answer one question: **"How do M×N×K arithmetic and data movement map onto the grid/block/warp/thread/tensor-core hardware structure?"** Every named variant is a coordinate in this 6-dimensional space.

## D1 — Memory Hierarchy Ladder

The most important dimension. Each rung moves data reuse one level closer to the compute unit. Same GEMM, different memory layer doing the reuse.

| Rung | Term | Key idea |
|---|---|---|
| 1 | naive | one output per thread, no reuse |
| 2 | coalesced | fix warp→memory mapping |
| 3 | SMEM tiling | block-level tile reuse in shared memory |
| 4 | 1D blocktile | register reuse of B |
| 5 | 2D blocktile | outer product, register reuse of A AND B |
| 6 | vectorized load | float4 GMEM loads, reduce instruction count |
| 7 | warp-tiled | warp-level sub-tile coordination |
| 8 | double buffering | ping-pong SMEM, overlap compute and load |
| 9 | tensor core (WMMA) | hardware matrix units, 16×16 fragment |

First-hand data on H100 FP32: naive 5.3T → vectorized 34.8T (6.6×), reaching 67% of cuBLAS. See `sgemm-ladder-h100.md`.

Key lesson: autotune finds the best within an algorithm's ceiling; only a new algorithm raises the ceiling.

## D2 — K-Axis Splitting

When M×N is too small to fill all SMs, split along K for more parallelism.

- **split-K**: K → S chunks, S blocks compute partial C, atomic add or 2nd-kernel reduce. Parallelism gain vs reduction overhead.
- **stream-K**: scheduler-driven K-chunk assignment. Fixes split-K's quantization waste (some blocks get more K than others, trailing blocks idle).
- **persistent kernel**: launch exactly #SMs blocks, each processes multiple tiles until done. Avoids launch overhead, enables work stealing.

Critical for decode-shape FFN (small M) and skinny GEMMs.

## D3 — Shape Regime

Different M/N/K ratios need fundamentally different kernel strategies.

- **GEMV** (M=1): pure memory-bound, each weight read once, optimize for bandwidth not FLOPS
- **skinny GEMM** (M<64): toggle to split-K / small tile
- **large/square GEMM**: compute-bound, standard tiling ladder
- **batched GEMM**: N independent same-shape GEMMs, one launch
- **grouped GEMM**: variable-shape GEMMs packed into one launch (MoE dispatch)

cuBLAS exposes separate APIs per regime: `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx`.

## D4 — Precision

BLAS naming convention — same algorithm, different dtype.

- **SGEMM** (FP32), **DGEMM** (FP64), **HGEMM** (FP16), **BGEMM** (BF16)
- **FP8 GEMM** (e4m3/e5m2): adds sub-dimension of scaling mode — tensorwise, rowwise, blockwise (1×128). Each has different accuracy/perf trade-offs. See `fp8-gemm-accuracy-scaling-modes.md`.
- **INT8 GEMM**: quantized weights, dequant in-kernel

## D5 — Hardware Instruction Layer

Which silicon unit does the math, at what abstraction level.

| API | Era | Control level |
|---|---|---|
| CUDA cores (FFMA) | all gen | full control, slowest |
| WMMA (`nvcuda::wmma`) | Volta–Ampere | easy but rigid fragment layout |
| mma.sync (PTX) | Ampere+ | fast, manual fragment layout (what CUTLASS uses) |
| WGMMA | Hopper | warpgroup (4 warps) cooperate, pairs with TMA |
| TMA | Hopper | hardware async DMA to SMEM, no thread involvement |
| MFMA | AMD CDNA | AMD's tensor core equivalent |
| AITER | AMD GFX950 | MI350X matrix instruction library |

Progression: WMMA = easy but rigid → mma.sync = fast but manual → WGMMA = warpgroup-level, requires TMA for max throughput.

## D6 — Implementation Stack

Who writes the kernel.

- **hand-written CUDA/HIP**: max control, max effort
- **CUTLASS**: NVIDIA's open-source C++ template framework. Makes D1×D2×D5 composable via templates (`ThreadblockShape × WarpShape × InstructionShape × Epilogue`). Each template parameter maps to a dimension above.
- **Triton**: compiler auto-tiles and auto-pipelines. Hides D2 entirely (compiler decides). `tl.dot` lowering chooses tile + pipeline strategy.
- **cuBLAS / hipBLAS**: closed-source black box. Dispatches to hundreds of internal kernels by shape/dtype.
- **TinyGrad**: minimal abstraction, generates kernels from graph

## How to use this taxonomy

When you encounter a new GEMM term, ask: which dimension?
- "ping-pong scheduling" → D1 (double buffering)
- "hopper GEMM" → D5 (WGMMA + TMA)
- "quantized GEMM" → D4 (precision) + possibly D1 (dequant in epilogue)
- "MoE GEMM" → D3 (grouped GEMM)

If you can't place it, it's likely a synonym or a combination of coordinates.

## Related
- [sgemm-ladder-h100.md](sgemm-ladder-h100.md) — first-hand D1 ladder data on H100
- [fp8-gemm-accuracy-scaling-modes.md](fp8-gemm-accuracy-scaling-modes.md) — D4 FP8 deep dive
- [hardware-h100.md](hardware-h100.md) — D5 NVIDIA hardware details
- [hardware-mi325x.md](hardware-mi325x.md) — D5 AMD hardware details
- `../tutorial/gemm-roadmap.html` — visual roadmap with all 6 dimensions
