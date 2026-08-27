# Kernel Deep Dive — Roadmap

## Why

After reviewing Mike Martin's SGLang fork (Gemma 4 31B on AMD MI325X), realized how little I hands-on understand about inference kernel internals — attention variants, FFN tuning, flash attention, paged attention, SWA, prefill vs decode, re-prefill. Goal: write and tune every major inference kernel myself, develop sense for trade-offs, configuration tuning, and what makes things really fast.

End goal: be one of ~10-15 people who can write DSA sparse MLA decode kernels from scratch.

## Language choices

- **PyTorch** — reference implementations / correctness checking
- **Triton** — GPU kernels, auto-tiling, easy to iterate
- **CUDA / HIP C++** — low-level, maximum control, profiling (ncu/rocprof)
- **Mojo** — model architecture + selective kernel dispatch (Phase 7, stretch)

## Existing work in this repo

- `batch_softmax/` — 8 variants (naive → online → warp → multi-warp → hybrid → cudnn)
- `reduce/` — sum, max, atomic
- `vector_init.cu`, `vector_triton.py` — elementwise
- `docs/ncu-profiling-2026-05-30.md` — ncu profiling template
- SGEMM optimization kernels (siboehm.com ladder)
- Triton + TinyGrad comparison benchmarks

## KB notes

- `~/code/kb/cs/gpu-kernel-programming/` — fundamentals, GEMM ladder, techniques
- `~/code/kb/opensource/sglang/dsv4-mla-backends.md` — SGLang MLA backends
- `~/code/kb/opensource/vllm/dsv4-sparse-mla-vs-sglang.md` — vLLM vs SGLang

---

## Roadmap

### Phase 1: Foundations (primitives)

Goal: understand GPU memory hierarchy, thread hierarchy, and how to reason about kernel performance.

| # | Component | Variants to try | Languages | Key concepts |
|---|---|---|---|---|
| 1.1 | **Vector add / elementwise** | naive, coalesced, vectorized load | CUDA, Triton, PyTorch | memory coalescing, warp, grid-stride loop |
| 1.2 | **Reduction** (sum/max) | naive, warp shuffle, multi-warp, two-pass, tree | CUDA, Triton | warp-level primitives, divergence, occupancy |
| 1.3 | **Softmax** | naive, online softmax, warp-tiled, multi-warp, CUDNN | CUDA, Triton | online algorithm, numerical stability, tiling |
| 1.4 | **GEMM** (matmul) | naive, shared-memory tiled, split-K, tensor core (wmma) | CUDA, Triton, HIP | shared memory, tiling, bank conflicts, tensor cores |
| 1.5 | **GEMV** (matvec) | naive, warp-reduce, split-K | CUDA, Triton | memory-bound vs compute-bound, M=1 special case |

**Status**: 1.1 ✅, 1.2 ✅, 1.3 ✅ (8 variants), 1.4 🔄 (SGEMM ladder in progress, at warptile), 1.5 ❌

### Phase 2: FFN (Feed-Forward Network)

Goal: understand how to tune GEMM for different shapes, and how activation functions interact.

| # | Component | Variants to try | Key concepts |
|---|---|---|---|
| 2.1 | **Dense FFN** (gate × up → silu → down) | fused vs separate GEMM, activation fusion, split-K tuning | shape-dependent tuning, M/N/K vs GEMM efficiency |
| 2.2 | **FFN shape tuning** | benchmark same FFN at M=1, 8, 32, 128, 512, 2048. Find crossover points. | prefill (large M) vs decode (M=1) optimization |
| 2.3 | **Fused activation GEMM** | GEMM+SiLU fused, GELU fused, compare vs unfused | kernel fusion, launch overhead |
| 2.4 | **MoE FFN** | expert routing, scattered GEMM, grouped GEMM | MoE dispatch, memory layout for expert weights |

### Phase 3: Attention — Core

Goal: write every attention variant from scratch, understand the math and the engineering.

| # | Component | Variants to try | Key concepts |
|---|---|---|---|
| 3.1 | **Naive attention** | Q×K→S→softmax→S×V. Full materialization. | O(N²) memory, why this is bad |
| 3.2 | **Flash Attention** (tiling) | forward + backward. Tile Q, loop over K/V blocks. | online softmax + tiling, SRAM hierarchy, IO-awareness |
| 3.3 | **Flash Attention v2** | improve parallelism (split K loop across thread blocks), reduce shared memory | work partitioning, occupancy tuning |
| 3.4 | **Grouped-Query Attention (GQA)** | Q heads > KV heads, broadcast or expand strategy | GQA ratio impact, memory savings |
| 3.5 | **Multi-Head Attention (MHA)** | standard, compare with GQA/MQA | baseline for comparison |

### Phase 4: Attention — KV Cache Management

Goal: understand how paged attention works, and how cache layout affects kernel design.

| # | Component | Variants to try | Key concepts |
|---|---|---|---|
| 4.1 | **Paged attention** (prefill) | block table lookup, paged Q×K with non-contiguous V blocks | page table, block-level indirection |
| 4.2 | **Paged attention** (decode) | single-token query, paged KV, single-warp per head | decode-specific optimization, M=1 GEMV with paged KV |
| 4.3 | **Prefix cache hit** | KV from different request, different block layout | cache hit path, block reuse |
| 4.4 | **KV cache quantization** (fp8/int8) | quantized KV, dequant in-kernel, compare accuracy/speed | on-the-fly dequant, mixed precision |

### Phase 5: Attention — Sliding Window & Hybrid

Goal: deeply understand SWA, the receptive-field pyramid, and hybrid model serving.

| # | Component | Variants to try | Key concepts |
|---|---|---|---|
| 5.1 | **Sliding window attention** | naive (masked full), efficient (only load window), ring buffer | window masking, memory savings |
| 5.2 | **SWA prefill** | full prefill, partial prefill (last W tokens), re-prefill after eviction | prefill strategies, correctness vs approximation |
| 5.3 | **SWA re-prefill** (the pyramid) | flat re-prefill (current approach) vs staircase (exact) — implement both, measure quality difference | receptive-field pyramid, faithful region shrinkage |
| 5.4 | **Hybrid SWA+FULL** | interleave SWA and FULL layers, separate KV pools, match boundary | hybrid model serving, component-specific cache |
| 5.5 | **SWA with HiCache** | host restore of FULL KV, SWA recompute, load-back latency | 3-tier cache interaction with SWA |

### Phase 6: Full Model Forward Pass

Goal: put it all together — attention + FFN + normalization + routing.

| # | Component | Variants to try | Key concepts |
|---|---|---|---|
| 6.1 | **RMSNorm** | naive, fused (weight+norm in one kernel), warp-level | norm kernel, numerical precision |
| 6.2 | **RoPE** (rotary position embedding) | naive, fused with attention, different rope types | position encoding, complex number math on GPU |
| 6.3 | **Full transformer block** | attention + FFN + norm + residual. Prefill vs decode. | end-to-end, launch overhead, kernel fusion opportunities |
| 6.4 | **Prefill vs decode scheduling** | batched prefill, continuous batching, mixed batch | scheduler-level, batch composition |
| 6.5 | **Speculative decoding** | draft model + verify, tree attention | multi-token attention, rejection sampling |

### Phase 7: Mojo Model Repository (stretch)

Goal: use Mojo to write model architectures with selective kernel dispatch.

| # | Component | Description |
|---|---|---|
| 7.1 | Mojo basics | learn Mojo syntax, MLIR, struct/fn, SIMD types |
| 7.2 | Mojo GEMM | write GEMM in Mojo, compare with CUDA/Triton |
| 7.3 | Mojo attention | implement flash attention in Mojo |
| 7.4 | Mojo model repo | port tinyllm models to Mojo, plug in custom kernels per platform |

Note: Mojo is great for kernels, but not yet mature for scheduling/KV cache management. Focus Mojo work on model architecture + kernel dispatch, not building a full inference engine.

---

## How to use this roadmap

- Pick ONE item at a time
- Start with PyTorch reference → Triton → CUDA/HIP (progressive low-level)
- Always benchmark against framework baseline (PyTorch / vLLM / SGLang)
- Write a short note after each: what I learned, what surprised me, what trade-off I discovered
- ncu profiling for every CUDA kernel (template at `docs/ncu-profiling-2026-05-30.md`)

## Related

- Park item: `~/park/gpu-kernel-learning.md`
- Tutorial HTML: `~/code/projectrevolution/projects/2026-07-15-gpu-kernel-learning-roadmap/html-tutorial/`
- Model repo: `~/code/tinyllm` → `github.com/ziliangpeng/tinyllm`
- Mojo research: `~/park/2026-08-18_01-modular-max-stack.md` (calendar blocked 2026-09-03)
