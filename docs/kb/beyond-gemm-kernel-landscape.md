# Beyond GEMM — The Kernel Landscape

GEMM is the best learning starting point because it has one variable (tiling). But "interesting" kernels are interesting precisely because they introduce dimensions GEMM doesn't have.

## What GEMM teaches

GEMM teaches **"same algorithm, how to find reuse at each memory hierarchy level."** All GEMM variants are the same math (C = A×B) executed differently. The optimization space is purely tiling: HBM tile → SMEM tile → register tile → tensor core fragment.

## What GEMM does NOT teach

| Challenge type | GEMM has it? | Representative kernel |
|---|---|---|
| Tiling / memory reuse | ✅ core | GEMM ladder |
| Algorithm redesign for SRAM | ❌ | Flash Attention |
| Data-dependent memory access | ❌ | Paged Attention |
| Memory-bound (no reuse possible) | ❌ | GEMV / decode |
| Receptive field math | ❌ | SWA pyramid |
| System-level composition | ❌ | DSA sparse MLA |
| Resource management | ❌ | Custom allocator |
| Fusion granularity trade-off | ❌ | Fused epilogue |

## The 7 kernel frontiers beyond GEMM

### 1. Flash Attention — algorithm redesign for SRAM

Naive attention: `S = Q×K^T → softmax(S) → O = S×V`. Problem: S is O(N²), doesn't fit in SRAM.

Flash Attention's insight: **online softmax**. You don't need the entire row of S to compute softmax. Process K/V in blocks, maintain running max + running sum, rescale at the end. Rewrites O(N²) SRAM algorithm into O(N) SRAM.

This is NOT tiling optimization (like GEMM ladder). It's **algorithm-level re-derivation to fit SRAM**. Different level of thinking entirely.

Key concept: online softmax (also used in `batch_softmax/` — 8 variants already written in this repo).

### 2. Paged Attention — pointer chasing on GPU

GEMM memory access is deterministic: `A[row][col]`, fixed stride. Paged attention access is **data-dependent**: each request's KV cache is scattered across pages, must query block table before reading.

Challenges:
- Indirect indexing breaks memory coalescing
- Variable sequence length → variable loop count per request in batch
- Block table lookup is serial dependency (read table → read data)

See `paged-attention-tradeoff.md` for detailed analysis.

### 3. GEMV / Decode — extreme memory-bound

GEMM is compute-bound (large M). GEMV (M=1) is memory-bound: each weight read once, no reuse possible.

Completely inverts optimization intuition:
- GEMM: reduce memory reads (tile + reuse) → faster
- GEMV: memory reads are mandatory, only optimization is **bandwidth saturation** — fill all memory channels, use max vector width, hide latency with pipeline

Key concept: roofline model. When arithmetic intensity < ops:byte ratio, you are always memory-bound. No compute optimization helps.

### 4. Sliding Window Attention — receptive field math

Not a tiling problem — a **receptive field math problem**. Stack N layers of SWA, each layer shrinks faithful region by W−1. At sufficient depth, flat re-prefill degrades to full recompute.

Key finding (from MEMORY): Gemma 4 31B has 50 SWA + 10 FULL layers, sliding_window=1024. Faithful region = 1024 + 1023×49 = 51,151 tokens > 15k context → staircase degrades to full recompute. None of SGLang/vLLM/TRT-LLM implement correct staircase.

This is "understand what the correct computation should be, then find an efficient way" — not "write a fast kernel."

### 5. DSA Sparse MLA — the composite boss

Combines ALL challenges above:
- Attention (online softmax + tiling — Flash Attention logic)
- Paged KV (indirect indexing, block table lookup)
- Sparse selection (top-k routing, data-dependent branching)
- Multi-head latent (compressed KV, projection GEMM mixed into attention)
- Decode shape (M=1, memory-bound)

Goal: be one of ~10-15 people who can write this from scratch. This is a kernel SYSTEM, not a single kernel — each component has its own optimization challenge, and they must compose.

### 6. Custom Memory Allocator — resource management

Not a compute kernel — a **memory management kernel**. `cudaMalloc` is synchronous and kills inference loops. Custom allocator must:
- Maintain free list (lock-free queue?)
- Support CUDA graph capture (graph-safe: no alloc during capture)
- Manage fragmentation

Teaches that GPU programming isn't just "compute fast" — it includes resource management. See triton-kernels skill for custom allocator integration patterns.

### 7. Kernel Fusion — macro-level optimization

GEMM is one op. Real models are a chain: norm → QKV → attention → projection → FFN → norm. Each op writes to HBM then reads back.

Fusion question: why not fuse norm + GEMM? Because fusion has cost — fused kernel can't reuse cuBLAS optimized GEMM, must write custom epilogue. Fusion makes kernel more complex, harder to tune.

Key concept: **granularity trade-off**. Fine-grained (one kernel per op, use library) vs coarse-grained (fuse multiple ops, write yourself). Triton exists to make fusion easier to write.

## Progression recommendation

**Original progression (now revised per GPT-5.6 Sol review, 2026-08-28):**

1. **Finish GEMM ladder** (D1 complete: warp-tile → double buffer → light WMMA benchmark)
2. **GEMV** (natural extension — same matmul but M=1, teaches roofline)
3. **Warp reduction & online softmax** (Sol-flagged missing frontier — the bridge from primitives to attention)
4. **Dense decode attention** (contiguous KV → fused → GQA/MQA)
5. **Paged attention** (adds indirect indexing on top of decode attention)
6. **Dense MLA decode** (latent KV layout, decoupled RoPE)
7. **Sparse selection → DSA sparse MLA** (top-k gather, sparse softmax, fused end-to-end)

**Sol's key corrections to the original ordering:**

- The original (GEMM → GEMV → Flash Attention → paged → SWA → DSA) had two problems: Flash Attention as a full production-quality detour delays the decode path, and SWA was incorrectly placed as a DSA prerequisite.
- **Flash Attention: learn the algorithm (online softmax recurrence, tiled QK/PV, stable rescaling), don't chase FA-2/3 production performance.** It's mostly prefill material; the decode path matters for DSA.
- **SWA moves to a side branch** — it's a different attention sparsity pattern, not a DSA prerequisite. Valuable receptive-field math, but doesn't block the main line.
- **Missing frontiers now added**: warp-level reduction/shuffle, prefix scan, top-k/selection, gather/scatter with cache-line analysis, ragged batching, GQA/MQA→MLA data-structure evolution, multi-CTA reduction, load balancing for uneven selected-token counts, numerical stability under masked/sparse softmax, graph-safe workspace/allocator behaviour.
- **Explicitly deprioritized for the DSA goal**: DGEMM, exhaustive SGEMM autotuning, full cuBLAS dispatch reverse-engineering, all CUTLASS template features as a gate, Hopper WGMMA/TMA before attention exists (learn them when they fix a real bottleneck in your MLA kernel).

Each frontier builds on the previous. Don't skip to DSA without understanding dense decode attention and paged KV first.

## Related
- `gemm-variant-taxonomy.md` — the 6-dimension GEMM framework
- `sgemm-ladder-h100.md` — first-hand GEMM ladder data (where you are now)
- `paged-attention-tradeoff.md` — detailed paged attention analysis
- `../ROADMAP.md` — full 7-phase kernel learning roadmap
