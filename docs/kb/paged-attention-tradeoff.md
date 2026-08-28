# Paged Attention — Non-Contiguous KV Performance Trade-Off

## The question

Paged attention (vLLM) scatters KV cache across non-contiguous pages via a block table. Does the loss of memory contiguosity hurt performance?

**Short answer: yes, but the memory savings enable 10-15× larger batch sizes, which dwarfs the 20-30% coalescing efficiency loss.**

## How non-contiguous hurts

### 1. Coalescing broken

Traditional contiguous KV: one warp reads consecutive addresses → 1 transaction (128-byte cache line), perfect coalescing.

Paged KV: threads in the same warp may read different pages at different physical addresses. A 128-byte cache line only covers one page. Two threads hitting two pages = 2 transactions instead of 1.

### 2. Indirect indexing = serial dependency

```
// Traditional: address computed at compile time
k = kv_cache[seq_offset + head * stride + i]

// Paged: 3-step pointer chase
page_idx = block_table[seq][i // block_size]   // step 1: read table
block_base = page_table[page_idx]               // step 2: get physical base
k = kv_cache[block_base + (i % block_size)]     // step 3: read data
```

GPU memory pipeline wants many independent loads in-flight. Pointer chasing makes each load depend on the previous one's result, breaking the pipeline.

### 3. Tile fragmentation at page boundaries

Traditional attention can freely choose tile size (e.g. 64×128) because KV is contiguous. Paged attention's inner loop must stop every `block_size` tokens (default 16) to look up the next page. This fragments the loop structure and makes large-tile amortization harder.

## Why it's still worth it

### A. Page size amortizes lookup overhead

vLLM default `block_size = 16`. One page = 16 tokens × num_heads × head_dim. For Gemma 4: 16 × 8 × 256 = 32KB per page. One load fills multiple cache lines. The page boundary lookup cost is amortized over 16 tokens — small per-token overhead.

- block_size=1 → deadly (lookup per token)
- block_size=16 → sweet spot (amortized, still flexible allocation)
- block_size=256 → near-contiguous, defeats memory savings

### B. Decode is memory-bound, not coalescing-bound

Paged attention's coalescing loss matters most in **prefill** (large M, compute-bound). But paged attention's primary use case is **decode** (M=1, memory-bound).

Decode per-request KV read: ~8K context × 8 heads × 256 dim × 2 (K+V) × 2 bytes (BF16) ≈ 64MB. Batch 32 = 2GB. H100 HBM 3TB/s → 0.67ms. Even 30% coalescing loss → 0.87ms vs 0.67ms.

The real bottleneck isn't coalescing efficiency — it's whether you have enough batch to saturate bandwidth. Paged attention lets you batch more requests (memory savings), which **increases** overall throughput.

### C. Memory savings dwarf coalescing loss

| Dimension | Traditional contiguous | Paged |
|---|---|---|
| Coalescing efficiency | 100% | ~70-80% |
| Memory waste | ~90%+ (pre-allocate max len) | ~0% (allocate on demand) |
| Max batch size | memory-limited | 10-15× more |
| Indirect indexing overhead | 0 | 1 lookup per 16 tokens |
| Overall throughput | baseline | far higher (batch multiplies) |

Traditional pre-allocates max context length per request. If max=8K but actual avg=500, 94% of KV memory is wasted. Paged eliminates this waste entirely.

## Analogy to GPU capacity math

This trade-off mirrors the hard rule in MEMORY: **"Memory is binding constraint for scaling: GPU count overstates capacity 5-10× for memory-heavy pods."** Paged attention trades coalescing efficiency (compute-adjacent) for memory efficiency (the actual binding constraint). In LLM serving, memory is always the binding constraint, so this trade is always worth it.

## Mitigation strategies in production

- **Block table sorting**: vLLM sorts block tables so adjacent sequences get physically adjacent pages where possible, partially restoring coalescing
- **FlashAttention with paging**: FlashAttention's tiling structure naturally accommodates page boundaries — each K/V tile load can come from a different page
- **Larger block_size for prefill, smaller for decode**: some implementations adaptively choose block size by phase

## Related
- vLLM paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- SGLang alternative: RadixAttention (prefix-tree-based KV reuse, different paging strategy)
- `gemm-variant-taxonomy.md` — D3 shape regime (decode = GEMV, M=1)
