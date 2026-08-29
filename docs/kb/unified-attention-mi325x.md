# Unified Attention — AITER vs vLLM TRITON_ATTN on MI325X (gfx942)

**As-of**: 2026-08-29 · All numbers correctness-gated (rel_err < 0.15% vs dense float32 torch
reference) before every timed run · Harness: `~/code/kernels/bench/unified-attn/bench_full_matrix.py`
(45 cases, all passed; raw data `matrix_v4.csv`)

## The design (what we're actually testing)

Continuous batching mixes **prefill** (q_len ≫ 1, compute-bound) and **decode** (q_len = 1,
memory-bound) in one step. Old two-path design launches two kernels sequentially on one stream —
decode waits for prefill despite zero data dependency, and decode-only launches under-fill the GPU.

**Unified**: one kernel, one grid, blocks for both regimes. Each block reads its own metadata
(sequence, q_len, kv_len). Not "two kernels in parallel" — one kernel, mixed blocks, no barrier.
Not "sum → max" — you win by removing barriers/launch/wave-fragmentation, not by deleting work.

Two implementations benchmarked (same algorithm: online softmax, paged KV, GQA M-packing):
- **aiter unified** — `aiter.ops.triton.attention.unified_attention`, AMD gfx-tuned configs
- **vLLM TRITON_ATTN** — `vllm.v1.attention.ops.triton_unified_attention`, generic portable config

Difference is tile/stage/split config only. SGLang's unified path calls the SAME aiter function.

## Headline

**AITER wins 45/45 correctness-gated cases on gfx942: 1.42×–18.31×.** This contradicts the early-2026
"AITER is usually slower" narrative (that was pre-retune, on gfx950, with head_dim 256/512 LDS
crashes — fixed by config-only retunes #4044/#4918). On gfx942 the vLLM generic config has no
gfx942-specific tiers, so the gap is even larger.

## Results (hq=16, d=128, blk=32, fp8-e4m3 KV)

### Decode (sq=1) — gap shrinks as batch saturates

| bs | sk | aiter ms | vllm ms | speedup |
|---|---|---|---|---|
| 8 | 8192 | 0.0431 | 0.4134 | 9.6× |
| 16 | 16384 | 0.0452 | 0.8198 | 18.1× |
| 32 | 8192 | 0.0485 | 0.4160 | 8.6× |
| 64 | 32768 | 0.1880 | 1.6446 | 8.7× |
| 128 | 8192 | 0.1172 | 0.4246 | 3.6× |
| 256 | 4096 | 0.1192 | 0.3215 | 2.7× |

Low-batch decode is vLLM's weakest corner: ~86GB/s effective vs ~6TB/s peak (serial KV-tile chain,
num_stages=1). aiter's gfx-tiered config double-buffers.

### Skew experiment (256 decode, same total KV bytes) — the production amplifier

| distribution | aiter ms | vllm ms | ratio |
|---|---|---|---|
| uniform 8k | 0.193 | 0.628 | 3.2× |
| heavy-tail 128–65k | 0.364 | 3.886 | **10.7×** |
| bimodal 1k/32k (80/20) | 0.209 | 1.668 | **8.0×** |
| one 65k + 255×1k | 0.178 | 3.255 | **18.3×** |

**Skew amplifies the gap; bytes don't.** One 65K straggler serializes vLLM's fixed-tile loop; aiter's
split-KV absorbs it. Production traffic is bimodal (long docs + short turns) → expect ~8×+ in reality.
aiter's skew time stays flat (0.36 vs 0.34 uniform).

### Context sweep (decode bs=64) — gap grows with context

| ctx | ratio |
|---|---|
| 1K | 1.4× |
| 8K | 5.3× |
| 32K | 8.2× |
| 64K | **10.0×** |

vLLM degrades linearly with context; aiter is sub-linear (split-KV parallelizes the straggler).
Both hit the launch/latency floor at 1K.

### Prefill granularity — aiter robust to phase mix

 aitер stays 0.44ms from 1→8 requests at fixed budget; ≤128-token chunks hurt even aiter (per-seq
overhead). Across prefill-heavy → decode-heavy mixes, aiter stays 0.70–0.80ms while vLLM swings
1.5→3.7ms — low variance matters for ITL stability.

## E2E: vLLM vs SGLang (same GPU, Qwen3-1.7B bf16, 96 prompts, conc 32, warmed)

| | vLLM (TRITON_ATTN vs AITER_UNIFIED) | SGLang (ragged vs unified) |
|---|---|---|
| short 512/256 | 5205 → 4934 (−5%) | 5474 → 5384 (−1.7%) |
| long 16384/256 | 373 → **865 (+132%)** | 3726 → 3728 (parity) |
| TPOT long | 4.1 → 3.8ms | 5.99 → 4.84ms |

**Key synthesis — e2e delta = kernel gap × baseline weakness.** vLLM's generic Triton baseline is
the gfx942 outlier (4–10× kernel gaps), so unified "wins" +132% there. SGLang's CK baseline is
already in the fast tier → parity, with unified's decode (TPOT) still 19% faster. **Never
extrapolate e2e gains across frameworks** — the +132% measures "how weak generic portable Triton is
on gfx942", not "how magic unified is".

**SGLang hard gotchas**: (1) `--page-size 32` is MANDATORY for the unified path — sglang default
page_size=1 makes block_table 8192 entries/seq → unified e2e collapses 4.9× (747 tok/s). sglang
auto-sets page_size=16 for Qwen3VL+unified, 64 for DSA — but NOT generic MHA. (2)
`SGLANG_USE_AITER_UNIFIED_ATTN` doesn't exist in v0.5.9 images (sglang 0.5.10) — silently inert.
(3) v0.5.9 images bundle aiter 0.1.10 (Feb 2026, pre-retune) — 20–156% slower on same shapes than
v0.5.17's aiter 0.1.20 (Aug 2026). **The bundled aiter snapshot is the true perf dependency, more
than the framework version.**

## Methodology (how to benchmark any inference kernel)

1. **Inputs by hand** — query `randn*0.3`→bf16 (scaled so softmax isn't one-hot); paged pool
   `(num_blocks, blk, kv_heads, hs)`; metadata exactly what the scheduler feeds: `block_table`
   (int32 page ids per seq), `cu_seqlens` (cumsum q_lens = packed-Q offsets), `seqused_k` (KV lens).
2. **Correctness gate before EVERY timed run** — dense float32 torch reference: gather paged KV via
   block_table → GQA repeat_interleave → QKᵀ·scale → causal mask (row s sees kv[: kl−ql+s+1]) →
   softmax → PV. Gate rel_err<0.15%, BAD → abort. **Never publish un-gated numbers.**
3. **`triton.testing.do_bench(fn, warmup=25, rep=100)`** — warmup is in milliseconds (JIT compile +
   clock ramp land here); 100ms window, median; L2 auto-flushed between iterations (critical for
   memory-bound kernels — else 2–10× inflated).
4. **Identical tensors to both kernels**; timing symmetric; A/B within one process.
5. **E2E layer**: serve per arm (backend at startup) → verify from logs which backend loaded → warm
   mixed shapes manually (built-in JIT warmup covers decode-only shapes only) → bench → compare
   percentiles side-by-side, never single-arm.

### Gotchas hit (all real, will recur)
- vLLM standalone `unified_attention` needs explicit `kv_quant_mode=KVQuantMode.FP8_PER_TENSOR` for
  fp8 KV — default silently skips dequant → 49× wrong, no error. Gate caught it.
- `VLLM_ATTENTION_BACKEND` env var is ignored by vLLM (unknown-env warning) — both arms silently ran
  ROCM_ATTN. Must use `--attention-backend` flag.
- vLLM JIT warmup covers decode-only shapes only ("finished in 0.00s") — mixed shapes JIT on first
  encounter, poisoning TTFT.
- pa_ragged (CK) cannot be benched with hand-built metadata — needs sglang's real req_to_token pool.
- `num_blocks = sum(ceil(kl/blk))`, not `sum(kl)//blk` — off-by-one at large skew.
- Background servers in `kubectl exec` get reaped at exec exit — use detached script + sentinel file.

## Conclusions

1. **Correctness is binary; performance is continuous** — two bit-comparable kernels 10× apart is
   normal; config, not math, decides.
2. **Skew is the production amplifier** — uniform same-bytes 3.3× vs skewed 10.6×.
3. **e2e delta = kernel gap × baseline weakness** — vLLM +132% vs SGLang parity proves the point.
4. **Tuning ownership wins the fast tier** — aiter tracks gfx tuning continuously; vLLM generic
   config lags. Both converge over time (upstream ports aiter's retunes).
5. **The bundled aiter snapshot is the real dependency** for sglang users — check it, not the
   framework version.
6. **Production-flip gate (per Sol)**: ≥5% e2e at matched SLO, zero-JIT window, ABBA interleaved
   ≥5 rounds ≥1000 reqs, no >2% p99 regression, canary 1–5% one traffic cycle.

## See also
- [paged-attention-tradeoff.md](paged-attention-tradeoff.md) — paged KV coalescing vs memory trade
- [beyond-gemm-kernel-landscape.md](beyond-gemm-kernel-landscape.md) — attention kernel frontiers
- `~/code/agent-kb/vllm/unified-attention-kernel-benchmark-aiter-vs-triton.md` +
  `sglang-aiter-unified-attention-ab.md` (agent-kb copies, ops-focused)
- Harnesses: `~/code/kernels/bench/unified-attn/` (bench_full_matrix.py, bench_sglang_ab.py,
  matrix_v4.csv)
- Upstream: ROCm/aiter PRs #4044, #4918, #4761, #5004 · SGLang PRs #31856, #23146, #20897
