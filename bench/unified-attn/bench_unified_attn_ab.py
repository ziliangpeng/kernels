"""Kernel-level A/B benchmark: AITER unified attention vs vLLM TRITON_ATTN unified.

Both kernels share the same interface (paged KV + cu_seqlens). Correctness is
checked against a torch reference, then timed with triton.testing.do_bench.

Covers:
  - dtype matrix: bf16/bf16, bf16 Q + fp8 KV, fp8 Q + fp8 KV (per-tensor descales)
  - workload shapes: pure decode, pure prefill, mixed prefill+decode batches
  - varied prefill lengths and per-seq KV lengths
  - prefix-cache "hit rate": prefill KV reuse is emulated by giving each prefill
    request a pre-computed KV of (ctx_len) tokens and only computing attention
    over the new tokens (that is what chunked prefill does in vLLM).

Usage (on GPU pod):
  python3 bench_unified_attn_ab.py --suite decode
  python3 bench_unified_attn_ab.py --suite prefill
  python3 bench_unified_attn_ab.py --suite mixed
  python3 bench_unified_attn_ab.py --suite all --check
"""
import argparse
import itertools
import math

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention as aiter_unified
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as vllm_unified

E4M3 = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn


# ---------------------------------------------------------------- inputs

def make_inputs(query_lens, kv_lens, num_q_heads, num_kv_heads, head_size,
                block_size, q_dtype, kv_dtype, seed=0):
    device = "cuda"
    num_seqs = len(query_lens)
    max_kv_len = max(kv_lens)
    num_blocks = sum((kl + block_size - 1) // block_size for kl in kv_lens) + num_seqs + 16
    max_kv_len = max(kv_lens)

    torch.manual_seed(seed)
    query = (torch.randn(sum(query_lens), num_q_heads, head_size,
                         dtype=torch.float32, device=device) * 0.3).to(q_dtype)
    key_cache = (torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                             dtype=torch.float32, device=device) * 0.3).to(kv_dtype)
    value_cache = (torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                               dtype=torch.float32, device=device) * 0.3).to(kv_dtype)

    # per-tensor descales (vLLM uses calibrated scales; aiter test uses random in [1e-4, 1])
    q_descale = k_descale = v_descale = None
    if q_dtype != torch.bfloat16:
        q_descale = torch.tensor([0.02], dtype=torch.float32, device=device)
    if kv_dtype != torch.bfloat16:
        k_descale = torch.tensor([0.02], dtype=torch.float32, device=device)
        v_descale = torch.tensor([0.02], dtype=torch.float32, device=device)

    cu = [0]
    bt_rows = []
    bt_rows_needed = (max_kv_len + block_size - 1) // block_size
    free_block = num_seqs + 1
    for ql, kl in zip(query_lens, kv_lens):
        nblk = (kl + block_size - 1) // block_size
        bt = list(range(free_block, free_block + nblk))
        free_block += nblk
        bt += [0] * (bt_rows_needed - nblk)
        bt_rows.append(bt)
        cu.append(cu[-1] + ql)
    block_table = torch.tensor(bt_rows, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return query, key_cache, value_cache, block_table, cu_seqlens, seq_lens_t, q_descale, k_descale, v_descale


def ref_attention(query, key_cache, value_cache, block_table, cu, kv_lens, query_lens,
                  nq, nk, hs, bs, scale, q_descale, k_descale, v_descale, out_dtype):
    """Dense torch reference with FP8 dequant support (float32)."""
    outs = []
    for i, (ql, kl) in enumerate(zip(query_lens, kv_lens)):
        start = cu[i]
        q = query[start:start + ql].float()
        if q_descale is not None:
            q = q * q_descale
        nblk = (kl + bs - 1) // bs
        k = torch.cat([key_cache[b].float() for b in block_table[i][:nblk]], dim=0)[:kl]
        v = torch.cat([value_cache[b].float() for b in block_table[i][:nblk]], dim=0)[:kl]
        if k_descale is not None:
            k = k * k_descale
            v = v * v_descale
        rep = nq // nk
        k = k.repeat_interleave(rep, dim=1).transpose(0, 1)  # (nq, kl, hs)
        v = v.repeat_interleave(rep, dim=1).transpose(0, 1)
        offset = kl - ql
        for s in range(ql):
            scores = torch.einsum("hd,htd->ht", q[s], k) * scale
            scores[:, offset + s + 1:] = float("-inf")
            p = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("ht,htd->hd", p, v).reshape(-1))
    return torch.stack(outs).to(out_dtype)


# ---------------------------------------------------------------- suites

def suite_decode():
    """Pure decode sweeps: batch size x kv len."""
    cases = []
    for bs, sk in [
        (8, 8192), (32, 8192), (64, 16384), (128, 8192), (256, 4096), (64, 32768),
    ]:
        cases.append(dict(name=f"decode bs={bs} sk={sk}", query_lens=[1] * bs,
                          kv_lens=[sk] * bs))
    return cases


def suite_prefill():
    """Prefill: total prefill tokens held ~constant, vary request lengths."""
    cases = []
    for lens in [[2048], [512] * 4, [256] * 8, [128] * 16, [64] * 32,
                 [1024, 512, 256, 256]]:  # homogeneous vs skewed
        cases.append(dict(name=f"prefill {lens}", query_lens=lens,
                          kv_lens=[8192] * len(lens)))
    # long context single
    cases.append(dict(name="prefill [8192] ctx=16k", query_lens=[8192],
                      kv_lens=[16384]))
    return cases


def suite_mixed():
    """Mixed batches: vary decode:prefill ratio at ~4k-token batch budget."""
    cases = []
    for n_pre, pre_len, n_dec in [
        (1, 2048, 32), (1, 2048, 128), (4, 512, 32), (8, 256, 64),
        (2, 1024, 256), (1, 4096, 64), (8, 128, 128),
    ]:
        ql = [1] * n_dec + [pre_len] * n_pre
        kl = [8192] * (n_dec + n_pre)
        cases.append(dict(name=f"mixed pre={n_pre}x{pre_len} dec={n_dec}",
                          query_lens=ql, kv_lens=kl))
    return cases


def suite_hitrate():
    """Prefix-cache hit emulation: prefill requests with different ctx reuse.
    Longer reused context = fewer new tokens to compute for same KV footprint."""
    cases = []
    for ctx, new in [(0, 1024), (256, 768), (512, 512), (768, 256), (896, 128)]:
        n = 8
        # query = new tokens only; KV = ctx (cached) + new
        cases.append(dict(
            name=f"hit-rate {ctx}/{ctx+new}",
            query_lens=[new] * n,
            kv_lens=[ctx + new] * n,
        ))
    return cases


def suite_random():
    """Randomized production-like batches: heavy-tail lengths, mixed phases,
    skewed concurrency. Each case = one draw from a realistic serving pool."""
    import random as _rnd
    rng = _rnd.Random(42)
    cases = []
    for draw in range(8):
        n_dec = rng.choice([32, 64, 96, 128, 192, 256])
        n_pre = rng.choice([1, 2, 3, 4, 6, 8, 12])
        # prefill lengths: heavy-tail (lots of small chunks + few long docs)
        pre_lens = []
        for _ in range(n_pre):
            if rng.random() < 0.25:
                pre_lens.append(rng.randint(1024, 4096))   # long doc chunk
            else:
                pre_lens.append(rng.randint(64, 512))       # small turn / chunked resume
        # KV lengths: heavy-tail too (short chats + long-context docs)
        def draw_kv():
            if rng.random() < 0.2:
                return rng.randint(16384, 65536)            # long-context user
            if rng.random() < 0.5:
                return rng.randint(1024, 8192)              # typical chat
            return rng.randint(128, 1024)                   # short turns
        ql = [1] * n_dec + pre_lens
        kl = [draw_kv() for _ in range(n_dec)] + [max(draw_kv(), pl) for pl in pre_lens]
        total = sum(ql)
        cases.append(dict(
            name=f"rand#{draw} dec={n_dec} pre={n_pre}({','.join(map(str, pre_lens))}) tok={total}",
            query_lens=ql, kv_lens=kl,
        ))
    return cases


def suite_sweep():
    """Controlled 2-factor sweep: decode-heavy vs prefill-heavy at fixed budgets."""
    cases = []
    # axis 1: same total query tokens (4096), vary granularity & phase mix
    for n_pre, pre_len, n_dec, sk in [
        (16, 256, 0, 8192),    # pure prefill, fine chunks
        (8, 512, 0, 8192),
        (4, 1024, 0, 8192),
        (8, 512, 64, 8192),    # 2k prefill + 64 dec
        (4, 512, 128, 8192),   # 2k prefill + 128 dec
        (2, 512, 256, 8192),   # 1k prefill + 256 dec
        (1, 512, 512, 8192),   # decode-dominated
        (0, 0, 512, 8192),     # pure decode
        (4, 512, 64, 16384),   # long ctx variant
        (4, 512, 64, 32768),   # very long ctx
        (4, 512, 64, 65536),   # extreme ctx (fit check)
    ]:
        ql = [1] * n_dec + [pre_len] * n_pre
        kl = [sk] * (n_dec + n_pre)
        label = f"pre={n_pre}x{pre_len} dec={n_dec} sk={sk}"
        if not ql:
            continue
        cases.append(dict(name=label, query_lens=ql, kv_lens=kl))
    return cases


# ---------------------------------------------------------------- runner

def run_case(case, args):
    query_lens = case["query_lens"]
    kv_lens = case["kv_lens"]
    query, key_cache, value_cache, block_table, cu_seqlens, seq_lens_t, q_des, k_des, v_des = make_inputs(
        query_lens, kv_lens, args.hq, args.hk, args.d, args.block_size, args.q_dtype, args.kv_dtype,
    )
    num_tokens = sum(query_lens)
    out = torch.empty(num_tokens, args.hq, args.d,
                      dtype=torch.bfloat16, device="cuda")
    scale = args.d ** -0.5

    # descale shapes differ per kernel: aiter wants per-tensor (1,);
    # vllm triton kernel wants (num_seqs, num_kv_heads) [expanded per seq]
    common = dict(
        q=query, k=key_cache, v=value_cache,
        cu_seqlens_q=cu_seqlens, max_seqlen_q=max(query_lens),
        seqused_k=seq_lens_t, max_seqlen_k=max(kv_lens),
        softmax_scale=scale, causal=True,
        window_size=(-1, -1), block_table=block_table,
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None,
    )
    if q_des is not None:
        common["q_descale"] = q_des
    if k_des is not None:
        common["k_descale"] = k_des
        common["v_descale"] = v_des
    common_vllm = dict(common)
    if k_des is not None:
        # vllm standalone wrapper defaults to KV_QUANT_MODE=NONE (silent wrong numerics!);
        # must pass FP8_PER_TENSOR explicitly
        from vllm.v1.kv_cache_interface import KVQuantMode
        common_vllm["kv_quant_mode"] = KVQuantMode.FP8_PER_TENSOR

    ref = None
    if args.check:
        ref = ref_attention(query.float().to(torch.bfloat16), key_cache, value_cache,
                            block_table, cu_seqlens.tolist(), kv_lens, query_lens,
                            args.hq, args.hk, args.d, args.block_size, scale,
                            q_des, k_des, v_des, torch.bfloat16)

    results = {}
    for name, fn, kw in [("aiter", aiter_unified, common), ("vllm", vllm_unified, common_vllm)]:
        try:
            out.zero_()
            fn(**dict(kw, out=out))
            torch.cuda.synchronize()
        except Exception as e:
            results[name] = (None, f"FAIL: {type(e).__name__}: {str(e)[:100]}")
            continue
        if args.check:
            got = out.reshape(num_tokens, -1).float()
            ref32 = ref.float()
            rel = (got - ref32).norm() / ref32.norm().clamp(min=1e-6)
            ok = "OK " if rel.item() < 0.15 else "BAD"
            print(f"  [{name}] {ok} rel_err={rel.item():.4f}")
        ms = triton.testing.do_bench(lambda f=fn, k=kw: f(**dict(k, out=out)),
                                     warmup=25, rep=100)
        results[name] = (ms, f"{ms:.4f} ms")

    a_ms, a_str = results["aiter"]
    v_ms, v_str = results["vllm"]
    ratio = f"{v_ms / a_ms:.3f}x" if (a_ms and v_ms) else "n/a"
    print(f"  aiter={a_str}  vllm={v_str}  speedup={ratio}")
    return a_ms, v_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite",
                    choices=["decode", "prefill", "mixed", "hit-rate", "random", "sweep", "all"],
                    default="all")
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hk", type=int, default=1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["bf16", "fp8kv", "fp8qkv"],
                    default="bf16", help="bf16 | fp8 KV | fp8 Q+KV")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--csv", default=None, help="append results to CSV path")
    args = ap.parse_args()

    args.q_dtype = {"bf16": torch.bfloat16, "fp8kv": torch.bfloat16, "fp8qkv": E4M3}[args.dtype]
    args.kv_dtype = {"bf16": torch.bfloat16, "fp8kv": E4M3, "fp8qkv": E4M3}[args.dtype]

    suites = {
        "decode": suite_decode, "prefill": suite_prefill, "mixed": suite_mixed,
        "hit-rate": suite_hitrate, "random": suite_random, "sweep": suite_sweep,
    }
    suite_keys = (["decode", "prefill", "mixed", "hit-rate", "random", "sweep"]
                  if args.suite == "all" else [args.suite])
    to_run = list(itertools.chain(*(suites[s]() for s in suite_keys)))

    print(f"suite={args.suite} dtype={args.dtype} h={args.hq}/{args.hk} d={args.d} blk={args.block_size}")
    rows = []
    for case in to_run:
        print(f"\n[{case['name']}]")
        a, v = run_case(case, args)
        rows.append((case["name"], a, v))

    if args.csv:
        import csv as csvmod
        with open(args.csv, "a", newline="") as f:
            w = csvmod.writer(f)
            if f.tell() == 0:
                w.writerow(["case", "dtype", "hq", "hk", "d", "block", "aiter_ms", "vllm_ms", "ratio"])
            for name, a, v in rows:
                if a and v:
                    w.writerow([name, args.dtype, args.hq, args.hk, args.d, args.block_size,
                                f"{a:.4f}", f"{v:.4f}", f"{v / a:.3f}"])
        print(f"\nsaved -> {args.csv}")


if __name__ == "__main__":
    main()
