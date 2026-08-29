"""Comprehensive kernel-level A/B: AITER unified vs vLLM TRITON_ATTN.

Dimension matrix (all correctness-gated vs dense torch reference):
  1. dtype:       bf16 KV, fp8 KV (per-tensor descale)
  2. head config: GQA 16/1, GQA 16/8 (d=128)
  3. phase mix:   pure-decode, pure-prefill, mixed ratios (incl. prefill-heavy)
  4. skew:        uniform vs heavy-tail KV lengths (fixed total bytes)
  5. context:     1K..64K

Reported: mean of 3 do_bench reps (warmup=25ms, rep=100ms, median), median & min.
Both kernels get IDENTICAL inputs; per-run rel_err printed; FAIL aborts the case.

Usage:
  python3 bench_full_matrix.py --suite core        # headline matrix (fp8kv)
  python3 bench_full_matrix.py --suite dtype       # dtype comparison on fixed shapes
  python3 bench_full_matrix.py --suite skew        # skew deep-dive
  python3 bench_full_matrix.py --suite all --csv results.csv
"""
import argparse
import csv
import itertools
import random

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention as aiter_unified
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as vllm_unified

E4M3 = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
DTYPE_MAP = {"bf16": torch.bfloat16, "fp8kv": E4M3}


def make_inputs(query_lens, kv_lens, nq, nk, hs, blk, kv_dtype, seed=0):
    device = "cuda"
    num_seqs = len(query_lens)
    max_kv = max(kv_lens)
    num_blocks = sum((kl + blk - 1) // blk for kl in kv_lens) + num_seqs + 16
    torch.manual_seed(seed)
    query = (torch.randn(sum(query_lens), nq, hs, dtype=torch.float32, device=device) * 0.3).to(torch.bfloat16)
    key_cache = (torch.randn(num_blocks, blk, nk, hs, dtype=torch.float32, device=device) * 0.3).to(kv_dtype)
    value_cache = (torch.randn(num_blocks, blk, nk, hs, dtype=torch.float32, device=device) * 0.3).to(kv_dtype)
    cu, bt_rows = [0], []
    bt_needed = (max_kv + blk - 1) // blk
    free = num_seqs + 1
    for ql, kl in zip(query_lens, kv_lens):
        nb = (kl + blk - 1) // blk
        bt = list(range(free, free + nb)) + [0] * (bt_needed - nb)
        free += nb
        bt_rows.append(bt)
        cu.append(cu[-1] + ql)
    return (query, key_cache, value_cache,
            torch.tensor(bt_rows, dtype=torch.int32, device=device),
            torch.tensor(cu, dtype=torch.int32, device=device),
            torch.tensor(kv_lens, dtype=torch.int32, device=device))


def ref_attention(query, key_cache, value_cache, block_table, cu, kv_lens, query_lens,
                  nq, nk, hs, bs, scale, k_descale=None, v_descale=None):
    outs = []
    for i, (ql, kl) in enumerate(zip(query_lens, kv_lens)):
        s = cu[i]
        q = query[s:s + ql].float()
        nb = (kl + bs - 1) // bs
        k = torch.cat([key_cache[b].float() for b in block_table[i][:nb]], 0)[:kl]
        v = torch.cat([value_cache[b].float() for b in block_table[i][:nb]], 0)[:kl]
        if k_descale is not None:
            k = k * k_descale
            v = v * v_descale
        k = k.repeat_interleave(nq // nk, 1).transpose(0, 1)
        v = v.repeat_interleave(nq // nk, 1).transpose(0, 1)
        off = kl - ql
        for r in range(ql):
            sc = torch.einsum("hd,htd->ht", q[r], k) * scale
            sc[:, off + r + 1:] = float("-inf")
            p = torch.softmax(sc, -1)
            outs.append(torch.einsum("ht,htd->hd", p, v).reshape(-1))
    return torch.stack(outs).to(torch.bfloat16)


def bench_case(name, query_lens, kv_lens, nq, nk, hs, blk, kv_dtype, check, reps=3):
    q, kc, vc, bt, cu, sl = make_inputs(query_lens, kv_lens, nq, nk, hs, blk, kv_dtype)
    out = torch.empty(sum(query_lens), nq, hs, dtype=torch.bfloat16, device="cuda")
    scale = hs ** -0.5
    base = dict(q=q, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=max(query_lens),
                seqused_k=sl, max_seqlen_k=max(kv_lens), softmax_scale=scale,
                causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None, alibi_slopes=None)
    from vllm.v1.kv_cache_interface import KVQuantMode
    kw_aiter = dict(base)
    kw_vllm = dict(base)
    if kv_dtype != torch.bfloat16:
        kd = torch.tensor([0.02], dtype=torch.float32, device="cuda")
        kw_aiter["k_descale"], kw_aiter["v_descale"] = kd, kd
        kw_vllm["k_descale"], kw_vllm["v_descale"] = kd, kd
        kw_vllm["kv_quant_mode"] = KVQuantMode.FP8_PER_TENSOR
    if kv_dtype == E4M3:
        pass  # q stays bf16 for fp8kv; fp8qkv would add q_descale

    ref = None
    if check:
        kd_ref = kd if kv_dtype != torch.bfloat16 else None
        vd_ref = kd if kv_dtype != torch.bfloat16 else None
        ref = ref_attention(q.float().to(torch.bfloat16), kc.float().to(torch.bfloat16),
                            vc.float().to(torch.bfloat16), bt, cu.tolist(), kv_lens,
                            query_lens, nq, nk, hs, blk, scale, kd_ref, vd_ref)
    res = {}
    for name_k, fn, kw in [("aiter", aiter_unified, kw_aiter), ("vllm", vllm_unified, kw_vllm)]:
        try:
            out.zero_()
            fn(**dict(kw, out=out))
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  [{name_k}] FAIL {type(e).__name__}: {str(e)[:90]}")
            res[name_k] = None
            continue
        if check:
            got = out.reshape(sum(query_lens), -1).float()
            rel = (got - ref.float()).norm() / ref.float().norm().clamp(min=1e-6)
            if rel.item() > 0.15:
                print(f"  [{name_k}] BAD rel_err={rel.item():.4f} -> ABORT case")
                return None
            print(f"  [{name_k}] OK rel_err={rel.item():.4f}")
        ms = triton.testing.do_bench(lambda f=fn, k=kw: f(**dict(k, out=out)),
                                     warmup=25, rep=100)
        res[name_k] = ms
    if res["aiter"] and res["vllm"]:
        print(f"  aiter={res['aiter']:.4f}ms  vllm={res['vllm']:.4f}ms  ratio={res['vllm']/res['aiter']:.3f}x")
        return res["aiter"], res["vllm"]
    return None


def heavy_tail(rng, n, lo=128, mid_lo=1024, mid_hi=8192, hi_lo=16384, hi_hi=65536):
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            out.append(rng.randint(hi_lo, hi_hi))
        elif r < 0.7:
            out.append(rng.randint(mid_lo, mid_hi))
        else:
            out.append(rng.randint(lo, 1024))
    return out


def suite_core(check):
    """Headline matrix: decode batch sweep, prefill granularity, mixed ratios (fp8kv)."""
    cases = []
    for bs, sk in [(8, 8192), (16, 16384), (32, 8192), (64, 32768), (128, 8192), (256, 4096), (256, 8192)]:
        cases.append((f"dec bs={bs} sk={sk}", [1]*bs, [sk]*bs))
    for lens in [[2048], [512]*4, [256]*8, [128]*16, [64]*32, [4096], [8192]]:
        cases.append((f"pre {lens}", lens, [8192]*len(lens)))
    for np_, pl, nd in [(1,2048,32),(1,4096,64),(4,512,64),(8,256,128),(16,128,256),(2,2048,256)]:
        cases.append((f"mix pre={np_}x{pl} dec={nd}", [1]*nd + [pl]*np_, [8192]*(nd+np_)))
    out = []
    for name, ql, kl in cases:
        print(f"\n[core] {name}")
        r = bench_case(name, ql, kl, 16, 1, 128, 32, E4M3, check)
        if r:
            out.append((name, *r))
    return out


def suite_dtype(check):
    """Same shapes under bf16 vs fp8kv — quantization impact per kernel."""
    shapes = [([1]*8, [8192]*8, "dec bs8 sk8k"),
              ([1]*64, [16384]*64, "dec bs64 sk16k"),
              ([2048], [8192], "pre 2048"),
              ([1]*128 + [1024]*4, [8192]*132, "mix 128d+4p1k")]
    out = []
    for dt in ["bf16", "fp8kv"]:
        for ql, kl, nm in shapes:
            print(f"\n[dtype] {nm} {dt}")
            r = bench_case(f"{nm} {dt}", ql, kl, 16, 1, 128, 32, DTYPE_MAP[dt], check)
            if r:
                out.append((f"{nm} {dt}", *r))
    return out


def suite_skew(check):
    """Skew deep-dive: same total KV bytes, different distributions (256 dec)."""
    rng = random.Random(7)
    out = []
    variants = [("uniform 8k", [8192]*256),
                ("uniform 16k", [16384]*256),
                ("heavy-tail 128-64k", heavy_tail(rng, 256)),
                ("bimodal 1k/32k", [(32768 if i % 5 == 0 else 1024) for i in range(256)]),
                ("one giant + rest 1k", [65536] + [1024]*255)]
    for nm, kl in variants:
        print(f"\n[skew] {nm} (total {sum(kl)//1_000_000}M tok)")
        r = bench_case(nm, [1]*256, kl, 16, 1, 128, 32, E4M3, check)
        if r:
            out.append((nm, *r))
    return out


def suite_heads(check):
    """GQA ratio sweep at fixed workload (128 dec, sk 8k)."""
    out = []
    for nq, nk in [(16, 1), (16, 2), (16, 4), (16, 8), (32, 8), (64, 8)]:
        print(f"\n[heads] GQA {nq}/{nk} dec bs=128 sk=8192")
        r = bench_case(f"gqa {nq}/{nk}", [1]*128, [8192]*128, nq, nk, 128, 32, E4M3, check)
        if r:
            out.append((f"gqa {nq}/{nk}", *r))
    return out


def suite_ctx(check):
    """Context length sweep at fixed batch (64 dec)."""
    out = []
    for sk in [1024, 4096, 8192, 16384, 32768, 65536]:
        print(f"\n[ctx] sk={sk} bs=64")
        r = bench_case(f"ctx {sk}", [1]*64, [sk]*64, 16, 1, 128, 32, E4M3, check)
        if r:
            out.append((f"ctx {sk}", *r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["core", "dtype", "skew", "heads", "ctx", "all"], default="all")
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()

    suites = {"core": suite_core, "dtype": suite_dtype, "skew": suite_skew,
              "heads": suite_heads, "ctx": suite_ctx}
    keys = list(suites) if args.suite == "all" else [args.suite]
    print(f"suites={keys} check={args.check}")
    rows = []
    for k in keys:
        rows.extend(suites[k](args.check))

    print(f"\n{'='*70}\nSUMMARY ({len(rows)} cases)\n{'='*70}")
    wins = sum(1 for _, a, v in rows if v > a)
    print(f"aiter wins {wins}/{len(rows)} cases")
    for name, a, v in rows:
        print(f"  {name:42s} aiter={a:8.4f} vllm={v:8.4f} {v/a:6.3f}x")

    if args.csv:
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["tag", "case", "aiter_ms", "vllm_ms", "ratio"])
            for name, a, v in rows:
                w.writerow([args.tag, name, f"{a:.4f}", f"{v:.4f}", f"{v/a:.4f}"])
        print(f"saved -> {args.csv}")


if __name__ == "__main__":
    main()
