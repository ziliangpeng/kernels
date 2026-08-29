"""Kernel-level A/B #2: SGLang aiter unified_attention vs alternatives.

Candidates:
  - sglang_unified : aiter.ops.triton.attention.unified_attention (same kernel SGLang uses;
                     sglang wraps it — we call it the same way sglang's aiter_backend does)
  - aiter_pa_ragged: aiter paged_attention_ragged (SGLang's OTHER decode path — the
                     "alternative" branch in aiter_backend when not using unified)
  - vllm_unified   : vLLM's own triton unified kernel (TRITON_ATTN) for cross-reference

All correctness-gated vs dense torch reference. Same inputs to every kernel.

Usage:
  python3 bench_sglang_ab.py --suite core
  python3 bench_sglang_ab.py --suite all --csv sglang_matrix.csv
"""
import argparse
import random

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention as sglang_unified
from aiter import paged_attention_ragged
# vllm reference removed for sglang-image run

E4M3 = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
E4M3_STR = "fp8_e4m3"


def make_inputs(query_lens, kv_lens, nq, nk, hs, blk, kv_dtype, seed=0):
    device = "cuda"
    num_seqs = len(query_lens)
    max_kv = max(kv_lens)
    num_blocks = sum((kl + blk - 1) // blk for kl in kv_lens) + num_seqs + 16
    torch.manual_seed(seed)
    query = (torch.randn(sum(query_lens), nq, hs, dtype=torch.float32, device=device) * 0.3).to(torch.bfloat16)
    key_cache = (torch.randn(num_blocks, blk, nk, hs, dtype=torch.float32, device=device) * 0.3).to(kv_dtype)
    value_cache = (torch.randn(num_blocks, blk, nk, hs, dtype=torch.float32, device=device) * 0.3).to(kv_dtype)

    # block-level page table (for unified kernels)
    cu, bt_rows = [0], []
    bt_needed = (max_kv + blk - 1) // blk
    free = num_seqs + 1
    for ql, kl in zip(query_lens, kv_lens):
        nb = (kl + blk - 1) // blk
        bt = list(range(free, free + nb)) + [0] * (bt_needed - nb)
        free += nb
        bt_rows.append(bt)
        cu.append(cu[-1] + ql)
    block_table = torch.tensor(bt_rows, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # token-level kv_indices (for paged_attention_ragged):
    # per-token physical slot = physical_block * blk + offset_in_block, gathered per-seq
    kv_flat = []
    kv_indptr = [0]
    for row, (ql_, kl) in zip(bt_rows, zip(query_lens, kv_lens)):
        nb = (kl + blk - 1) // blk
        toks = []
        for bi in range(nb):
            b = row[bi]
            for o in range(blk):
                toks.append(b * blk + o)
        kv_flat.extend(toks[:kl])
        kv_indptr.append(len(kv_flat))
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
    kv_indices = torch.tensor(kv_flat, dtype=torch.int32, device=device)
    last_len = [((kl - 1) % blk) + 1 for kl in kv_lens]

    k_descale = torch.tensor([0.02], dtype=torch.float32, device=device)
    v_descale = torch.tensor([0.02], dtype=torch.float32, device=device)
    return (query, key_cache, value_cache, block_table, cu_seqlens, seq_lens,
            kv_indptr, kv_indices, last_len, k_descale, v_descale)


def ref_attention(query, key_cache, value_cache, block_table, cu, kv_lens, query_lens,
                  nq, nk, hs, bs, scale, k_descale, v_descale):
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


def bench_one(name, ql, kl, nq, nk, hs, blk, kv_dtype, check, reps=3):
    query, kc, vc, bt, cu, sl, kv_indptr, kv_indices, last_len, k_des, v_des = make_inputs(
        ql, kl, nq, nk, hs, blk, kv_dtype)
    num_tokens = sum(ql)
    out = torch.empty(num_tokens, nq, hs, dtype=torch.bfloat16, device="cuda")
    scale = hs ** -0.5

    is_fp8 = kv_dtype != torch.bfloat16
    aiter_kw = dict(q=query, k=kc, v=vc, out=out, cu_seqlens_q=cu,
                    max_seqlen_q=max(ql), seqused_k=sl, max_seqlen_k=max(kl),
                    softmax_scale=scale, causal=True, window_size=(-1, -1),
                    block_table=bt, softcap=0.0,
                    q_descale=None, k_descale=k_des if is_fp8 else None,
                    v_descale=v_des if is_fp8 else None)
    # paged_attention_ragged: SGLang's alternative decode path
    ws_buffer = torch.empty(
        (max(len(ql), 1) * nq * ((max(kl) + 255) // 256) * hs) * 4
        + 2 * (len(ql) * nq * ((max(kl) + 255) // 256)) * 4,
        dtype=torch.uint8, device="cuda")
    ragged_kw = dict(out=out.view(-1, nq, hs),
                     workspace_buffer=ws_buffer,
                     q=query.view(-1, nq, hs),
                     key_cache=kc.view(-1, 1, nk, hs),
                     value_cache=vc.view(-1, 1, nk, hs),
                     scale=scale,
                     kv_indptr=kv_indptr,
                     kv_indices=kv_indices,
                     kv_last_page_len=torch.tensor(last_len, dtype=torch.int32, device="cuda"),
                     none_num=None,
                     max_num_partitions=(max(kl) + 255) // 256,
                     kv_cache_dtype=E4M3_STR if is_fp8 else "auto",
                     kv_cache_layout="NHD",
                     logits_soft_cap=0.0,
                     k_scale=torch.tensor([0.02], dtype=torch.float32, device="cuda"),
                     v_scale=torch.tensor([0.02], dtype=torch.float32, device="cuda"),
                     block_table=bt)

    ref = None
    if check:
        ref = ref_attention(query.float().to(torch.bfloat16), kc.float().to(torch.bfloat16),
                            vc.float().to(torch.bfloat16), bt, cu.tolist(), kl, ql,
                            nq, nk, hs, blk, scale, k_des if is_fp8 else None, v_des if is_fp8 else None)

    res = {}
    for nm, fn, kw in [("sglang_unified", sglang_unified, aiter_kw)]:
        if nm == "sgl_pa_ragged":
            # call via aiter.paged_attention_ragged signature
            def call():
                paged_attention_ragged(
                    ragged_kw["out"], ragged_kw["workspace_buffer"], ragged_kw["q"],
                    ragged_kw["key_cache"], ragged_kw["value_cache"], ragged_kw["scale"],
                    ragged_kw["kv_indptr"], ragged_kw["kv_indices"], ragged_kw["kv_last_page_len"],
                    32, ragged_kw["max_num_partitions"], None,
                    ragged_kw["kv_cache_dtype"], ragged_kw["kv_cache_layout"],
                    ragged_kw["logits_soft_cap"], ragged_kw["k_scale"], ragged_kw["v_scale"],
                )
            fn_call = call
        else:
            fn_call = (lambda f=fn, k=kw: f(**dict(k, out=out)))
        try:
            out.zero_()
            fn_call()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  [{nm}] FAIL {type(e).__name__}: {str(e)[:90]}")
            res[nm] = None
            continue
        if check:
            got = out.reshape(num_tokens, -1).float()
            ref32 = ref.float()
            rel = (got - ref32).norm() / ref32.norm().clamp(min=1e-6)
            if rel.item() > 0.15:
                print(f"  [{nm}] BAD rel_err={rel.item():.4f} -> ABORT")
                return None
            print(f"  [{nm}] OK rel_err={rel.item():.4f}")
        ms = triton.testing.do_bench(fn_call, warmup=25, rep=100)
        res[nm] = ms

    if res.get("sglang_unified") and res.get("vllm_unified"):
        line = f"  sglang_unified={res['sglang_unified']:.4f}"
        if res.get("sgl_pa_ragged"):
            line += f"  pa_ragged={res['sgl_pa_ragged']:.4f}"
        line += f"  vllm_unified={res['vllm_unified']:.4f}"
        print(line)
        return res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["core", "skew", "ctx", "all"], default="core")
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import itertools
    cases = []
    # decode sweep (the main regime)
    for bs, sk in [(8, 8192), (32, 8192), (128, 8192), (64, 32768), (256, 4096)]:
        cases.append((f"dec bs={bs} sk={sk}", [1]*bs, [sk]*bs))
    # prefill
    for lens in [[2048], [512]*4, [256]*8]:
        cases.append((f"pre {lens}", lens, [8192]*len(lens)))
    # mixed
    for np_, pl, nd in [(1, 2048, 32), (4, 512, 64), (8, 128, 128)]:
        cases.append((f"mix {np_}x{pl}+{nd}d", [1]*nd + [pl]*np_, [8192]*(nd+np_)))
    # skew
    rng = random.Random(7)
    tail = []
    for _ in range(256):
        r = rng.random()
        if r < 0.2: tail.append(rng.randint(16384, 65536))
        elif r < 0.7: tail.append(rng.randint(1024, 8192))
        else: tail.append(rng.randint(128, 1024))
    cases.append(("skew heavy-tail", [1]*256, tail))
    cases.append(("skew one-giant", [1]*256, [65536] + [1024]*255))
    # ctx sweep
    for sk in [1024, 16384, 65536]:
        cases.append((f"ctx sk={sk}", [1]*64, [sk]*64))

    print(f"check={args.check}  cases={len(cases)}")
    rows = []
    for nm, ql, kl in cases:
        print(f"\n[{nm}]")
        r = bench_one(nm, ql, kl, 16, 1, 128, 32, E4M3, args.check)
        if r:
            rows.append((nm, r))

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for nm, r in rows:
        parts = [f"{k}={v:.4f}" for k, v in r.items() if v]
        print(f"  {nm:28s} {' '.join(parts)}")
    if args.csv:
        import csv as csvmod
        keys = ["sglang_unified", "sgl_pa_ragged", "vllm_unified"]
        with open(args.csv, "a", newline="") as f:
            w = csvmod.writer(f)
            if f.tell() == 0:
                w.writerow(["case"] + keys + ["unified_vs_vllm"])
            for nm, r in rows:
                row = [nm] + [f"{r.get(k):.4f}" if r.get(k) else "" for k in keys]
                if r.get("sglang_unified") and r.get("vllm_unified"):
                    row.append(f"{r['vllm_unified']/r['sglang_unified']:.4f}")
                w.writerow(row)
        print(f"saved -> {args.csv}")


if __name__ == "__main__":
    main()
