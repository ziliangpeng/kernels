#!/usr/bin/env python3
"""
FP16 vs BF16 non-batch GEMM determinism comparison.

Same as fp16_vs_bf16.py but uses torch.mm (vary M, not B).
This tests cross-M non-determinism: same input row 0, different M values.

Usage:
    python fp16_vs_bf16_mm.py
"""

import argparse
import hashlib
import time

import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def checksum(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return hashlib.md5(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def rel_diff_pct(a, b):
    abs_diff = (a.float() - b.float()).abs().max().item()
    abs_val = a.float().abs().max().item()
    if abs_val == 0:
        return 0.0
    return abs_diff / abs_val * 100.0

def seeded_tensor(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # M_VALUES act as "batch" — we compare row 0 of M=1 vs row 0 of M=2,4,...
    M_VALUES_REF = [1]  # reference
    M_VALUES_TEST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    K_VALUES = [1024, 4096, 16384]
    N_VALUES = [1024, 4096, 16384]

    results = []

    for K in K_VALUES:
        for N in N_VALUES:
            print(f"{'='*90}")
            print(f"  K={K}  N={N}")
            print(f"{'='*90}")

            for dtype_name in ["fp16", "bf16"]:
                dtype = DTYPE_MAP[dtype_name]
                W = seeded_tensor(K, N, dtype=dtype, seed=99)

                # Reference: M=1
                x_ref = seeded_tensor(1, K, dtype=dtype, seed=42)
                ref_out = torch.mm(x_ref, W)  # [1, N]
                ref_cs = checksum(ref_out)

                print(f"\n  [{dtype_name}]  (ref: M=1)")
                print(f"  {'M':>8}  {'Bit-exact?':>12}  {'Rel diff':>10}")
                print(f"  {'-'*8}  {'-'*12}  {'-'*10}")

                for M in M_VALUES_TEST:
                    # Build [M, K] where row 0 = x_ref, rest = random
                    if M == 1:
                        x_batch = x_ref.clone()
                    else:
                        x_rest = seeded_tensor(M - 1, K, dtype=dtype, seed=100 + M)
                        x_batch = torch.cat([x_ref, x_rest], dim=0)

                    out = torch.mm(x_batch, W)  # [M, N]
                    row0 = out[0:1]  # [1, N]

                    exact = checksum(row0) == ref_cs
                    diff = rel_diff_pct(ref_out, row0)
                    status = "✅ exact" if exact else "❌ diff"
                    print(f"  {M:>8}  {status:>12}  {diff:>9.4f}%")

                    results.append({
                        "dtype": dtype_name,
                        "M": M, "K": K, "N": N,
                        "bit_exact": exact,
                        "rel_diff_pct": diff,
                    })

            print()

    # Summary
    print(f"{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    for dtype_name in ["fp16", "bf16"]:
        dr = [r for r in results if r["dtype"] == dtype_name]
        exact = sum(1 for r in dr if r["bit_exact"])
        total = len(dr)
        diffs = [r["rel_diff_pct"] for r in dr if not r["bit_exact"]]

        print(f"\n  {dtype_name}:")
        print(f"    Bit-exact: {exact}/{total} ({exact/total*100:.1f}%)")
        if diffs:
            print(f"    Non-exact: {len(diffs)}")
            print(f"    Diff range: {min(diffs):.4f}% — {max(diffs):.4f}%")
            print(f"    Diff median: {sorted(diffs)[len(diffs)//2]:.4f}%")
            print(f"    Diff mean: {sum(diffs)/len(diffs):.4f}%")

    # Head-to-head
    print(f"\n  Head-to-head (BF16 worse than FP16?):")
    fp16_map = {(r["M"], r["K"], r["N"]): r["rel_diff_pct"] for r in results if r["dtype"] == "fp16"}
    bf16_map = {(r["M"], r["K"], r["N"]): r["rel_diff_pct"] for r in results if r["dtype"] == "bf16"}

    bf16_worse = sum(1 for k in fp16_map if bf16_map[k] > fp16_map[k])
    fp16_worse = sum(1 for k in fp16_map if fp16_map[k] > bf16_map[k])
    equal = sum(1 for k in fp16_map if fp16_map[k] == bf16_map[k])
    total = len(fp16_map)

    print(f"    BF16 worse: {bf16_worse}/{total} ({bf16_worse/total*100:.1f}%)")
    print(f"    FP16 worse: {fp16_worse}/{total} ({fp16_worse/total*100:.1f}%)")
    print(f"    Equal:      {equal}/{total} ({equal/total*100:.1f}%)")

if __name__ == "__main__":
    main()
