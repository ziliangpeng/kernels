#!/usr/bin/env python3
"""
FP16 vs BF16 batch GEMM determinism comparison.

Systematic test to determine if BF16 is consistently worse than FP16
in cross-batch non-determinism across many shapes and batch sizes.

Tests:
1. Same shape, both dtypes side by side
2. Multiple M values (1, 4, 16, 64, 128, 256, 512)
3. Multiple K values (1024, 4096, 16384)
4. Multiple N values (1024, 4096, 16384)
5. Multiple B values (2, 8, 32, 128)

For each (M, K, N, B, dtype): compute rel diff of slot 0 vs B=1 reference.

Usage:
    python fp16_vs_bf16.py                    # full sweep
    python fp16_vs_bf16.py --gpu amd           # specify GPU (just for labeling)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="unknown", help="GPU label for output")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    M_VALUES = [1, 4, 16, 64, 128, 256, 512]
    K_VALUES = [1024, 4096, 16384]
    N_VALUES = [1024, 4096, 16384]
    B_VALUES = [2, 8, 32, 128]

    results = []

    for K in K_VALUES:
        for N in N_VALUES:
            print(f"{'='*90}")
            print(f"  K={K}  N={N}")
            print(f"{'='*90}")

            for dtype_name in ["fp16", "bf16"]:
                dtype = DTYPE_MAP[dtype_name]
                W = seeded_tensor(K, N, dtype=dtype, seed=99)

                print(f"\n  [{dtype_name}]")
                print(f"  {'M':>6}  {'B=2':>10}  {'B=8':>10}  {'B=32':>10}  {'B=128':>10}")
                print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

                for M in M_VALUES:
                    x1 = seeded_tensor(M, K, dtype=dtype, seed=42)
                    ref_out = torch.mm(x1, W)
                    ref_cs = checksum(ref_out)

                    row_vals = []
                    for B in B_VALUES:
                        x_batch = x1.repeat(B, 1)
                        out = torch.mm(x_batch, W)
                        slot0 = out[0:M]

                        exact = checksum(slot0) == ref_cs
                        diff = rel_diff_pct(ref_out, slot0)

                        if exact:
                            val = "✅"
                        else:
                            val = f"{diff:.4f}%"
                        row_vals.append(val)

                        results.append({
                            "dtype": dtype_name,
                            "M": M, "K": K, "N": N, "B": B,
                            "bit_exact": exact,
                            "rel_diff_pct": diff,
                        })

                    print(f"  {M:>6}  {row_vals[0]:>10}  {row_vals[1]:>10}  {row_vals[2]:>10}  {row_vals[3]:>10}")

            print()

    # Summary statistics
    print(f"{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    for dtype_name in ["fp16", "bf16"]:
        dtype_results = [r for r in results if r["dtype"] == dtype_name]
        exact_count = sum(1 for r in dtype_results if r["bit_exact"])
        total = len(dtype_results)
        diffs = [r["rel_diff_pct"] for r in dtype_results if not r["bit_exact"]]

        print(f"\n  {dtype_name}:")
        print(f"    Bit-exact: {exact_count}/{total} ({exact_count/total*100:.1f}%)")
        if diffs:
            print(f"    Non-exact cases: {len(diffs)}")
            print(f"    Diff range: {min(diffs):.4f}% — {max(diffs):.4f}%")
            print(f"    Diff median: {sorted(diffs)[len(diffs)//2]:.4f}%")
            print(f"    Diff mean: {sum(diffs)/len(diffs):.4f}%")

    # Head-to-head: for each (M,K,N,B), is bf16 worse than fp16?
    print(f"\n  Head-to-head (BF16 worse than FP16?):")
    fp16_map = {(r["M"], r["K"], r["N"], r["B"]): r["rel_diff_pct"] for r in results if r["dtype"] == "fp16"}
    bf16_map = {(r["M"], r["K"], r["N"], r["B"]): r["rel_diff_pct"] for r in results if r["dtype"] == "bf16"}

    bf16_worse = 0
    fp16_worse = 0
    equal = 0
    for key in fp16_map:
        f = fp16_map[key]
        b = bf16_map[key]
        if b > f:
            bf16_worse += 1
        elif f > b:
            fp16_worse += 1
        else:
            equal += 1

    total = len(fp16_map)
    print(f"    BF16 worse: {bf16_worse}/{total} ({bf16_worse/total*100:.1f}%)")
    print(f"    FP16 worse: {fp16_worse}/{total} ({fp16_worse/total*100:.1f}%)")
    print(f"    Equal:      {equal}/{total} ({equal/total*100:.1f}%)")

if __name__ == "__main__":
    main()
