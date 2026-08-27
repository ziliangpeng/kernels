#!/usr/bin/env python3
"""
FP16 vs BF16 non-batch GEMM determinism — expanded version.

More M values, more K/N combinations, multiple random seeds.
Runs each configuration and compares row 0 of M=K vs M=1 reference.

Usage:
    python fp16_vs_bf16_mm_expanded.py
    python fp16_vs_bf16_mm_expanded.py --seeds 5    # number of random seeds (default 3)
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
    parser.add_argument("--seeds", type=int, default=3, help="number of random input seeds")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Seeds: {args.seeds}")
    print()

    M_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    K_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
    N_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
    SEEDS = [42 + i * 1000 for i in range(args.seeds)]
    WEIGHT_SEED = 99

    results = []
    total_configs = len(M_VALUES) * len(K_VALUES) * len(N_VALUES) * len(SEEDS) * 2
    config_idx = 0

    for K in K_VALUES:
        for N in N_VALUES:
            for dtype_name in ["fp16", "bf16"]:
                dtype = DTYPE_MAP[dtype_name]
                W = seeded_tensor(K, N, dtype=dtype, seed=WEIGHT_SEED)

                for seed in SEEDS:
                    x_ref = seeded_tensor(1, K, dtype=dtype, seed=seed)
                    ref_out = torch.mm(x_ref, W)
                    ref_cs = checksum(ref_out)

                    for M in M_VALUES:
                        config_idx += 1
                        if M == 1:
                            x_batch = x_ref.clone()
                        else:
                            x_rest = seeded_tensor(M - 1, K, dtype=dtype, seed=100 + M + seed)
                            x_batch = torch.cat([x_ref, x_rest], dim=0)

                        out = torch.mm(x_batch, W)
                        row0 = out[0:1]

                        exact = checksum(row0) == ref_cs
                        diff = rel_diff_pct(ref_out, row0)

                        results.append({
                            "dtype": dtype_name,
                            "M": M, "K": K, "N": N, "seed": seed,
                            "bit_exact": exact,
                            "rel_diff_pct": diff,
                        })

            # Progress
            total_done = config_idx
            if total_done % 100 == 0:
                print(f"  ... {total_done}/{total_configs} configs done", flush=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ({len(results)} configs per dtype = {len(M_VALUES)} M × {len(K_VALUES)} K × {len(N_VALUES)} N × {len(SEEDS)} seeds)")
    print(f"{'='*80}")

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
            sorted_diffs = sorted(diffs)
            print(f"    Diff p25:   {sorted_diffs[len(sorted_diffs)//4]:.4f}%")
            print(f"    Diff median: {sorted_diffs[len(sorted_diffs)//2]:.4f}%")
            print(f"    Diff p75:   {sorted_diffs[3*len(sorted_diffs)//4]:.4f}%")
            print(f"    Diff mean:  {sum(diffs)/len(diffs):.4f}%")

    # Head-to-head: group by (M, K, N, seed)
    print(f"\n  Head-to-head (BF16 worse than FP16?):")
    fp16_map = {(r["M"], r["K"], r["N"], r["seed"]): r["rel_diff_pct"] for r in results if r["dtype"] == "fp16"}
    bf16_map = {(r["M"], r["K"], r["N"], r["seed"]): r["rel_diff_pct"] for r in results if r["dtype"] == "bf16"}

    bf16_worse = sum(1 for k in fp16_map if bf16_map[k] > fp16_map[k])
    fp16_worse = sum(1 for k in fp16_map if fp16_map[k] > bf16_map[k])
    equal = sum(1 for k in fp16_map if fp16_map[k] == bf16_map[k])
    total = len(fp16_map)

    print(f"    BF16 worse: {bf16_worse}/{total} ({bf16_worse/total*100:.1f}%)")
    print(f"    FP16 worse: {fp16_worse}/{total} ({fp16_worse/total*100:.1f}%)")
    print(f"    Equal:      {equal}/{total} ({equal/total*100:.1f}%)")

    # Break down: when both non-exact, which is worse?
    both_nonexact = {k for k in fp16_map if not (fp16_map[k] == 0 and bf16_map[k] == 0)}
    if both_nonexact:
        b_worse = sum(1 for k in both_nonexact if bf16_map[k] > fp16_map[k])
        f_worse = sum(1 for k in both_nonexact if fp16_map[k] > bf16_map[k])
        print(f"\n  When at least one is non-exact ({len(both_nonexact)} cases):")
        print(f"    BF16 worse: {b_worse}/{len(both_nonexact)} ({b_worse/len(both_nonexact)*100:.1f}%)")
        print(f"    FP16 worse: {f_worse}/{len(both_nonexact)} ({f_worse/len(both_nonexact)*100:.1f}%)")

if __name__ == "__main__":
    main()
