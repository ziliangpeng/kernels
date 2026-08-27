#!/usr/bin/env python3
"""
Cross-batch GEMM determinism test v2 — multi-run.

For each batch size M, run GEMM N times with the SAME input.
Record how many unique outputs appear across N runs.

If non-determinism is probabilistic (e.g. L2 cache timing, 
cuBLASLt heuristic flapping), we need many runs to catch it.

Usage:
    python cross_batch_gemm_v2.py                    # all dtypes, all M
    python cross_batch_gemm_v2.py --dtype fp32        # single dtype
    python cross_batch_gemm_v2.py --n 100             # runs per (M, dtype) (default 100)
"""

import argparse
import hashlib
import os
import time

import torch

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def checksum(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return hashlib.md5(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()

def seeded_tensor(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--n", type=int, default=100, help="runs per (M, dtype)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda or 'N/A (ROCm)'}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Runs per (M, dtype): {args.n}")
    print()

    dtypes = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())
    K, N = 4096, 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for dtype_name in dtypes:
        dtype = DTYPE_MAP[dtype_name]
        print(f"{'='*80}")
        print(f"  dtype: {dtype_name}   K={K}  N={N}   runs={args.n}")
        print(f"{'='*80}")

        W = seeded_tensor(K, N, dtype=dtype, seed=99)

        print(f"\n  {'Batch M':>8}  {'Unique':>7}  {'Stable?':>8}  {'Max pairwise diff':>18}  {'Avg ms':>8}")
        print(f"  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*18}  {'-'*8}")

        for M in batch_sizes:
            x = seeded_tensor(M, K, dtype=dtype, seed=42)

            # Run N times, record every output checksum + keep outputs for diff
            checksums = []
            outputs = []
            times = []

            for i in range(args.n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = torch.mm(x, W)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

                # Only store row 0 for comparison (same input across all M)
                row0 = out[0:1]
                cs = checksum(row0)
                checksums.append(cs)
                if len(outputs) < 10:  # keep first 10 outputs for pairwise diff
                    outputs.append(row0.clone())

            unique_cs = set(checksums)
            n_unique = len(unique_cs)
            stable = n_unique == 1

            # Max pairwise diff across stored outputs
            max_pairwise = 0.0
            if not stable:
                for i in range(len(outputs)):
                    for j in range(i + 1, len(outputs)):
                        if checksums[i] != checksums[j]:
                            d = max_abs_diff(outputs[i], outputs[j])
                            if d > max_pairwise:
                                max_pairwise = d

            avg_ms = sum(times) / len(times)
            status = "✅ stable" if stable else f"❌ {n_unique} uniq"
            diff_str = f"{max_pairwise:.2e}" if not stable else "—"
            print(f"  {M:>8}  {n_unique:>7}  {status:>8}  {diff_str:>18}  {avg_ms:>8.3f}")

        print()

    print(f"{'='*80}")
    print("Done. Look for ❌ rows — those M values have run-to-run non-determinism.")
    print("If all rows are ✅, then non-determinism is only cross-batch (different M),")
    print("not run-to-run (same M, different run).")

if __name__ == "__main__":
    main()
