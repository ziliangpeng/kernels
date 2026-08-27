#!/usr/bin/env python3
"""
Cross-batch GEMM determinism test.

Tests whether cuBLAS produces bit-exact results when the same input
is computed under different batch sizes (M dimension).

This is the L2 batch non-invariance phenomenon described by Thinking Machines.

Usage:
    python cross_batch_gemm.py                    # all dtypes
    python cross_batch_gemm.py --dtype fp32        # single dtype
    python cross_batch_gemm.py --n 200             # iterations (default 200)
"""

import argparse
import hashlib
import os
import time

import torch

DTYPE_MAP = {
    "fp32": (torch.float32, "sgemm"),
    "fp16": (torch.float16, "hgemm"),
    "bf16": (torch.bfloat16, "bgemm"),
}

def checksum(tensor):
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
    parser.add_argument("--n", type=int, default=200, help="iterations for run-to-run check")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dtypes = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())

    # Test shapes: (M, K, N) — we test M=1 as baseline, then various batch sizes
    # The input vector (row 0) is the SAME across all batch sizes
    K = 4096
    N = 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for dtype_name in dtypes:
        dtype, blas_name = DTYPE_MAP[dtype_name]
        print(f"{'='*70}")
        print(f"  dtype: {dtype_name} ({blas_name})   K={K}  N={N}")
        print(f"{'='*70}")

        # Fixed weight matrix (same across all tests)
        W = seeded_tensor(K, N, dtype=dtype, seed=99)

        # Reference: M=1, single vector
        x1 = seeded_tensor(1, K, dtype=dtype, seed=42)
        ref_output = torch.mm(x1, W)  # [1, N]
        ref_cs = checksum(ref_output)
        print(f"\n  Reference (M=1): checksum={ref_cs[:16]}...")
        print(f"  {'Batch M':>8}  {'Bit-exact?':>12}  {'Max abs diff':>14}  {'Checksum (first 16)':>20}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*20}")

        results = []

        for M in batch_sizes:
            # Create batch: row 0 = same x1, rows 1..M-1 = random (won't compare)
            if M == 1:
                x_batch = x1.clone()
            else:
                x_rest = seeded_tensor(M - 1, K, dtype=dtype, seed=100 + M)
                x_batch = torch.cat([x1, x_rest], dim=0)  # [M, K]

            # Run GEMM
            out_batch = torch.mm(x_batch, W)  # [M, N]
            out_row0 = out_batch[0:1]  # extract row 0 — same input as M=1

            # Compare with reference
            cs = checksum(out_row0)
            bit_exact = (cs == ref_cs)
            diff = max_abs_diff(ref_output, out_row0)

            status = "✅ exact" if bit_exact else "❌ diff"
            print(f"  {M:>8}  {status:>12}  {diff:>14.2e}  {cs[:16]:>20}...")
            results.append({
                "dtype": dtype_name,
                "M": M,
                "K": K,
                "N": N,
                "bit_exact": bit_exact,
                "max_abs_diff": diff,
            })

        # Also test: does the SAME batch size produce run-to-run stable results?
        # Pick M=128 as representative
        print(f"\n  Run-to-run stability (M=128, {args.n} iterations):")
        x128 = seeded_tensor(128, K, dtype=dtype, seed=42)
        unique_checksums = set()
        for i in range(args.n):
            out = torch.mm(x128, W)
            unique_checksums.add(checksum(out[0:1]))

        if len(unique_checksums) == 1:
            print(f"    ✅ stable — all {args.n} runs identical")
        else:
            print(f"    ❌ {len(unique_checksums)} unique outputs out of {args.n} runs")

        # Test TF32 effect (FP32 only)
        if dtype_name == "fp32":
            print(f"\n  TF32 effect (M=1 vs M=128):")
            for tf32 in [False, True]:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                out1 = torch.mm(x1, W)
                x128a = seeded_tensor(128, K, dtype=dtype, seed=42)
                out128 = torch.mm(x128a, W)[0:1]
                diff = max_abs_diff(out1, out128)
                label = "TF32 ON" if tf32 else "TF32 OFF"
                exact = "✅" if diff == 0 else "❌"
                print(f"    {label}: {exact} diff={diff:.2e}")

            # Reset
            torch.backends.cuda.matmul.allow_tf32 = False

        print()

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_exact = all(r["bit_exact"] for r in results)
    if all_exact:
        print("All batch sizes produced bit-exact results vs M=1 reference.")
        print("→ No cross-batch non-determinism detected on this GPU/CUDA/PyTorch.")
    else:
        print("Cross-batch non-determinism DETECTED:")
        for r in results:
            if not r["bit_exact"]:
                print(f"  {r['dtype']} M={r['M']}: diff={r['max_abs_diff']:.2e}")
        print("→ Same input produces different output under different batch sizes.")
        print("→ This is cuBLAS selecting different algorithms for different M values.")

if __name__ == "__main__":
    main()
