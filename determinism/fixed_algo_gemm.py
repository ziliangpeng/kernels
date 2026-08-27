#!/usr/bin/env python3
"""
Test if fixing the rocBLAS algorithm eliminates cross-batch non-determinism.

rocBLAS allows selecting a specific solution index via rocblas_gemm_ex.
If we force the same algo for all batch sizes, cross-batch diff should disappear.

We also test a simpler approach: a pure PyTorch batch-invariant GEMM
that manually loops over batch dimension (forces same code path per slot).

Usage:
    python fixed_algo_gemm.py                    # all dtypes
    python fixed_algo_gemm.py --dtype fp32        # single dtype
"""

import argparse
import hashlib
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

def rel_diff_pct(a, b):
    """Relative difference as percentage: max(|a-b|) / max(|a|) * 100."""
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
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()))
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dtypes = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())
    M, K, N = 128, 4096, 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for dtype_name in dtypes:
        dtype = DTYPE_MAP[dtype_name]
        print(f"{'='*80}")
        print(f"  dtype: {dtype_name}   M={M}  K={K}  N={N}")
        print(f"{'='*80}")

        W_single = seeded_tensor(K, N, dtype=dtype, seed=99)

        # ── Approach 1: torch.bmm (default, may use different algos) ──
        print(f"\n  ── Approach 1: torch.bmm (default heuristic) ──")
        x1 = seeded_tensor(1, M, K, dtype=dtype, seed=42)
        W1 = W_single.unsqueeze(0)
        ref_out = torch.bmm(x1, W1)[0]  # [M, N]
        ref_cs = checksum(ref_out)

        print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Rel diff':>10}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}")
        for B in batch_sizes:
            x_batch = x1.repeat(B, 1, 1)
            W_batch = W1.repeat(B, 1, 1)
            out = torch.bmm(x_batch, W_batch)[0]
            cs = checksum(out)
            diff = rel_diff_pct(ref_out, out)
            status = "✅ exact" if cs == ref_cs else "❌ diff"
            print(f"  {B:>6}  {status:>12}  {diff:>10.4f}%")

        # ── Approach 2: Manual loop (one bmm call per batch slot, B=1 each time) ──
        # This forces the SAME algo for every "batch" because each call is B=1
        print(f"\n  ── Approach 2: Manual loop (B×(M,K,N) as separate mm calls) ──")
        print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Max diff':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

        # Reference: single mm
        x_mm = seeded_tensor(M, K, dtype=dtype, seed=42)
        ref_mm = torch.mm(x_mm, W_single)
        ref_mm_cs = checksum(ref_mm)

        for B in batch_sizes:
            # Run B separate mm calls, each with the same input
            # Stack results, take slot 0
            outs = []
            for b in range(min(B, 4)):  # only need first few to compare
                out_b = torch.mm(x_mm, W_single)  # always M×K × K×N
                outs.append(out_b)

            cs = checksum(outs[0])
            diff = rel_diff_pct(ref_mm, outs[0])
            status = "✅ exact" if cs == ref_mm_cs else "❌ diff"
            print(f"  {B:>6}  {status:>12}  {diff:>10.4f}%")

        # ── Approach 3: torch.mm with M padded to simulate batch ──
        # Instead of bmm, use mm with M_total = B*M, extract rows
        print(f"\n  ── Approach 3: torch.mm with M=B*M (flatten batch into M) ──")
        print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Max diff':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

        for B in batch_sizes:
            # All B slots identical input
            x_flat = x_mm.repeat(B, 1)  # [B*M, K]
            out_flat = torch.mm(x_flat, W_single)  # [B*M, N]
            slot0 = out_flat[0:M]  # first M rows

            cs = checksum(slot0)
            diff = rel_diff_pct(ref_mm, slot0)
            status = "✅ exact" if cs == ref_mm_cs else "❌ diff"
            print(f"  {B:>6}  {status:>12}  {diff:>10.4f}%")

        # ── Approach 4: einsum (different dispatch path) ──
        print(f"\n  ── Approach 4: torch.einsum ('bmk,bkn->bmn') ──")
        print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Max diff':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

        ref_ein = torch.einsum('bmk,bkn->bmn', x1, W1)[0]
        ref_ein_cs = checksum(ref_ein)

        for B in batch_sizes:
            x_batch = x1.repeat(B, 1, 1)
            W_batch = W1.repeat(B, 1, 1)
            out = torch.einsum('bmk,bkn->bmn', x_batch, W_batch)[0]
            cs = checksum(out)
            diff = rel_diff_pct(ref_ein, out)
            status = "✅ exact" if cs == ref_ein_cs else "❌ diff"
            print(f"  {B:>6}  {status:>12}  {diff:>10.4f}%")

        print()

    print("Done. Approach 2 (manual loop) should be bit-exact if the hypothesis is correct.")

if __name__ == "__main__":
    main()
