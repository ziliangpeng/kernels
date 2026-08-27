#!/usr/bin/env python3
"""
Batched GEMM determinism test.

Tests torch.bmm [B, M, K] × [B, K, N] → [B, M, N] for:
1. Cross-batch: same input at B=1 vs B=2,4,8... — does batch slot 0 produce same output?
2. Run-to-run: same B, run 100 times — any drift?
3. Cross-slot: within one run, are all B slots treated identically (given identical input)?

Also tests torch.mm with M treated as batch (for comparison with bmm).

Usage:
    python batched_gemm.py                    # all dtypes
    python batched_gemm.py --dtype fp32        # single dtype
    python batched_gemm.py --n 100             # runs for run-to-run (default 100)
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

def seeded_tensor(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--n", type=int, default=100, help="runs for run-to-run test")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda or 'N/A (ROCm)'}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dtypes = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())
    M, K, N = 128, 4096, 4096  # fixed inner dimensions
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for dtype_name in dtypes:
        dtype = DTYPE_MAP[dtype_name]
        print(f"{'='*80}")
        print(f"  dtype: {dtype_name}   M={M}  K={K}  N={N}   runs={args.n}")
        print(f"{'='*80}")

        # For bmm: each batch slot has its own weight. We make all weights identical
        # so batch slot 0 should always produce the same output regardless of B.
        W_single = seeded_tensor(K, N, dtype=dtype, seed=99)  # [K, N]

        # Reference: B=1 bmm
        x1 = seeded_tensor(1, M, K, dtype=dtype, seed=42)  # [1, M, K]
        W1 = W_single.unsqueeze(0)  # [1, K, N]
        ref_out = torch.bmm(x1, W1)  # [1, M, N]
        ref_row0 = ref_out[0]  # [M, N] — batch slot 0
        ref_cs = checksum(ref_row0)

        print(f"\n  Reference (B=1, slot 0): checksum={ref_cs[:16]}...")

        # ── Test 1: Cross-batch (same input, different B) ──
        print(f"\n  ── Cross-batch: same input at slot 0, different B ──")
        print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Max diff':>12}  {'Checksum':>20}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*20}")

        for B in batch_sizes:
            # All B slots get the same x and W (identical input)
            x_batch = x1.repeat(B, 1, 1)  # [B, M, K] — all slots identical
            W_batch = W1.repeat(B, 1, 1)  # [B, K, N] — all slots identical

            out_batch = torch.bmm(x_batch, W_batch)  # [B, M, N]
            slot0 = out_batch[0]  # [M, N] — batch slot 0

            cs = checksum(slot0)
            bit_exact = (cs == ref_cs)
            diff = max_abs_diff(ref_row0, slot0)

            status = "✅ exact" if bit_exact else "❌ diff"
            print(f"  {B:>6}  {status:>12}  {diff:>12.2e}  {cs[:16]:>20}...")

        # ── Test 2: Cross-slot (within one B, are all slots identical?) ──
        print(f"\n  ── Cross-slot: B=32, all slots identical input, are outputs identical? ──")
        B_test = 32
        x_same = x1.repeat(B_test, 1, 1)
        W_same = W1.repeat(B_test, 1, 1)
        out_same = torch.bmm(x_same, W_same)

        slot_checksums = set()
        max_slot_diff = 0.0
        for b in range(B_test):
            cs = checksum(out_same[b])
            slot_checksums.add(cs)
            if b > 0:
                d = max_abs_diff(out_same[0], out_same[b])
                if d > max_slot_diff:
                    max_slot_diff = d

        n_unique_slots = len(slot_checksums)
        if n_unique_slots == 1:
            print(f"    ✅ All {B_test} slots identical")
        else:
            print(f"    ❌ {n_unique_slots}/{B_test} unique outputs, max slot diff={max_slot_diff:.2e}")

        # ── Test 3: Run-to-run stability ──
        print(f"\n  ── Run-to-run: B=32, {args.n} runs ──")
        unique_runs = set()
        for i in range(args.n):
            out = torch.bmm(x_same, W_same)
            unique_runs.add(checksum(out[0]))

        if len(unique_runs) == 1:
            print(f"    ✅ stable — all {args.n} runs identical")
        else:
            print(f"    ❌ {len(unique_runs)} unique outputs out of {args.n} runs")

        # ── Test 4: bmm vs mm comparison ──
        # Does bmm[B=1] give same result as mm[M, K]×[K, N]?
        print(f"\n  ── bmm vs mm: same math, different API ──")
        x_mm = seeded_tensor(M, K, dtype=dtype, seed=42)  # [M, K]
        out_mm = torch.mm(x_mm, W_single)  # [M, N]

        x_bmm1 = x_mm.unsqueeze(0)  # [1, M, K]
        W_bmm1 = W_single.unsqueeze(0)  # [1, K, N]
        out_bmm1 = torch.bmm(x_bmm1, W_bmm1)[0]  # [M, N]

        cs_mm = checksum(out_mm)
        cs_bmm = checksum(out_bmm1)
        diff_api = max_abs_diff(out_mm, out_bmm1)
        exact_api = cs_mm == cs_bmm
        status = "✅ bit-exact" if exact_api else f"❌ diff={diff_api:.2e}"
        print(f"    mm vs bmm: {status}")

        print()

    print(f"{'='*80}")
    print("Done.")

if __name__ == "__main__":
    main()
