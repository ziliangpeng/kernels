#!/usr/bin/env python3
"""
Cross-batch GEMM determinism across different matrix sizes.
Tests M=1 (decode) and M=128 (prefill) with various K, N dimensions.

Usage:
    python size_sweep.py                    # all dtypes
    python size_sweep.py --dtype bf16        # single dtype
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

# Test configurations: (label, M, K, N)
# M=1 = decode (GEMV path), M=128 = prefill (tiled path)
SHAPES = [
    ("decode-small",    1, 1024,  1024),
    ("decode-med",      1, 4096,  4096),
    ("decode-large",    1, 16384, 4096),
    ("decode-wide",     1, 4096,  16384),
    ("prefill-small",   128, 1024,  1024),
    ("prefill-med",     128, 4096,  4096),
    ("prefill-large",   128, 16384, 4096),
    ("prefill-wide",    128, 4096,  16384),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()))
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"HIP: {torch.version.hip or 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dtypes = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())

    for dtype_name in dtypes:
        dtype = DTYPE_MAP[dtype_name]
        print(f"{'='*90}")
        print(f"  dtype: {dtype_name}")
        print(f"{'='*90}")

        for label, M, K, N in SHAPES:
            W = seeded_tensor(K, N, dtype=dtype, seed=99)
            x1 = seeded_tensor(M, K, dtype=dtype, seed=42)

            # Reference: B=1
            ref_out = torch.mm(x1, W)
            ref_cs = checksum(ref_out)

            print(f"\n  [{label}] M={M} K={K} N={N}")
            print(f"  {'B':>6}  {'Bit-exact?':>12}  {'Rel diff':>10}")
            print(f"  {'-'*6}  {'-'*12}  {'-'*10}")

            for B in BATCH_SIZES:
                x_batch = x1.repeat(B, 1)
                W_batch = W  # same weight for all
                out = torch.mm(x_batch, W_batch)
                slot0 = out[0:M]

                cs = checksum(slot0)
                exact = cs == ref_cs
                diff = rel_diff_pct(ref_out, slot0)
                status = "✅ exact" if exact else "❌ diff"
                print(f"  {B:>6}  {status:>12}  {diff:>9.4f}%")

        print()

    print("Done.")

if __name__ == "__main__":
    main()
