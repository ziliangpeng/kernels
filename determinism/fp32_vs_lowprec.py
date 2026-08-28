#!/usr/bin/env python3
"""
FP32 reference vs FP16 vs BF16 GEMM accuracy comparison.

Measures how far low-precision GEMM output drifts from the FP32 ground
truth, across decode-shaped and prefill-shaped GEMMs.

Methodology:
- One set of seeded FP32 inputs (same on every run, seed=42)
- FP32 reference: torch.matmul in FP32 with TF32 DISABLED (true FP32 cuBLAS)
- FP16 / BF16 runs: inputs CAST down to fp16/bf16, output cast back to
  fp32 for comparison. This mirrors real inference: weights stored in
  low precision, accumulate in FP32 (library default).
- Metrics vs FP32 reference, per (M, K, N, dtype):
    * max relative diff %   : max|a-b| / max|a|  (global)
    * mean abs element diff : mean(|a-b|)
    * relative frobenius err: ||A-B||_F / ||A||_F  (norm-based, more stable)
    * 99th pct element diff %

Usage:
    python fp32_vs_lowprec.py [--gpu LABEL]
"""

import argparse
import json
import time

import torch

SHAPES = [
    # (M, K, N) - decode M=1, small M=8, medium 64, large 512, prefill 4096
    (1, 4096, 4096),
    (8, 4096, 4096),
    (64, 4096, 4096),
    (512, 4096, 4096),
    (4096, 4096, 4096),
    (1, 16384, 4096),   # big-K decode (LSQ-like: 60 layers, big KV/FFN)
    (4096, 16384, 4096),  # big-K prefill
]

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def rel_diff_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict:
    """All-vs-reference diff metrics. ref/out both fp32 on same device."""
    d = (out.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    return {
        # max element diff relative to the largest reference magnitude
        "max_rel_pct": (d.max().item() / ref_abs.max().item() * 100.0),
        # mean of |diff| relative to RMS scale of ref
        "mean_rel_pct": (d.mean().item() / ref.float().pow(2).mean().sqrt().item() * 100.0),
        # frobenius: ||A-B||_F / ||A||_F
        "fro_rel_pct": (d.pow(2).sum().sqrt().item() /
                        ref.float().pow(2).sum().sqrt().item() * 100.0),        # 99th percentile of |diff| relative to ref RMS scale
        "p99_rel_pct": (d.flatten().float().quantile(0.99).item() /
                        ref.float().pow(2).mean().sqrt().item() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="unknown", help="GPU label for reporting")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"

    # Hard-disable TF32 so FP32 run is true FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = []
    for (M, K, N) in SHAPES:
        # Same seeded FP32 inputs for all dtype variants
        g = torch.Generator(device=dev)
        g.manual_seed(42)
        A32 = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g)
        B32 = torch.randn(K, N, device=dev, dtype=torch.float32, generator=g)

        # FP32 ground truth
        ref = torch.matmul(A32, B32)

        for name, dt in DTYPES.items():
            # Real inference path: inputs cast to low precision, library
            # accumulates in FP32 (default True for cublas/hipblaslt)
            A = A32.to(dt)
            B = B32.to(dt)
            out = torch.matmul(A, B).float()
            m = rel_diff_metrics(ref, out)
            results.append({"gpu": args.gpu, "M": M, "K": K, "N": N,
                            "dtype": name, **m})
            print(f"({args.gpu}) M={M:>5} K={K:>6} N={N:>5} {name}: "
                  f"max={m['max_rel_pct']:.4f}%  mean={m['mean_rel_pct']:.4f}%  "
                  f"fro={m['fro_rel_pct']:.4f}%  p99={m['p99_rel_pct']:.4f}%")

    # Summary: fp16 vs bf16 who is closer to FP32?
    print(f"\n=== {args.gpu}: fp16 vs bf16 closeness to FP32 ===")
    for (M, K, N) in SHAPES:
        r = {x["dtype"]: x for x in results
             if (x["M"], x["K"], x["N"]) == (M, K, N)}
        f, b = r["fp16"]["fro_rel_pct"], r["bf16"]["fro_rel_pct"]
        winner = "fp16" if f < b else "bf16"
        print(f"  M={M:>5} K={K:>6}: fp16={f:.4f}%  bf16={b:.4f}%  -> {winner} closer")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"results_fp32_vs_lowprec_{args.gpu}_{ts}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved: {fname}")


if __name__ == "__main__":
    main()
