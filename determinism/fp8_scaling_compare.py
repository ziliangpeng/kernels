#!/usr/bin/env python3
"""
FP8 scaling mode comparison: TensorWise vs RowWise.

For each mode, measure:
  - accuracy vs FP32 reference (frobenius rel err %, max, p99)
  - speed (TFLOPS, 20-iter average)

Both H100 (e4m3fn) and MI325X (e4m3fnuz).

TensorWise: per-tensor scale (1,1) for both A and B.
RowWise:    per-row scale_a (M,1), per-col scale_b (1,N).

Usage:
    python fp8_scaling_compare.py --gpu LABEL
"""

import argparse
import json
import time

import torch

E4M3_MAX = 448.0
E4M3_FNUZ_MAX = 240.0
IS_AMD = torch.version.hip is not None
FP8_DTYPE = torch.float8_e4m3fnuz if IS_AMD else torch.float8_e4m3fn
E4M3_MAX_VAL = E4M3_FNUZ_MAX if IS_AMD else E4M3_MAX

SHAPES = [
    (1, 4096, 4096),
    (8, 4096, 4096),
    (64, 4096, 4096),
    (512, 4096, 4096),
    (4096, 4096, 4096),
    (1, 16384, 4096),
    (4096, 16384, 4096),
]
N_WARMUP = 5
N_TIMED = 20


def seeded(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)


def quantize_tensorwise(t):
    """Per-tensor scale: one scalar for entire tensor. Returns (1,1) scale."""
    scale = t.abs().amax().item() / E4M3_MAX_VAL
    s = torch.tensor(scale, dtype=torch.float32, device=t.device).reshape(1, 1)
    return (t / scale).to(FP8_DTYPE), s


def quantize_rowwise_a(t):
    """Per-row scale for A (M,K): scale (M,1), each row gets its own scale."""
    scale = t.abs().amax(dim=1, keepdim=True) / E4M3_MAX_VAL  # (M,1)
    return (t / scale).to(FP8_DTYPE), scale.float()


def quantize_rowwise_b(t):
    """Per-col scale for B (K,N): scale (1,N), each column gets its own scale.
    B is stored as (N,K) col-major for _scaled_mm, so we quantize the
    transposed tensor per-row (which = per-column of original B)."""
    Bt = t.t().contiguous()  # (N,K)
    scale = Bt.abs().amax(dim=1, keepdim=True) / E4M3_MAX_VAL  # (N,1)
    B8 = (Bt / scale).to(FP8_DTYPE)  # (N,K) row-major = (K,N) col-major
    return B8.t(), scale.t().contiguous().float()  # (K,N) col-major view, scale (1,N)


def run_gemm(A8, B8_cm, scale_a, scale_b, mode):
    """Run one FP8 GEMM. B8_cm is col-major (K,N) view."""
    if IS_AMD:
        # tensorwise: both scales (1,1) singletons
        # rowwise: scale_a (M,1), scale_b (1,N), both contiguous
        s_a = scale_a.contiguous()
        s_b = scale_b.contiguous()
        out = torch._scaled_mm(A8, B8_cm, scale_a=s_a, scale_b=s_b,
                               out_dtype=torch.bfloat16, use_fast_accum=True)
    else:
        if mode == "tensorwise":
            s_a = scale_a
            s_b = scale_b
        else:  # rowwise — H100 also accepts (M,1) and (1,N)
            s_a = scale_a.contiguous()
            s_b = scale_b.contiguous()
        out = torch._scaled_mm(A8, B8_cm, scale_a=s_a, scale_b=s_b,
                               out_dtype=torch.bfloat16, use_fast_accum=True)
    return out.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="unknown")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"=== {args.gpu}: FP8 scaling mode comparison ===")
    print(f"torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    print(f"FP8 dtype: {FP8_DTYPE}\n")

    results = []
    for (M, K, N) in SHAPES:
        A32, B32 = seeded(M, K), seeded(K, N)
        ref32 = torch.matmul(A32, B32)  # FP32 reference (TF32 off)

        # Prepare quantized tensors for each mode
        # TensorWise
        A8_tw, s_a_tw = quantize_tensorwise(A32)
        B8_tw, s_b_tw = quantize_tensorwise(B32.t().contiguous())
        B8_tw_cm = B8_tw.t()  # col-major (K,N)

        # RowWise
        A8_rw, s_a_rw = quantize_rowwise_a(A32)       # (M,K) fp8, scale (M,1)
        B8_rw_cm, s_b_rw = quantize_rowwise_b(B32)     # (K,N) col-major, scale (1,N)

        for mode, A8, B8cm, sa, sb in [
            ("tensorwise", A8_tw, B8_tw_cm, s_a_tw, s_b_tw),
            ("rowwise",    A8_rw, B8_rw_cm, s_a_rw, s_b_rw),
        ]:
            # --- Accuracy ---
            out = run_gemm(A8, B8cm, sa, sb, mode)
            d = (out - ref32).abs()
            rms = ref32.pow(2).mean().sqrt().item()
            fro_pct = d.pow(2).sum().sqrt().item() / ref32.pow(2).sum().sqrt().item() * 100.0
            max_pct = d.max().item() / ref32.abs().max().item() * 100.0
            p99_pct = d.flatten().quantile(0.99).item() / rms * 100.0

            # --- Speed ---
            for _ in range(N_WARMUP):
                run_gemm(A8, B8cm, sa, sb, mode)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_TIMED):
                run_gemm(A8, B8cm, sa, sb, mode)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / N_TIMED * 1000
            tflops = 2 * M * K * N / ms * 1e-9

            r = {"gpu": args.gpu, "M": M, "K": K, "N": N, "mode": mode,
                 "fro_pct": round(fro_pct, 4), "max_pct": round(max_pct, 4),
                 "p99_pct": round(p99_pct, 4),
                 "ms": round(ms, 4), "tflops": round(tflops, 1)}
            results.append(r)
            print(f"  M={M:>5} K={K:>6} {mode:12s}: fro={fro_pct:.4f}%  "
                  f"max={max_pct:.4f}%  p99={p99_pct:.4f}%  |  "
                  f"{ms:.3f}ms  {tflops:.1f} TFLOPS")

        del A32, B32, ref32
        torch.cuda.empty_cache()

    print(f"\n=== {args.gpu}: TensorWise vs RowWise summary ===")
    for (M, K, N) in SHAPES:
        tw = next(r for r in results if r["M"]==M and r["K"]==K and r["mode"]=="tensorwise")
        rw = next(r for r in results if r["M"]==M and r["K"]==K and r["mode"]=="rowwise")
        fro_ratio = tw["fro_pct"] / rw["fro_pct"] if rw["fro_pct"] > 0 else float('inf')
        speed_ratio = rw["tflops"] / tw["tflops"] if tw["tflops"] > 0 else 0
        print(f"  M={M:>5} K={K:>6}: fro tw={tw['fro_pct']:.4f}% rw={rw['fro_pct']:.4f}% "
              f"(tw/rw={fro_ratio:.1f}x)  |  speed tw={tw['tflops']:.0f} rw={rw['tflops']:.0f} "
              f"(rw/tw={speed_ratio:.2f}x)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"results_fp8_scaling_{args.gpu}_{ts}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved: {fname}")


if __name__ == "__main__":
    main()
