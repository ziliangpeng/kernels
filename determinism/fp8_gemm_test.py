#!/usr/bin/env python3
"""
FP8 (e4m3) GEMM: determinism, batch invariance, and accuracy vs FP32.

Three tests, following the methodology of the earlier FP16/BF16 work:

1. Non-batch determinism : same fp8 gemm 100x, md5 checksum each run,
   report how many runs differ (run-to-run bit-exactness).
2. Batch invariance: compute B=1 reference, then batched runs at
   B in {2, 8, 32, 128}; compare slot 0 output against the B=1
   reference (rel diff %). Mirrors fp16_vs_bf16.py methodology.
   NVIDIA: native batched _scaled_mm. AMD (this ROCm build): 2D-only,
   so test BOTH loop-per-slot AND fold-batch-into-M paths.
3. Accuracy vs FP32: same seeded fp32 inputs; fp32 matmul reference
   (TF32 off) vs fp8 _scaled_mm path; frobenius rel err % + max/p99.

FP8 quantization: per-tensor scale = amax / e4m3_max.
Path: fp32 -> (amax, scale) -> cast fp8 -> _scaled_mm -> out bf16.
_scaled_mm does the dequant (scale_a * scale_b) internally.

Platform quirks (verified 2026-08-27):
- MI325X (gfx942, ROCm): needs float8_e4m3fnuz (bias-128 variant),
  rowwise scales scale_a (M,1) / scale_b (1,N) contiguous, 2D only.
- H100: plain float8_e4m3fn, scalar 1-elem tensor scales OK, 3D batched OK.

Usage:
    python fp8_gemm_test.py --gpu LABEL
"""

import argparse
import json
import time

import torch

E4M3_MAX = 448.0   # fp8_e4m3fn (NVIDIA) max finite
E4M3_FNUZ_MAX = 240.0  # fp8_e4m3fnuz (AMD) max finite
IS_AMD = torch.version.hip is not None
FP8_DTYPE = torch.float8_e4m3fnuz if IS_AMD else torch.float8_e4m3fn

SHAPES = [
    (1, 4096, 4096),
    (8, 4096, 4096),
    (64, 4096, 4096),
    (512, 4096, 4096),
    (4096, 4096, 4096),
    (1, 16384, 4096),
    (4096, 16384, 4096),
]
BATCH_SIZES = [2, 8, 32, 128]
N_DETERMINISM_RUNS = 100


def seeded(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)


def checksum(t):
    import hashlib
    return hashlib.md5(
        t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def rel_diff_pct(a, b):
    abs_diff = (a.float() - b.float()).abs().max().item()
    abs_val = a.float().abs().max().item()
    return 0.0 if abs_val == 0 else abs_diff / abs_val * 100.0


def fp8_quant(t):
    """Per-tensor scale; returns (fp8 tensor, 1-elem fp32 scale tensor)."""
    e4m3_max = E4M3_FNUZ_MAX if IS_AMD else E4M3_MAX
    scale = t.abs().amax().item() / e4m3_max
    scale_t = torch.tensor(scale, dtype=torch.float32, device=t.device)
    return (t / scale).to(
        torch.float8_e4m3fnuz if IS_AMD else torch.float8_e4m3fn), scale_t


def scaled_mm_2d(A32, B32, out_dtype=torch.bfloat16):
    """One 2D fp8 GEMM: A (M,K), B (K,N). Returns fp32 tensor."""
    A8, s_a = fp8_quant(A32)
    B8, s_b = fp8_quant(B32.t().contiguous())
    if IS_AMD:
        B8m = B8.t()  # (K,N) col-major view
        s_a = (torch.ones(A8.shape[0], 1, device=A8.device) * s_a).contiguous()
        s_b = (torch.ones(1, B32.shape[1], device=B32.device) * s_b).contiguous()
        out = torch._scaled_mm(A8, B8m, scale_a=s_a, scale_b=s_b,
                               out_dtype=out_dtype, use_fast_accum=True)
    else:
        out = torch._scaled_mm(A8, B8.t(), scale_a=s_a, scale_b=s_b,
                               out_dtype=out_dtype, use_fast_accum=True)
    return out.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="unknown")
    args = parser.parse_args()
    results = {"gpu": args.gpu, "determinism": [], "batch_invariance": [],
               "accuracy": []}

    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"=== {args.gpu}: FP8 GEMM ({'fnuz' if IS_AMD else 'fn'} e4m3) ===")
    print(f"torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}\n")

    # ---- Test 1: non-batch determinism (100 runs, checksum) ----
    print("--- Test 1: run-to-run determinism (100 runs, non-batch) ---")
    for (M, K, N) in [(1, 4096, 4096), (64, 4096, 4096), (4096, 4096, 4096)]:
        A32, B32 = seeded(M, K), seeded(K, N)
        checksums = set()
        for _ in range(N_DETERMINISM_RUNS):
            out = scaled_mm_2d(A32, B32)
            checksums.add(checksum(out))
        stable = len(checksums) == 1
        n_distinct = len(checksums)
        print(f"  M={M:>5}: {'BIT-EXACT' if stable else f'UNSTABLE ({n_distinct} distinct)'}")
        results["determinism"].append({"M": M, "K": K, "N": N,
                                       "runs": N_DETERMINISM_RUNS,
                                       "distinct_outputs": n_distinct,
                                       "bit_exact": stable})
    print()

    # ---- Test 2: batch invariance ----
    print("--- Test 2: batch invariance (slot0 vs B=1 ref, rel diff %) ---")
    for (M, K, N) in SHAPES:
        A32, B32 = seeded(M, K), seeded(K, N)
        ref = scaled_mm_2d(A32, B32)
        row = {"M": M, "K": K, "N": N}
        cells = []
        # Memory guard: fold path materializes B*M*K fp32 temp. At
        # M=4096 K=16384 B=128 that's 128GB+ -> skip B=128 on the
        # last big shape (B=32 fold result already captured).
        batch_list = BATCH_SIZES[:-1] if (M, K, N) == (4096, 16384, 4096) else BATCH_SIZES
        for B in batch_list:
            if IS_AMD:
                # Path A: loop slots (each 2D)
                out_loop = torch.stack([scaled_mm_2d(A32, B32) for _ in range(B)])
                # Path B: fold batch into M, one big 2D GEMM
                Abig = A32.expand(B, M, K).reshape(B * M, K).contiguous()
                out_fold = scaled_mm_2d(Abig, B32)
                d_loop = rel_diff_pct(ref, out_loop[0])
                d_fold = rel_diff_pct(ref, out_fold[0:M])
                row[f"B{B}_loop_pct"] = d_loop
                row[f"B{B}_fold_pct"] = d_fold
                cells.append(f"B={B}: loop {d_loop:.4f}% fold {d_fold:.4f}%")
            else:
                # torch 2.7 _scaled_mm is 2D-only too; fold batch into M
                Abig = A32.expand(B, M, K).reshape(B * M, K).contiguous()
                out_fold = scaled_mm_2d(Abig, B32)
                d = rel_diff_pct(ref, out_fold[0:M])
                row[f"B{B}_fold_pct"] = d
                cells.append(f"B={B}: fold {d:.4f}%")
        results["batch_invariance"].append(row)
        print(f"  M={M:>5} K={K:>6}: " + "  ".join(cells))
        del ref, A32, B32
        torch.cuda.empty_cache()
    print()

    # ---- Test 3: accuracy vs FP32 reference ----
    print("--- Test 3: accuracy vs FP32 (TF32 off) ---")
    for (M, K, N) in SHAPES:
        A32, B32 = seeded(M, K), seeded(K, N)
        ref32 = torch.matmul(A32, B32)
        out = scaled_mm_2d(A32, B32).float()
        d = (out - ref32).abs()
        rms = ref32.pow(2).mean().sqrt().item()
        m = {
            "max_rel_pct": d.max().item() / ref32.abs().max().item() * 100.0,
            "mean_rel_pct": d.mean().item() / rms * 100.0,
            "fro_rel_pct": (d.pow(2).sum().sqrt().item() /
                            ref32.pow(2).sum().sqrt().item() * 100.0),
            "p99_rel_pct": (d.flatten().quantile(0.99).item() / rms * 100.0),
        }
        results["accuracy"].append({"M": M, "K": K, "N": N, **m})
        print(f"  M={M:>5} K={K:>6} N={N:>5}: max={m['max_rel_pct']:.4f}%  "
              f"mean={m['mean_rel_pct']:.4f}%  fro={m['fro_rel_pct']:.4f}%  "
              f"p99={m['p99_rel_pct']:.4f}%")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"results_fp8_{args.gpu}_{ts}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved: {fname}")


def scaled_mm_2d(A32, B32, out_dtype=torch.bfloat16):
    """2D fp8 GEMM wrapper: A (M,K), B (K,N), returns fp32."""
    A8, s_a = fp8_quant(A32)
    B8, s_b = fp8_quant(B32.t().contiguous())
    if IS_AMD:
        B8m = B8.t()
        s_a2 = (torch.ones(A8.shape[0], 1, device=A8.device) * s_a).contiguous()
        s_b2 = (torch.ones(1, B32.shape[1], device=A8.device) * s_b).contiguous()
        out = torch._scaled_mm(A8, B8m, scale_a=s_a2, scale_b=s_b2,
                               out_dtype=out_dtype, use_fast_accum=True)
    else:
        out = torch._scaled_mm(A8, B8.t(), scale_a=s_a, scale_b=s_b,
                               out_dtype=out_dtype, use_fast_accum=True)
    return out.float()


if __name__ == "__main__":
    main()
