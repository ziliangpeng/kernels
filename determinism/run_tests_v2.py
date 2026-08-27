#!/usr/bin/env python3
"""
Kernel determinism test suite v2 — multi-run pairwise comparison.

Runs each kernel N times, records EVERY output checksum, then does:
1. Run-to-run: are all N checksums identical?
2. Pairwise drift: if not identical, how many unique outputs? max pairwise diff?
3. Cross-impl: compare output distributions across implementations.

Usage:
    python run_tests_v2.py                    # all ops, all dtypes
    python run_tests_v2.py --op softmax        # single op
    python run_tests_v2.py --dtype fp32        # single dtype
    python run_tests_v2.py --n 200             # iterations (default 200)
    python run_tests_v2.py --deterministic     # enable deterministic flags
"""

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime
from itertools import combinations

import torch
import torch.nn.functional as F

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def checksum(tensor: torch.Tensor) -> str:
    return hashlib.md5(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()

def seeded_tensor(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

# ── Ops ──────────────────────────────────────────────────────────────────────

def softmax_naive(x):
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def softmax_online(x):
    m = torch.full((x.shape[0], 1), float('-inf'), device=x.device, dtype=x.dtype)
    s = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        xi = x[:, i:i+1]
        new_m = torch.maximum(m, xi)
        s = s * torch.exp(m - new_m) + torch.exp(xi - new_m)
        m = new_m
    return torch.exp(x - m) / s

def softmax_torch(x):
    return F.softmax(x, dim=-1)

def reduction_naive(x):
    result = torch.zeros(1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[0]):
        result += x[i]
    return result

def reduction_torch(x):
    return x.sum()

def reduction_torch_stable(x):
    return x.float().sum().to(x.dtype)

def gemm_torch_mm(a, b):
    return torch.mm(a, b)

def gemm_torch_matmul(a, b):
    return torch.matmul(a, b)

def gemm_torch_addmm(a, b):
    M, K = a.shape
    N = b.shape[1]
    bias = torch.zeros(M, N, device=a.device, dtype=a.dtype)
    return torch.addmm(bias, a, b)

def rmsnorm_naive(x, weight, eps=1e-6):
    ms = (x ** 2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight

def rmsnorm_fused(x, weight, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight

def rmsnorm_fp32_upcast(x, weight, eps=1e-6):
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(var + eps)
    return (x_normed * weight.float()).to(x.dtype)

def layernorm_naive(x, weight, bias, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias

def layernorm_torch(x, weight, bias, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

# ── Op config ────────────────────────────────────────────────────────────────

OPS = {
    "softmax": {
        "shapes": [(128, 4096), (1, 4096), (256, 8192), (1, 128000)],
        "impls": {
            "naive": (softmax_naive, 1),
            "online": (softmax_online, 1),
            "torch": (softmax_torch, 1),
        },
    },
    "reduction": {
        "shapes": [(100000,), (1000000,), (128, 4096)],
        "impls": {
            "naive": (reduction_naive, 1),
            "torch": (reduction_torch, 1),
            "torch_stable": (reduction_torch_stable, 1),
        },
    },
    "gemm": {
        "shapes": [(1024, 1024, 1024), (1, 4096, 4096), (128, 4096, 4096), (4096, 4096, 4096)],
        "impls": {
            "torch_mm": (gemm_torch_mm, 2),
            "torch_matmul": (gemm_torch_matmul, 2),
            "torch_addmm": (gemm_torch_addmm, 2),
        },
    },
    "rmsnorm": {
        "shapes": [(128, 4096), (1, 4096), (256, 8192)],
        "impls": {
            "naive": (rmsnorm_naive, 2),
            "fused": (rmsnorm_fused, 2),
            "fp32_upcast": (rmsnorm_fp32_upcast, 2),
        },
    },
    "layernorm": {
        "shapes": [(128, 4096), (1, 4096), (256, 8192)],
        "impls": {
            "naive": (layernorm_naive, 3),
            "torch": (layernorm_torch, 3),
        },
    },
}

def impl_should_skip(op_name, impl_name, shape, dtype_name):
    """Skip impls that are too slow for large shapes."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    if op_name == "softmax" and impl_name == "online" and total_elements > 100000:
        return True
    if op_name == "reduction" and impl_name == "naive" and total_elements > 500000:
        return True
    if op_name == "softmax" and dtype_name == "fp16" and total_elements > 100000:
        return True
    return False

def make_inputs(op_name, shape, dtype):
    if op_name == "gemm":
        M, K, N = shape
        return (seeded_tensor(M, K, dtype=dtype), seeded_tensor(K, N, dtype=dtype))
    elif op_name == "rmsnorm":
        return (seeded_tensor(*shape, dtype=dtype), seeded_tensor(shape[-1], dtype=dtype, seed=99))
    elif op_name == "layernorm":
        return (seeded_tensor(*shape, dtype=dtype), seeded_tensor(shape[-1], dtype=dtype, seed=99),
                seeded_tensor(shape[-1], dtype=dtype, seed=88))
    else:
        return (seeded_tensor(*shape, dtype=dtype),)

def shape_to_str(shape):
    return "×".join(str(s) for s in shape)

# ── Multi-run test ───────────────────────────────────────────────────────────

def run_multi(fn, args, n):
    """Run fn(*args) n times. Return list of (checksum, output_clone, time_ms)."""
    runs = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        runs.append((checksum(out), out.clone(), (t1 - t0) * 1000))
    return runs

def analyze_runs(op, impl, dtype_name, shape_str, runs):
    """Analyze N runs of a single implementation."""
    checksums = [r[0] for r in runs]
    outputs = [r[1] for r in runs]
    unique_cs = set(checksums)
    n_unique = len(unique_cs)

    # Pairwise max diff across ALL pairs of runs
    max_pairwise = 0.0
    if n_unique > 1:
        for i, j in combinations(range(len(outputs)), 2):
            if checksums[i] != checksums[j]:
                d = max_abs_diff(outputs[i], outputs[j])
                if d > max_pairwise:
                    max_pairwise = d

    return {
        "op": op,
        "impl": impl,
        "dtype": dtype_name,
        "shape": shape_str,
        "test": "run_to_run",
        "n_runs": len(runs),
        "unique_outputs": n_unique,
        "deterministic": n_unique == 1,
        "max_pairwise_diff": max_pairwise,
        "avg_ms": sum(r[2] for r in runs) / len(runs),
    }

def analyze_cross_impl(op, dtype_name, shape_str, ref_name, ref_runs, other_name, other_runs):
    """Compare output distributions across two implementations.
    Uses first run of each as canonical, but also checks if unique output counts differ."""
    ref_outputs = [r[1] for r in ref_runs]
    other_outputs = [r[1] for r in other_runs]

    # Compare run 0 of each impl
    bit_exact = checksum(ref_outputs[0]) == checksum(other_outputs[0])
    diff = max_abs_diff(ref_outputs[0], other_outputs[0])

    # Also check: does impl A run 0 == impl B run 5? etc.
    # If both are run-to-run deterministic and differ, every pair differs by same amount
    cross_max = diff
    if not bit_exact:
        # Sample a few cross-pairs to find max divergence
        n_check = min(10, len(ref_outputs))
        for i in range(n_check):
            for j in range(n_check):
                d = max_abs_diff(ref_outputs[i], other_outputs[j])
                if d > cross_max:
                    cross_max = d

    return {
        "op": op,
        "dtype": dtype_name,
        "shape": shape_str,
        "test": "cross_impl",
        "impl": f"{other_name} vs {ref_name}",
        "bit_exact": bit_exact,
        "max_abs_diff": cross_max,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kernel determinism test suite v2")
    parser.add_argument("--op", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--n", type=int, default=200, help="iterations per test (default 200)")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print("⚠️  Deterministic mode ON")
    else:
        print("🚀 Default mode (no determinism flags)")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iterations per test: {args.n}")
    print()

    dtypes_to_run = [args.dtype] if args.dtype else list(DTYPE_MAP.keys())
    ops_to_run = [args.op] if args.op else list(OPS.keys())
    results = []

    for dtype_name in dtypes_to_run:
        dtype = DTYPE_MAP[dtype_name]
        print(f"{'#'*60}")
        print(f"# dtype: {dtype_name} ({dtype})")
        print(f"{'#'*60}")

        for op_name in ops_to_run:
            if op_name not in OPS:
                continue
            cfg = OPS[op_name]

            for shape in cfg["shapes"]:
                shape_str = shape_to_str(shape)
                print(f"\n  [{dtype_name}] {op_name} shape={shape_str}")

                inputs = make_inputs(op_name, shape, dtype)

                # Run each implementation N times
                all_runs = {}
                for impl_name, (fn, nargs) in cfg["impls"].items():
                    if impl_should_skip(op_name, impl_name, shape, dtype_name):
                        print(f"    → {impl_name}... ⏭️  skip")
                        continue
                    call_args = inputs[:nargs]
                    print(f"    → {impl_name}...", end=" ", flush=True)
                    try:
                        runs = run_multi(fn, call_args, n=args.n)
                        all_runs[impl_name] = runs
                        rr = analyze_runs(op_name, impl_name, dtype_name, shape_str, runs)
                        results.append(rr)
                        if rr["deterministic"]:
                            print(f"✅ stable ({rr['avg_ms']:.3f} ms)")
                        else:
                            print(f"❌ {rr['unique_outputs']}/{args.n} unique, max pairwise diff={rr['max_pairwise_diff']:.2e} ({rr['avg_ms']:.3f} ms)")
                    except Exception as e:
                        print(f"⚠️  error: {e}")

                # Cross-impl analysis
                impl_names = list(all_runs.keys())
                if len(impl_names) >= 2:
                    ref_name = impl_names[0]
                    ref_runs = all_runs[ref_name]
                    for other_name in impl_names[1:]:
                        ci = analyze_cross_impl(op_name, dtype_name, shape_str,
                                                ref_name, ref_runs, other_name, all_runs[other_name])
                        results.append(ci)
                        status = "✅ bit-exact" if ci["bit_exact"] else f"❌ max diff={ci['max_abs_diff']:.2e}"
                        print(f"    {other_name} vs {ref_name}: {status}")

        print()

    # Write CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{ts}.csv"
    fieldnames = set()
    for r in results:
        fieldnames.update(r.keys())
    fieldnames = sorted(fieldnames)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"{'='*60}")
    print(f"Results saved to {csv_path}")
    print(f"Total tests: {len(results)}")

    rr_tests = [r for r in results if r.get("test") == "run_to_run"]
    ci_tests = [r for r in results if r.get("test") == "cross_impl"]
    rr_pass = sum(1 for r in rr_tests if r["deterministic"])
    ci_pass = sum(1 for r in ci_tests if r.get("bit_exact"))
    print(f"\nRun-to-run deterministic: {rr_pass}/{len(rr_tests)}")
    print(f"Cross-impl bit-exact:    {ci_pass}/{len(ci_tests)}")

if __name__ == "__main__":
    main()
