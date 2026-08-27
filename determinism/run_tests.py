#!/usr/bin/env python3
"""
Kernel determinism test suite.

Tests run-to-run and cross-implementation determinism for common inference kernels.
Designed to run on any NVIDIA GPU with PyTorch + CUDA.

Usage:
    python run_tests.py                    # run all tests, default flags
    python run_tests.py --deterministic     # run with deterministic flags on
    python run_tests.py --op softmax        # run single op
    python run_tests.py --n 100             # number of iterations (default 100)

Output: CSV to stdout + results/<timestamp>.csv
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

# ── Helpers ──────────────────────────────────────────────────────────────────

def checksum(tensor: torch.Tensor) -> str:
    """Return a stable checksum: hex of raw bytes."""
    return hashlib.md5(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def seeded_tensor(*shape, dtype=torch.float32, seed=42):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

# ── Op: Softmax ──────────────────────────────────────────────────────────────

def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """Naive softmax: exp(x - max) / sum(exp(x - max)). Row-wise."""
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def softmax_online(x: torch.Tensor) -> torch.Tensor:
    """Online softmax: single pass, rescaling. Numerically stable."""
    m = torch.full((x.shape[0], 1), float('-inf'), device=x.device, dtype=x.dtype)
    s = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        xi = x[:, i:i+1]
        new_m = torch.maximum(m, xi)
        s = s * torch.exp(m - new_m) + torch.exp(xi - new_m)
        m = new_m
    return torch.exp(x - m) / s

# ── Op: Reduction (sum) ─────────────────────────────────────────────────────

def reduction_naive(x: torch.Tensor) -> torch.Tensor:
    """Naive sum via loop."""
    result = torch.zeros(1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[0]):
        result += x[i]
    return result

def reduction_torch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch built-in sum."""
    return x.sum()

# ── Op: GEMM ────────────────────────────────────────────────────────────────

def gemm_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)

def gemm_cublas(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # torch.mm already dispatches to cuBLAS, but force via matmul
    return torch.matmul(a, b)

# ── Op: RMSNorm ─────────────────────────────────────────────────────────────

def rmsnorm_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Naive RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    ms = (x ** 2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight

def rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused-style: compute in one kernel via torch ops (closer to fused)."""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + eps)
    return x_normed * weight

# ── Test runner ──────────────────────────────────────────────────────────────

def run_test(name: str, fn, args, n: int = 100):
    """Run fn(*args) n times, return (checksums, first_output, run_times)."""
    checksums = []
    times = []
    first_output = None
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        cs = checksum(out)
        checksums.append(cs)
        if first_output is None:
            first_output = out.clone()
    return checksums, first_output, times

def analyze_run_to_run(name: str, impl_name: str, checksums: list) -> dict:
    """Check if all checksums are the same (run-to-run deterministic)."""
    unique = len(set(checksums))
    return {
        "op": name,
        "impl": impl_name,
        "test": "run_to_run",
        "n_runs": len(checksums),
        "unique_checksums": unique,
        "deterministic": unique == 1,
    }

def analyze_cross_impl(name: str, ref_output: torch.Tensor, ref_name: str,
                       other_output: torch.Tensor, other_name: str) -> dict:
    """Compare outputs between two implementations."""
    bit_exact = checksum(ref_output) == checksum(other_output)
    diff = max_abs_diff(ref_output, other_output)
    return {
        "op": name,
        "impl": f"{other_name} vs {ref_name}",
        "test": "cross_impl",
        "bit_exact": bit_exact,
        "max_abs_diff": diff,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

OPS = {
    "softmax": {
        "shapes": [(128, 4096)],
        "dtype": torch.float32,
        "impls": {
            "naive": (softmax_naive, 1),  # (fn, num_args) 1 = just x
            "online": (softmax_online, 1),
            "torch": (F.softmax, 1),      # torch built-in (cuDNN/FlashInfer)
        },
        "kwargs": {"dim": -1},  # for F.softmax
    },
    "reduction": {
        "shapes": [(100000,)],
        "dtype": torch.float32,
        "impls": {
            "naive": (reduction_naive, 1),
            "torch": (reduction_torch, 1),
        },
    },
    "gemm": {
        "shapes": [(1024, 1024, 1024)],  # (M, N, K)
        "dtype": torch.float32,
        "impls": {
            "torch_mm": (gemm_torch, 2),
            "torch_matmul": (gemm_cublas, 2),
        },
    },
    "rmsnorm": {
        "shapes": [(128, 4096)],
        "dtype": torch.float32,
        "impls": {
            "naive": (rmsnorm_naive, 2),   # x, weight
            "fused": (rmsnorm_fused, 2),
        },
    },
}

def main():
    parser = argparse.ArgumentParser(description="Kernel determinism test suite")
    parser.add_argument("--op", type=str, default=None, help="single op to test")
    parser.add_argument("--n", type=int, default=100, help="iterations per test")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic flags")
    args = parser.parse_args()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print("⚠️  Deterministic mode ON (CUBLAS_WORKSPACE_CONFIG + torch.use_deterministic_algorithms)")
    else:
        print("🚀 Default mode (no determinism flags)")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iterations per test: {args.n}")
    print()

    results = []
    ops_to_run = [args.op] if args.op else list(OPS.keys())

    for op_name in ops_to_run:
        if op_name not in OPS:
            print(f"❌ Unknown op: {op_name}")
            continue
        cfg = OPS[op_name]
        print(f"{'='*60}")
        print(f"Op: {op_name}")
        print(f"{'='*60}")

        for shape in cfg["shapes"]:
            dtype = cfg["dtype"]
            print(f"\n  Shape: {shape}, dtype: {dtype}")

            # Create inputs
            if op_name == "gemm":
                M, N, K = shape
                a = seeded_tensor(M, K, dtype=dtype)
                b = seeded_tensor(K, N, dtype=dtype)
                inputs = {"a": a, "b": b}
            elif op_name == "rmsnorm":
                x = seeded_tensor(*shape, dtype=dtype)
                w = seeded_tensor(shape[-1], dtype=dtype, seed=99)
                inputs = {"x": x, "weight": w}
            else:
                x = seeded_tensor(*shape, dtype=dtype)
                inputs = {"x": x}

            # Run each implementation
            outputs = {}
            for impl_name, (fn, nargs) in cfg["impls"].items():
                if op_name == "softmax" and impl_name == "torch":
                    call_args = (x, -1)  # F.softmax needs dim
                elif nargs == 1:
                    call_args = (x,)
                elif nargs == 2:
                    if op_name == "rmsnorm":
                        call_args = (x, w)
                    else:
                        call_args = (a, b)
                else:
                    call_args = (x,)

                print(f"  → {impl_name}...", end=" ", flush=True)
                checksums, first_out, times = run_test(f"{op_name}/{impl_name}", fn, call_args, n=args.n)
                outputs[impl_name] = first_out

                # Run-to-run analysis
                rr = analyze_run_to_run(op_name, impl_name, checksums)
                results.append(rr)
                status = "✅ stable" if rr["deterministic"] else f"❌ {rr['unique_checksums']} unique"
                avg_ms = sum(times) / len(times) * 1000
                print(f"{status} ({avg_ms:.3f} ms/iter)")

            # Cross-impl analysis (first impl = reference)
            impl_names = list(outputs.keys())
            if len(impl_names) >= 2:
                ref_name = impl_names[0]
                ref_out = outputs[ref_name]
                print(f"\n  Cross-impl (ref={ref_name}):")
                for other_name in impl_names[1:]:
                    ci = analyze_cross_impl(op_name, ref_out, ref_name, outputs[other_name], other_name)
                    results.append(ci)
                    status = "✅ bit-exact" if ci["bit_exact"] else f"❌ diff={ci['max_abs_diff']:.2e}"
                    print(f"    {other_name}: {status}")

        print()

    # Write CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{ts}.csv"
    # Collect all possible fieldnames
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

    # Summary
    rr_tests = [r for r in results if r.get("test") == "run_to_run"]
    ci_tests = [r for r in results if r.get("test") == "cross_impl"]
    rr_pass = sum(1 for r in rr_tests if r["deterministic"])
    ci_pass = sum(1 for r in ci_tests if r.get("bit_exact"))
    print(f"\nRun-to-run deterministic: {rr_pass}/{len(rr_tests)}")
    print(f"Cross-impl bit-exact:    {ci_pass}/{len(ci_tests)}")

if __name__ == "__main__":
    main()
