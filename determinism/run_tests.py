#!/usr/bin/env python3
"""
Kernel determinism test suite.

Tests run-to-run and cross-implementation determinism for common inference kernels.
Designed to run on any NVIDIA GPU with PyTorch + CUDA.

Usage:
    python run_tests.py                    # run all ops, all dtypes, default flags
    python run_tests.py --deterministic     # run with deterministic flags on
    python run_tests.py --op softmax        # run single op
    python run_tests.py --dtype fp32        # run single dtype (fp32, fp16, bf16)
    python run_tests.py --n 500             # number of iterations (default 500)
    python run_tests.py --shapes all        # run all shapes (default: standard)

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

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def checksum(tensor: torch.Tensor) -> str:
    """Return a stable checksum: hex of raw bytes."""
    return hashlib.md5(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()

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

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)

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

def reduction_torch_stable(x: torch.Tensor) -> torch.Tensor:
    """PyTorch sum with upcast for FP16/BF16."""
    return x.float().sum().to(x.dtype)

# ── Op: GEMM ────────────────────────────────────────────────────────────────

def gemm_torch_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)

def gemm_torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)

def gemm_torch_addmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.addmm with zero bias — different cuBLAS algo path."""
    M, K = a.shape
    N = b.shape[1]
    bias = torch.zeros(M, N, device=a.device, dtype=a.dtype)
    return torch.addmm(bias, a, b)

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

def rmsnorm_fp32_upcast(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Upcast to FP32 for reduction, then cast back. Common production pattern."""
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(var + eps)
    return (x_normed * weight.float()).to(x.dtype)

# ── Op: LayerNorm ────────────────────────────────────────────────────────────

def layernorm_naive(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Naive LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias

def layernorm_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

# ── Test runner ──────────────────────────────────────────────────────────────

def run_test(fn, args, n: int = 500):
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

def analyze_run_to_run(op: str, impl: str, dtype: str, shape: str, checksums: list) -> dict:
    unique = len(set(checksums))
    return {
        "op": op,
        "impl": impl,
        "dtype": dtype,
        "shape": shape,
        "test": "run_to_run",
        "n_runs": len(checksums),
        "unique_checksums": unique,
        "deterministic": unique == 1,
    }

def analyze_cross_impl(op: str, dtype: str, shape: str, ref_name: str,
                       other_name: str, ref_output: torch.Tensor,
                       other_output: torch.Tensor) -> dict:
    bit_exact = checksum(ref_output) == checksum(other_output)
    diff = max_abs_diff(ref_output, other_output)
    return {
        "op": op,
        "dtype": dtype,
        "shape": shape,
        "test": "cross_impl",
        "impl": f"{other_name} vs {ref_name}",
        "bit_exact": bit_exact,
        "max_abs_diff": diff,
    }

# ── Op definitions ───────────────────────────────────────────────────────────

# Each op: { name: { shapes: [...], impls: { name: (fn, nargs) } } }
# For GEMM shapes are (M, K, N); for others shapes are the tensor shape.

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

def impl_should_skip_online(shape, dtype_name) -> bool:
    """Skip online softmax for large shapes (sequential loop too slow)."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    return total_elements > 100000

def make_inputs(op_name: str, shape, dtype):
    """Create input tensors for a given op and shape."""
    if op_name == "gemm":
        M, K, N = shape
        a = seeded_tensor(M, K, dtype=dtype)
        b = seeded_tensor(K, N, dtype=dtype)
        return (a, b)
    elif op_name in ("rmsnorm", "layernorm"):
        x = seeded_tensor(*shape, dtype=dtype)
        w = seeded_tensor(shape[-1], dtype=dtype, seed=99)
        if op_name == "layernorm":
            bias = seeded_tensor(shape[-1], dtype=dtype, seed=88)
            return (x, w, bias)
        return (x, w)
    else:
        x = seeded_tensor(*shape, dtype=dtype)
        return (x,)

def shape_to_str(shape) -> str:
    if len(shape) == 3:
        return f"{shape[0]}×{shape[1]}×{shape[2]}"
    return "×".join(str(s) for s in shape)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kernel determinism test suite")
    parser.add_argument("--op", type=str, default=None, help="single op to test")
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP.keys()),
                        help="single dtype to test (fp32, fp16, bf16)")
    parser.add_argument("--n", type=int, default=500, help="iterations per test (default 500)")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic flags")
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
                print(f"❌ Unknown op: {op_name}")
                continue
            cfg = OPS[op_name]

            for shape in cfg["shapes"]:
                shape_str = shape_to_str(shape)
                print(f"\n  [{dtype_name}] {op_name} shape={shape_str}")

                # Skip FP16 for online softmax (overflow risk)
                if op_name == "softmax" and dtype_name == "fp16" and shape[-1] > 8192:
                    print(f"    ⏭️  skip (fp16 online softmax overflow risk)")
                    continue

                inputs = make_inputs(op_name, shape, dtype)

                # Skip slow naive/online impls for large shapes
                skip_impls = set()
                if op_name == "softmax" and impl_should_skip_online(shape, dtype_name):
                    skip_impls.add("online")
                if op_name == "reduction" and shape[0] > 500000:
                    skip_impls.add("naive")  # sequential loop too slow

                # Run each implementation
                outputs = {}
                for impl_name, (fn, nargs) in cfg["impls"].items():
                    if impl_name in skip_impls:
                        print(f"    → {impl_name}... ⏭️  skip (too slow)")
                        continue
                    call_args = inputs[:nargs]
                    print(f"    → {impl_name}...", end=" ", flush=True)
                    try:
                        checksums, first_out, times = run_test(fn, call_args, n=args.n)
                        outputs[impl_name] = first_out
                        rr = analyze_run_to_run(op_name, impl_name, dtype_name, shape_str, checksums)
                        results.append(rr)
                        status = "✅ stable" if rr["deterministic"] else f"❌ {rr['unique_checksums']} unique"
                        avg_ms = sum(times) / len(times) * 1000
                        print(f"{status} ({avg_ms:.3f} ms/iter)")
                    except Exception as e:
                        print(f"⚠️  error: {e}")

                # Cross-impl analysis
                impl_names = list(outputs.keys())
                if len(impl_names) >= 2:
                    ref_name = impl_names[0]
                    ref_out = outputs[ref_name]
                    for other_name in impl_names[1:]:
                        ci = analyze_cross_impl(op_name, dtype_name, shape_str,
                                                ref_name, other_name, ref_out, outputs[other_name])
                        results.append(ci)
                        status = "✅ bit-exact" if ci["bit_exact"] else f"❌ diff={ci['max_abs_diff']:.2e}"
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
