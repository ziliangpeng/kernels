#!/usr/bin/env python3
"""
Simple script to run CUDA vs TinyGrad comparison and collect results.
"""

import os
import re
import subprocess

from bazel_tools.tools.python.runfiles import runfiles

sizes = [100000, 1000000, 10000000, 100000000]
size_labels = ["100K", "1M", "10M", "100M"]


def _workspace_name() -> str:
    return os.environ.get("TEST_WORKSPACE") or os.environ.get("BUILD_WORKSPACE_NAME") or "cyborg"


def _rlocation(path: str) -> str:
    runner = runfiles.Create()
    if not runner:
        raise RuntimeError("Bazel runfiles are unavailable; run via `bazel run`.")
    resolved = runner.Rlocation(path)
    if not resolved:
        raise RuntimeError(f"Runfile not found: {path}")
    return resolved


workspace = _workspace_name()
cuda_vector = _rlocation(f"{workspace}/cuda/vector")
cuda_reduce = _rlocation(f"{workspace}/cuda/reduce")
cuda_softmax = _rlocation(f"{workspace}/cuda/softmax/softmax")
tinygrad_bench = _rlocation(f"{workspace}/cuda/tinygrad_comparison")

print("=" * 80)
print("CUDA vs TinyGrad Performance Comparison")
print("Environment: H100 80GB, CUDA 12.4, sm_90")
print("=" * 80)
print()

# VMA Operation
print("Operation 1: VMA (d = a * b + c)")
print("-" * 80)
print(f"{'Size':<10} | {'CUDA Fused':<15} | {'TinyGrad':<15} | {'Slowdown':<10}")
print("-" * 80)

for size, label in zip(sizes, size_labels, strict=True):
    # Run CUDA
    cuda_cmd = [cuda_vector, "-n", str(size), "--mode", "vma", "--fused"]
    cuda_output = subprocess.run(cuda_cmd, capture_output=True, text=True, check=False)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad
    tinygrad_cmd = [tinygrad_bench, "-n", str(size), "-o", "vma", "--iterations", "100"]
    tinygrad_output = subprocess.run(tinygrad_cmd, capture_output=True, text=True, check=False)
    tinygrad_match = re.search(r"Median:\s+([\d.]+)", tinygrad_output.stdout)
    tinygrad_time = float(tinygrad_match.group(1)) if tinygrad_match else None

    if cuda_time and tinygrad_time:
        slowdown = tinygrad_time / cuda_time
        print(f"{label:<10} | {cuda_time:<15.3f} | {tinygrad_time:<15.3f} | {slowdown:<10.1f}x")
    else:
        print(f"{label:<10} | {'ERROR':<15} | {'ERROR':<15} | {'N/A':<10}")

print()

# Reduction Operation
print("Operation 2: Reduction (sum)")
print("-" * 80)
print(f"{'Size':<10} | {'CUDA Warp-opt':<15} | {'TinyGrad':<15} | {'Slowdown':<10}")
print("-" * 80)

for size, label in zip(sizes, size_labels, strict=True):
    # Run CUDA
    cuda_cmd = [cuda_reduce, "-n", str(size), "--method", "threshold", "--warp-opt"]
    cuda_output = subprocess.run(cuda_cmd, capture_output=True, text=True, check=False)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad
    tinygrad_cmd = [tinygrad_bench, "-n", str(size), "-o", "reduce", "--iterations", "100"]
    tinygrad_output = subprocess.run(tinygrad_cmd, capture_output=True, text=True, check=False)
    tinygrad_match = re.search(r"Median:\s+([\d.]+)", tinygrad_output.stdout)
    tinygrad_time = float(tinygrad_match.group(1)) if tinygrad_match else None

    if cuda_time and tinygrad_time:
        slowdown = tinygrad_time / cuda_time
        print(f"{label:<10} | {cuda_time:<15.3f} | {tinygrad_time:<15.3f} | {slowdown:<10.1f}x")
    else:
        print(f"{label:<10} | {'ERROR':<15} | {'ERROR':<15} | {'N/A':<10}")

print()

# Softmax Operation
print("Operation 3: Softmax")
print("-" * 80)
print(f"{'Size':<10} | {'CUDA Multi-pass':<15} | {'TinyGrad':<15} | {'Slowdown':<10}")
print("-" * 80)

for size, label in zip(sizes, size_labels, strict=True):
    # Run CUDA
    cuda_cmd = [cuda_softmax, "-n", str(size), "--method", "multi"]
    cuda_output = subprocess.run(cuda_cmd, capture_output=True, text=True, check=False)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad (extract built-in softmax time)
    tinygrad_cmd = [tinygrad_bench, "-n", str(size), "-o", "softmax", "--iterations", "100"]
    tinygrad_output = subprocess.run(tinygrad_cmd, capture_output=True, text=True, check=False)
    # Get the built-in softmax result (first median after "Built-in")
    tinygrad_lines = tinygrad_output.stdout.split("\n")
    tinygrad_time = None
    found_builtin = False
    for line in tinygrad_lines:
        if "Built-in" in line:
            found_builtin = True
        if found_builtin and "Median:" in line:
            match = re.search(r"Median:\s+([\d.]+)", line)
            if match:
                tinygrad_time = float(match.group(1))
                break

    if cuda_time and tinygrad_time:
        slowdown = tinygrad_time / cuda_time
        print(f"{label:<10} | {cuda_time:<15.3f} | {tinygrad_time:<15.3f} | {slowdown:<10.1f}x")
    else:
        print(f"{label:<10} | {'ERROR':<15} | {'ERROR':<15} | {'N/A':<10}")

print()
print("=" * 80)
print("Benchmark Complete!")
print("=" * 80)
