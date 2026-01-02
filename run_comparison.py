#!/usr/bin/env python3
"""
Simple script to run CUDA vs TinyGrad comparison and collect results.
"""

import re
import subprocess

sizes = [100000, 1000000, 10000000, 100000000]
size_labels = ["100K", "1M", "10M", "100M"]

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
    cuda_cmd = f"./vector -n {size} --mode vma --fused"
    cuda_output = subprocess.run(cuda_cmd, shell=True, capture_output=True, text=True)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad
    tinygrad_cmd = f"python tinygrad_comparison.py -n {size} -o vma --iterations 100"
    tinygrad_output = subprocess.run(tinygrad_cmd, shell=True, capture_output=True, text=True)
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
    cuda_cmd = f"./reduce -n {size} --method threshold --warp-opt"
    cuda_output = subprocess.run(cuda_cmd, shell=True, capture_output=True, text=True)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad
    tinygrad_cmd = f"python tinygrad_comparison.py -n {size} -o reduce --iterations 100"
    tinygrad_output = subprocess.run(tinygrad_cmd, shell=True, capture_output=True, text=True)
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
    cuda_cmd = f"./softmax -n {size} --method multi"
    cuda_output = subprocess.run(cuda_cmd, shell=True, capture_output=True, text=True)
    cuda_match = re.search(r"Median:\s+([\d.]+)", cuda_output.stdout)
    cuda_time = float(cuda_match.group(1)) if cuda_match else None

    # Run TinyGrad (extract built-in softmax time)
    tinygrad_cmd = f"python tinygrad_comparison.py -n {size} -o softmax --iterations 100"
    tinygrad_output = subprocess.run(tinygrad_cmd, shell=True, capture_output=True, text=True)
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
