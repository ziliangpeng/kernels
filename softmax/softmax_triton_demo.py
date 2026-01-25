#!/usr/bin/env python3
"""
Simple demo of Triton softmax kernel.

Usage:
    bazel run //cuda/softmax:softmax_triton_demo -- -n 1024

Note: This requires LD_LIBRARY_PATH to include the nvidia CUDA libraries.
      Use the wrapper script or set up the environment manually.
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Triton Softmax Demo")
    parser.add_argument("-n", "--size", type=int, default=1024, help="Array size")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Triton Softmax Demo (using PyTorch for GPU memory)")
    print(f"Array size: {args.size:,} elements")
    print()

    # Generate random input
    x = np.random.randn(args.size).astype(np.float32)
    print(f"Input: min={x.min():.3f}, max={x.max():.3f}")

    # Import and run Triton softmax
    from cuda.softmax.softmax_triton import benchmark, softmax

    # Compute softmax
    result = softmax(x)

    # Verify correctness
    result_sum = result.sum()
    print(f"Output: sum={result_sum:.6f} (expected ~1.0)")
    print(f"Output: min={result.min():.6e}, max={result.max():.6e}")

    if abs(result_sum - 1.0) < 1e-5:
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")

    # Compare with NumPy reference
    def numpy_softmax(x):
        x_max = x.max()
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum()

    ref = numpy_softmax(x)
    max_diff = np.abs(result - ref).max()
    print(f"Max diff from NumPy reference: {max_diff:.6e}")

    # Benchmark
    print(f"\nRunning {args.iterations} benchmark iterations...")
    times = benchmark(x, iterations=args.iterations)
    times_np = np.array(times)

    print(f"  P50: {np.percentile(times_np, 50):.3f} ms")
    print(f"  P90: {np.percentile(times_np, 90):.3f} ms")
    print(f"  P99: {np.percentile(times_np, 99):.3f} ms")
    print(f"  Mean: {np.mean(times_np):.3f} ms")


if __name__ == "__main__":
    main()
