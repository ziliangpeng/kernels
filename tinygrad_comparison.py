#!/usr/bin/env python3
"""
TinyGrad implementations of vector operations for comparison with CUDA.
Demonstrates abstraction gap and performance trade-offs.
"""

import time

import numpy as np
from tinygrad import Device
from tinygrad.tensor import Tensor

# Ensure GPU usage
Device.DEFAULT = "CUDA"


def benchmark(fn, *args, iterations=1000, warmup=100):
    """
    Benchmark a TinyGrad operation with proper warmup for JIT compilation.
    Returns: (times_list, result)
    """
    print(f"  Warming up ({warmup} iterations)...", end="", flush=True)
    # Warmup - important for JIT compilation
    for _i in range(warmup):
        result = fn(*args)
        result.realize()  # Force execution (TinyGrad is lazy)
        # Force synchronization by retrieving a value (ensures GPU work is complete)
        _ = result.numpy().flatten()[0]  # Read first element to force completion
    print(" done")

    # Benchmark with proper GPU synchronization
    print(f"  Benchmarking ({iterations} iterations)...", end="", flush=True)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn(*args)
        result.realize()  # Force execution
        # Force synchronization by retrieving a value
        _ = result.numpy().flatten()[0]  # Read first element to force GPU completion
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    print(" done")

    return times, result


# ============================================================================
# Operation 1: Vector Multiply-Add (VMA)
# ============================================================================


def vma_tinygrad(a, b, c):
    """Vector multiply-add: d = a * b + c"""
    return a * b + c


# ============================================================================
# Operation 2: Vector Reduction (Sum)
# ============================================================================


def reduction_tinygrad(x):
    """Vector sum reduction"""
    return x.sum()


# ============================================================================
# Operation 3: Softmax
# ============================================================================


def softmax_tinygrad(x):
    """Softmax using built-in operation"""
    return x.softmax()


def softmax_manual(x):
    """
    Manual softmax implementation (mirrors CUDA multi-pass).
    Shows how operations compose in TinyGrad.
    """
    max_val = x.max()
    exp_x = (x - max_val).exp()
    sum_exp = exp_x.sum()
    return exp_x / sum_exp


# ============================================================================
# Benchmarking and Statistics
# ============================================================================


def calculate_statistics(times):
    """Calculate statistics from timing results"""
    times_sorted = sorted(times)
    n = len(times)

    return {
        "min": times_sorted[0],
        "max": times_sorted[-1],
        "mean": sum(times) / n,
        "median": times_sorted[n // 2],
        "p90": times_sorted[int(n * 0.90)],
        "p95": times_sorted[int(n * 0.95)],
        "p99": times_sorted[int(n * 0.99)],
    }


def print_statistics(name, stats):
    """Print statistics in formatted table"""
    print(f"\n{'=' * 50}")
    print(f"{name} - Execution Statistics (1000 runs)")
    print(f"{'=' * 50}")
    print(f"  Min:    {stats['min']:.3f} ms")
    print(f"  Max:    {stats['max']:.3f} ms")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P90:    {stats['p90']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"{'=' * 50}")


# ============================================================================
# Main Benchmark Script
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TinyGrad operations benchmark")
    parser.add_argument("-n", "--size", type=int, default=1000000, help="Array size (default: 1000000)")
    parser.add_argument(
        "-o", "--operation", choices=["vma", "reduce", "softmax", "all"], default="all", help="Operation to benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations (default: 1000)")
    parser.add_argument(
        "--inspect-kernels", action="store_true", help="Enable kernel inspection (prints generated GPU code)"
    )

    args = parser.parse_args()

    # Enable kernel inspection if requested
    if args.inspect_kernels:
        import os

        os.environ["DEBUG"] = "4"
        print("Kernel inspection enabled (DEBUG=4)")

    n = args.size
    print(f"Array size: {n:,} elements")
    print(f"Device: {Device.DEFAULT}")

    # Run benchmarks
    if args.operation in ["vma", "all"]:
        print("\n" + "=" * 50)
        print("VMA Operation: d = a * b + c")
        print("=" * 50)

        # Prepare data
        a = Tensor.randn(n)
        b = Tensor.randn(n)
        c = Tensor.randn(n)

        # Benchmark
        times, result = benchmark(vma_tinygrad, a, b, c, iterations=args.iterations)
        stats = calculate_statistics(times)
        print_statistics("TinyGrad VMA", stats)

        # Verify correctness
        a_np = a.numpy()
        b_np = b.numpy()
        c_np = c.numpy()
        expected = a_np * b_np + c_np
        result_np = result.numpy()
        max_error = np.abs(result_np - expected).max()
        print(f"\nVerification: Max error = {max_error:.2e}")

    if args.operation in ["reduce", "all"]:
        print("\n" + "=" * 50)
        print("Reduction Operation: sum(x)")
        print("=" * 50)

        # Prepare data
        x = Tensor.randn(n)

        # Benchmark
        times, result = benchmark(reduction_tinygrad, x, iterations=args.iterations)
        stats = calculate_statistics(times)
        print_statistics("TinyGrad Reduction", stats)

        # Verify correctness
        x_np = x.numpy()
        expected = x_np.sum()
        result_val = result.numpy()
        error = abs(result_val - expected)
        rel_error = error / abs(expected)
        print(f"\nVerification: Relative error = {rel_error:.2e}")

    if args.operation in ["softmax", "all"]:
        print("\n" + "=" * 50)
        print("Softmax Operation")
        print("=" * 50)

        # Prepare data
        x = Tensor.randn(n)

        # Benchmark built-in softmax
        print("\n--- Built-in softmax ---")
        times, result_builtin = benchmark(softmax_tinygrad, x, iterations=args.iterations)
        stats = calculate_statistics(times)
        print_statistics("TinyGrad Softmax (built-in)", stats)

        # Benchmark manual softmax
        print("\n--- Manual softmax (multi-pass) ---")
        times, result_manual = benchmark(softmax_manual, x, iterations=args.iterations)
        stats = calculate_statistics(times)
        print_statistics("TinyGrad Softmax (manual)", stats)

        # Verify correctness
        result_np = result_builtin.numpy()
        sum_result = result_np.sum()
        print(f"\nVerification: Sum(output) = {sum_result:.6f} (expected 1.0)")
        print(f"              Error = {abs(sum_result - 1.0):.2e}")

        # Check all values in [0, 1]
        print(f"              Min value = {result_np.min():.6f}")
        print(f"              Max value = {result_np.max():.6f}")


if __name__ == "__main__":
    main()
