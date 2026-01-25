#!/usr/bin/env python3
"""
Python Benchmark CLI for Softmax: CUDA vs TinyGrad

Usage:
    python bench.py -n 1000000 -i 100 -m all
    python bench.py -n 1000000 -m cuda
    python bench.py -n 1000000 -m tinygrad
    python bench.py --sweep                      # Run all sizes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles


def _rlocation(path: str) -> str:
    """Locate a runfile using Bazel's runfiles API."""
    runner = runfiles.Create()
    if not runner:
        raise RuntimeError("Bazel runfiles unavailable; run via `bazel run`.")
    resolved = runner.Rlocation(path)
    if not resolved:
        raise RuntimeError(f"Runfile not found: {path}")
    return resolved


def calculate_statistics(times: list[float]) -> dict:
    """Calculate statistics from timing results."""
    times_np = np.array(times)
    return {
        "min": float(np.min(times_np)),
        "max": float(np.max(times_np)),
        "mean": float(np.mean(times_np)),
        "median": float(np.median(times_np)),
        "p50": float(np.percentile(times_np, 50)),
        "p90": float(np.percentile(times_np, 90)),
        "p95": float(np.percentile(times_np, 95)),
        "p99": float(np.percentile(times_np, 99)),
    }


def print_comparison_table(results: dict, size: int, iterations: int):
    """Print comparison table for all benchmarked methods."""
    print()
    print("=" * 80)
    print("Softmax Benchmark: CUDA vs TinyGrad")
    print(f"Size: {size:,} elements | Iterations: {iterations}")
    print("=" * 80)

    # Table header
    print(f"{'Method':<20} | {'P50 (ms)':>10} | {'P90 (ms)':>10} | {'P99 (ms)':>10} | {'vs CUDA':>10}")
    print("-" * 20 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10)

    # Get CUDA baseline for comparison
    cuda_p50 = results.get("cuda", {}).get("p50", 1.0)
    if cuda_p50 == 0:
        cuda_p50 = 1.0  # Avoid division by zero

    # Print results in order
    method_order = ["cuda", "tinygrad"]
    method_names = {
        "cuda": "CUDA (online warp)",
        "tinygrad": "TinyGrad",
    }

    for method in method_order:
        if method in results:
            stats = results[method]
            ratio = stats["p50"] / cuda_p50 if method != "cuda" else 1.0
            print(
                f"{method_names[method]:<20} | {stats['p50']:>10.3f} | {stats['p90']:>10.3f} | "
                f"{stats['p99']:>10.3f} | {ratio:>9.1f}x"
            )

    print("=" * 80)


def verify_correctness(result: np.ndarray, name: str) -> bool:
    """Verify softmax output is correct (sums to ~1.0, all values in [0, 1])."""
    total = result.sum()
    all_positive = (result >= 0).all()
    all_le_one = (result <= 1).all()

    is_correct = abs(total - 1.0) < 1e-5 and all_positive and all_le_one

    if not is_correct:
        print(f"  WARNING: {name} verification failed!")
        print(f"    Sum: {total:.6f} (expected ~1.0)")
        print(f"    Min: {result.min():.6f}, Max: {result.max():.6f}")
    else:
        print(f"  {name}: Sum={total:.6f}, Min={result.min():.6e}, Max={result.max():.6e} [OK]")

    return is_correct


def benchmark_cuda(x: np.ndarray, iterations: int, warmup: int) -> tuple[dict, np.ndarray]:
    """Benchmark CUDA implementation."""
    try:
        # Use Bazel runfiles to locate the .so extension
        so_path = _rlocation("_main/cuda/softmax/softmax_cuda.so")
        so_dir = str(Path(so_path).parent)
        if so_dir not in sys.path:
            sys.path.insert(0, so_dir)

        import softmax_cuda

        # Run benchmark
        times = softmax_cuda.benchmark(x, iterations, warmup)

        # Get result for verification
        result = softmax_cuda.softmax(x)

        return calculate_statistics(list(times)), result
    except ImportError as e:
        print(f"  CUDA extension not available: {e}")
        return None, None


def benchmark_tinygrad(x: np.ndarray, iterations: int, warmup: int) -> tuple[dict, np.ndarray]:
    """Benchmark TinyGrad implementation."""
    try:
        from cuda.softmax.softmax_tinygrad import benchmark, softmax

        # Run benchmark
        times = benchmark(x, iterations, warmup)

        # Get result for verification
        result = softmax(x)

        return calculate_statistics(times), result
    except ImportError as e:
        print(f"  TinyGrad not available: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Softmax benchmark: CUDA vs TinyGrad")
    parser.add_argument("-n", "--size", type=int, default=1000000, help="Array size (default: 1000000)")
    parser.add_argument(
        "-i", "--iterations", type=int, default=100, help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument(
        "-m",
        "--method",
        choices=["cuda", "tinygrad", "all"],
        default="all",
        help="Method to benchmark (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-verify", action="store_true", help="Skip correctness verification")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run benchmark across multiple sizes (1K, 10K, 100K, 1M, 10M, 100M)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Determine sizes to benchmark
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000] if args.sweep else [args.size]

    # Methods to run
    methods_to_run = ["cuda", "tinygrad"] if args.method == "all" else [args.method]

    # Run sweep mode
    if args.sweep:
        run_sweep(sizes, methods_to_run, args.iterations, args.warmup, args.no_verify)
    else:
        run_single(args.size, methods_to_run, args.iterations, args.warmup, args.no_verify)


def run_single(size: int, methods: list[str], iterations: int, warmup: int, no_verify: bool):
    """Run benchmark for a single size."""
    print(f"Generating input data: {size:,} float32 elements")
    x = np.random.randn(size).astype(np.float32)
    print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")

    results = {}
    verification_results = {}

    for method in methods:
        print(f"\nBenchmarking {method.upper()}...")

        if method == "cuda":
            stats, result = benchmark_cuda(x, iterations, warmup)
        elif method == "tinygrad":
            stats, result = benchmark_tinygrad(x, iterations, warmup)
        else:
            continue

        if stats is not None:
            results[method] = stats
            verification_results[method] = result

            print(f"  P50: {stats['p50']:.3f} ms")
            print(f"  P90: {stats['p90']:.3f} ms")
            print(f"  P99: {stats['p99']:.3f} ms")
            print(f"  Mean: {stats['mean']:.3f} ms")
            print(f"  Min: {stats['min']:.3f} ms, Max: {stats['max']:.3f} ms")

    # Verification
    if not no_verify and verification_results:
        print("\nCorrectness verification:")
        for method, result in verification_results.items():
            if result is not None:
                verify_correctness(result, method.upper())

    # Print comparison table if multiple methods were run
    if len(results) > 1:
        print_comparison_table(results, size, iterations)
    elif len(results) == 1:
        print("\n" + "=" * 50)
        method = list(results.keys())[0]
        stats = results[method]
        print(f"{method.upper()} Results (n={size:,}, iterations={iterations})")
        print("=" * 50)
        for key, value in stats.items():
            print(f"  {key:<8}: {value:.3f} ms")
        print("=" * 50)


def run_sweep(sizes: list[int], methods: list[str], iterations: int, warmup: int, no_verify: bool):
    """Run benchmark across multiple sizes and print summary table."""
    all_results = {}  # {size: {method: stats}}

    for size in sizes:
        print(f"\n{'=' * 60}")
        print(f"Size: {size:,} elements")
        print("=" * 60)

        x = np.random.randn(size).astype(np.float32)
        all_results[size] = {}

        for method in methods:
            print(f"  Benchmarking {method.upper()}...", end=" ", flush=True)

            if method == "cuda":
                stats, result = benchmark_cuda(x, iterations, warmup)
            elif method == "tinygrad":
                stats, result = benchmark_tinygrad(x, iterations, warmup)
            else:
                continue

            if stats is not None:
                all_results[size][method] = stats
                print(f"P50: {stats['p50']:.3f} ms")

                # Quick verification
                if not no_verify and result is not None:
                    total = result.sum()
                    if abs(total - 1.0) >= 1e-5:
                        print(f"    WARNING: verification failed (sum={total:.6f})")

    # Print summary table
    print("\n")
    print("=" * 90)
    print("SUMMARY: Softmax Benchmark (CUDA vs TinyGrad)")
    print(f"Iterations: {iterations} | Warmup: {warmup}")
    print("=" * 90)
    print(f"{'Size':>12} | {'CUDA P50 (ms)':>14} | {'TinyGrad P50 (ms)':>18} | {'Speedup':>10}")
    print("-" * 12 + "-+-" + "-" * 14 + "-+-" + "-" * 18 + "-+-" + "-" * 10)

    for size in sizes:
        cuda_p50 = all_results[size].get("cuda", {}).get("p50", float("nan"))
        tinygrad_p50 = all_results[size].get("tinygrad", {}).get("p50", float("nan"))

        if cuda_p50 > 0 and tinygrad_p50 > 0:
            speedup = tinygrad_p50 / cuda_p50
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"

        print(f"{size:>12,} | {cuda_p50:>14.3f} | {tinygrad_p50:>18.3f} | {speedup_str:>10}")

    print("=" * 90)


if __name__ == "__main__":
    main()
