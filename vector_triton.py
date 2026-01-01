#!/usr/bin/env python3
"""
Triton implementation of vector fused multiply-add (FMA)
Computes: d[i] = a[i] * b[i] + c[i]

Compare with CUDA implementation in vector.cu --mode vma --fused
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n"],  # Auto-tune based on array size
)
@triton.jit
def vector_fma_kernel(
    a_ptr,  # Pointer to input vector a
    b_ptr,  # Pointer to input vector b
    c_ptr,  # Pointer to input vector c
    d_ptr,  # Pointer to output vector d
    n,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused multiply-add kernel with automatic BLOCK_SIZE tuning.
    Triton will benchmark all BLOCK_SIZE configs and pick the fastest.

    Each program instance processes BLOCK_SIZE elements.
    Note: BLOCK_SIZE is elements per program, NOT CUDA threads per block.
    Triton compiler decides actual thread mapping.
    """
    # Get program ID (which chunk are we processing?)
    pid = tl.program_id(0)

    # Calculate element offsets for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary checking (last block might be partial)
    mask = offsets < n

    # Load inputs (automatically vectorized)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # Compute: d = a * b + c
    d = a * b + c

    # Store result (automatically vectorized)
    tl.store(d_ptr + offsets, d, mask=mask)


def vector_fma_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """
    Host function to launch Triton FMA kernel

    Args:
        a, b, c: Input tensors (must be same shape, contiguous, on GPU)

    Returns:
        d: Output tensor (d = a * b + c)
    """
    assert a.is_cuda and b.is_cuda and c.is_cuda, "Inputs must be on GPU"
    assert a.shape == b.shape == c.shape, "Input shapes must match"
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous(), "Inputs must be contiguous"

    n = a.numel()
    d = torch.empty_like(a)

    # Calculate grid size (number of programs to launch)
    # Triton autotune will try different BLOCK_SIZE values
    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # Launch kernel (autotune picks optimal BLOCK_SIZE)
    vector_fma_kernel[grid](a, b, c, d, n)

    return d


def print_kernel_metadata():
    """Print Triton kernel compilation metadata for learning"""
    print("\n===========================================")
    print("Triton Kernel Compilation Metadata:")
    print("===========================================")

    # Get compiled kernel from cache
    if len(vector_fma_kernel.cache) > 0:
        # Cache is dict - get first compiled kernel
        cache_items = list(vector_fma_kernel.cache.items())
        config_key, compiled_kernel = cache_items[0]

        # Basic compilation info
        cuda_threads = compiled_kernel.num_warps * 32
        print(f"  CUDA threads/block: {cuda_threads}")
        print(f"  Warps/block:        {compiled_kernel.num_warps}")

        # Try to get additional metadata if available
        try:
            print(f"  Registers/thread:   {compiled_kernel.n_regs}")
        except AttributeError:
            print("  Registers/thread:   (not available)")

        try:
            print(f"  Shared memory:      {compiled_kernel.shared} bytes")
        except AttributeError:
            print("  Shared memory:      (not available)")

        # Show how Triton mapped BLOCK_SIZE to CUDA threads
        # Config key contains the BLOCK_SIZE
        print("\n  Triton → CUDA Mapping:")
        print(f"    Configs tested:    {len(vector_fma_kernel.cache)}")

        # Print all tested configs (if autotune ran multiple)
        print("\n  Tested configurations:")
        for i, (cfg, kern) in enumerate(cache_items):
            cuda_t = kern.num_warps * 32
            # Try to extract BLOCK_SIZE from config kwargs
            block_sz = "unknown"
            if hasattr(cfg, "kwargs") and "BLOCK_SIZE" in cfg.kwargs:
                block_sz = cfg.kwargs["BLOCK_SIZE"]
            print(f"    Config {i + 1}: BLOCK_SIZE={block_sz}, warps={kern.num_warps}, threads={cuda_t}")

        print("\n  Note: Triton only caches winning config after autotune")
        print("        All 4 configs were benchmarked during warmup")

    else:
        print("  No compiled kernel found in cache")
        print("  (Kernel hasn't been run yet)")

    print("===========================================\n")


def benchmark_triton(n: int, num_iterations: int = 1000):
    """
    Benchmark Triton FMA kernel with auto-tuned BLOCK_SIZE

    Args:
        n: Array size
        num_iterations: Number of timing runs
    """
    print("Triton Vector FMA Benchmark")
    print(f"Array size: {n:,} elements")
    print(f"Iterations: {num_iterations}")
    print("Auto-tuning BLOCK_SIZE...\n")

    # Allocate GPU tensors
    a = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)
    c = torch.rand(n, device="cuda", dtype=torch.float32)

    # Warmup (triggers JIT compilation and auto-tuning)
    print("Warming up (JIT compilation + auto-tuning)...")
    for _ in range(10):
        d = vector_fma_triton(a, b, c)
    torch.cuda.synchronize()
    print("Warmup complete!\n")

    # Print what Triton compiled to
    print_kernel_metadata()

    # Create events ONCE outside loop (avoid overhead)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    timings = []
    for _ in range(num_iterations):
        start.record()
        d = vector_fma_triton(a, b, c)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    # Calculate statistics
    timings = np.array(timings)
    print("\n===========================================")
    print(f"Kernel Execution Statistics ({num_iterations} runs):")
    print("===========================================")
    print(f"  Min:         {np.min(timings):.3f} ms")
    print(f"  Max:         {np.max(timings):.3f} ms")
    print(f"  Mean:        {np.mean(timings):.3f} ms")
    print(f"  Median:      {np.median(timings):.3f} ms")
    print(f"  P90:         {np.percentile(timings, 90):.3f} ms")
    print(f"  P95:         {np.percentile(timings, 95):.3f} ms")
    print(f"  P99:         {np.percentile(timings, 99):.3f} ms")
    print("===========================================\n")

    # Verify correctness
    expected = a * b + c
    if torch.allclose(d, expected, rtol=1e-5):
        print(f"✓ Verification PASSED - All {n:,} elements correct")
    else:
        print("✗ Verification FAILED")
        max_diff = torch.max(torch.abs(d - expected)).item()
        print(f"  Max difference: {max_diff}")

    return np.median(timings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Triton Vector FMA Benchmark with Auto-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vector_triton.py -n 1000000
  python vector_triton.py -n 10000000 -i 500

Compare with CUDA:
  ./vector -n 1000000 --mode vma --fused
  python vector_triton.py -n 1000000
        """,
    )
    parser.add_argument("-n", "--size", type=int, default=1000000, help="Array size (default: 1000000)")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help="Number of iterations (default: 1000)")

    args = parser.parse_args()

    benchmark_triton(args.size, args.iterations)
