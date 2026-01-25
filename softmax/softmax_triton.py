"""
Triton softmax implementation for benchmarking.

This module provides softmax using a Triton JIT-compiled kernel.
Uses PyTorch for GPU memory management (required by Triton's CUDA driver).
Interface uses NumPy arrays for compatibility with other implementations.

Note: This module requires LD_LIBRARY_PATH to include NVIDIA pip package
library directories. Use cuda_py_binary from //cuda:defs.bzl to run this
with the correct library paths set hermetically by Bazel.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel - computes numerically stable softmax over a 1D array.

    Each program instance processes the entire input (single-row softmax).
    Uses the standard stable softmax algorithm:
      1. Find max for numerical stability
      2. Subtract max and exponentiate
      3. Sum exponentials
      4. Normalize by sum
    """
    # This kernel handles a single row (1D softmax)
    # For simplicity, program_id(0) should be 0 (single program launch)
    row_start = tl.program_id(0) * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load input values (masked load for out-of-bounds safety)
    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float("inf"))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)

    # Subtract max and exponentiate
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)

    # Sum for normalization (masked elements contribute 0 due to exp(-inf) = 0)
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_output = numerator / denominator

    # Store result
    tl.store(output_ptr + row_start + offsets, softmax_output, mask=mask)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax using Triton.

    Args:
        x: 1D float32 NumPy array

    Returns:
        1D float32 NumPy array with softmax output
    """
    # Transfer to GPU using PyTorch
    x_gpu = torch.from_numpy(x).cuda()
    output_gpu = torch.empty_like(x_gpu)

    n_cols = x_gpu.numel()

    # Choose block size (must be power of 2 and >= n_cols for single-row kernel)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Launch kernel (1 program for single row)
    softmax_kernel[(1,)](
        x_gpu,
        output_gpu,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Transfer back to CPU
    return output_gpu.cpu().numpy()


def benchmark(x: np.ndarray, iterations: int = 100, warmup: int = 10) -> list[float]:
    """
    Benchmark softmax kernel execution time only (no D2H transfer).

    Args:
        x: Input array (1D float32)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        List of timing results in milliseconds
    """
    # Transfer to GPU once
    x_gpu = torch.from_numpy(x).cuda()
    output_gpu = torch.empty_like(x_gpu)

    n_cols = x_gpu.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Warmup (triggers JIT compilation)
    for _ in range(warmup):
        softmax_kernel[(1,)](x_gpu, output_gpu, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Benchmark using CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iterations):
        start_event.record()
        softmax_kernel[(1,)](x_gpu, output_gpu, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        end_event.record()
        end_event.synchronize()
        # PyTorch event elapsed time is in milliseconds
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

    return times
