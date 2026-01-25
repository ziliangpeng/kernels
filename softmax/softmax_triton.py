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
def softmax_pass1_kernel(
    input_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First pass: compute partial max and sum for each block.

    Each program computes max and sum(exp(x - max)) for its chunk.
    Uses online algorithm within each block for numerical stability.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load block
    block_vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

    # Compute block max
    block_max = tl.max(block_vals, axis=0)

    # Compute sum of exp(x - max) for this block
    block_exp = tl.exp(block_vals - block_max)
    block_sum = tl.sum(tl.where(mask, block_exp, 0.0), axis=0)

    # Store partial results
    tl.store(partial_max_ptr + pid, block_max)
    tl.store(partial_sum_ptr + pid, block_sum)


@triton.jit
def softmax_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    global_max_ptr,
    global_sum_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduce partial max/sum to global max/sum using online algorithm.
    Single program processes all partial results.
    """
    running_max = -float("inf")
    running_sum = 0.0

    for i in range(0, n_blocks, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_blocks

        # Load partial max and sum
        p_max = tl.load(partial_max_ptr + offsets, mask=mask, other=-float("inf"))
        p_sum = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)

        # Process each element with online algorithm
        # For block reduction, we need to handle each element
        block_max = tl.max(p_max, axis=0)
        new_max = tl.maximum(running_max, block_max)

        # Rescale old sum
        old_scale = tl.exp(running_max - new_max)
        running_sum = running_sum * old_scale

        # Rescale and add partial sums
        # Each partial sum needs to be rescaled from its local max to new_max
        scales = tl.exp(p_max - new_max)
        scaled_sums = tl.where(mask, p_sum * scales, 0.0)
        running_sum = running_sum + tl.sum(scaled_sums, axis=0)

        running_max = new_max

    tl.store(global_max_ptr, running_max)
    tl.store(global_sum_ptr, running_sum)


@triton.jit
def softmax_pass2_kernel(
    input_ptr,
    output_ptr,
    global_max_ptr,
    global_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second pass: normalize using global max and sum.
    """
    # Load global stats (same for all programs)
    global_max = tl.load(global_max_ptr)
    global_sum = tl.load(global_sum_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, compute normalized output, store
    block_vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    block_exp = tl.exp(block_vals - global_max)
    block_out = block_exp / global_sum

    tl.store(output_ptr + offsets, block_out, mask=mask)


def _compute_grid(n_elements: int, block_size: int) -> int:
    """Compute number of blocks needed."""
    return (n_elements + block_size - 1) // block_size


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

    n_elements = x_gpu.numel()
    BLOCK_SIZE = 8192
    n_blocks = _compute_grid(n_elements, BLOCK_SIZE)

    # Allocate partial results
    partial_max = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    partial_sum = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    global_max = torch.empty(1, dtype=torch.float32, device="cuda")
    global_sum = torch.empty(1, dtype=torch.float32, device="cuda")

    # Pass 1: compute partial max/sum per block
    softmax_pass1_kernel[(n_blocks,)](x_gpu, partial_max, partial_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reduce: combine partial results
    REDUCE_BLOCK = min(1024, triton.next_power_of_2(n_blocks))
    softmax_reduce_kernel[(1,)](partial_max, partial_sum, global_max, global_sum, n_blocks, BLOCK_SIZE=REDUCE_BLOCK)

    # Pass 2: normalize
    softmax_pass2_kernel[(n_blocks,)](x_gpu, output_gpu, global_max, global_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)

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

    n_elements = x_gpu.numel()
    BLOCK_SIZE = 8192
    n_blocks = _compute_grid(n_elements, BLOCK_SIZE)

    # Allocate partial results
    partial_max = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    partial_sum = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    global_max = torch.empty(1, dtype=torch.float32, device="cuda")
    global_sum = torch.empty(1, dtype=torch.float32, device="cuda")

    REDUCE_BLOCK = min(1024, triton.next_power_of_2(n_blocks))

    def run_softmax():
        softmax_pass1_kernel[(n_blocks,)](x_gpu, partial_max, partial_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        softmax_reduce_kernel[(1,)](partial_max, partial_sum, global_max, global_sum, n_blocks, BLOCK_SIZE=REDUCE_BLOCK)
        softmax_pass2_kernel[(n_blocks,)](x_gpu, output_gpu, global_max, global_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Warmup (triggers JIT compilation)
    for _ in range(warmup):
        run_softmax()
    torch.cuda.synchronize()

    # Benchmark using CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iterations):
        start_event.record()
        run_softmax()
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

    return times
