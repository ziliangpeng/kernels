#!/usr/bin/env python3
"""Test different block sizes for Triton softmax kernel."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_pass1_kernel(input_ptr, partial_max_ptr, partial_sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    block_vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    block_max = tl.max(block_vals, axis=0)
    block_exp = tl.exp(block_vals - block_max)
    block_sum = tl.sum(tl.where(mask, block_exp, 0.0), axis=0)
    tl.store(partial_max_ptr + pid, block_max)
    tl.store(partial_sum_ptr + pid, block_sum)


@triton.jit
def softmax_reduce_kernel(
    partial_max_ptr, partial_sum_ptr, global_max_ptr, global_sum_ptr, n_blocks, BLOCK_SIZE: tl.constexpr
):
    running_max = -float("inf")
    running_sum = 0.0
    for i in range(0, n_blocks, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_blocks
        p_max = tl.load(partial_max_ptr + offsets, mask=mask, other=-float("inf"))
        p_sum = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
        block_max = tl.max(p_max, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        running_sum = running_sum * old_scale
        scales = tl.exp(p_max - new_max)
        scaled_sums = tl.where(mask, p_sum * scales, 0.0)
        running_sum = running_sum + tl.sum(scaled_sums, axis=0)
        running_max = new_max
    tl.store(global_max_ptr, running_max)
    tl.store(global_sum_ptr, running_sum)


@triton.jit
def softmax_pass2_kernel(input_ptr, output_ptr, global_max_ptr, global_sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    global_max = tl.load(global_max_ptr)
    global_sum = tl.load(global_sum_ptr)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    block_vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    block_exp = tl.exp(block_vals - global_max)
    block_out = block_exp / global_sum
    tl.store(output_ptr + offsets, block_out, mask=mask)


def bench_block_size(n_elements, block_size, iterations=100, warmup=10):
    x_gpu = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    output_gpu = torch.empty_like(x_gpu)

    n_blocks = (n_elements + block_size - 1) // block_size
    partial_max = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    partial_sum = torch.empty(n_blocks, dtype=torch.float32, device="cuda")
    global_max = torch.empty(1, dtype=torch.float32, device="cuda")
    global_sum = torch.empty(1, dtype=torch.float32, device="cuda")

    reduce_block = min(1024, triton.next_power_of_2(n_blocks))

    def run():
        softmax_pass1_kernel[(n_blocks,)](x_gpu, partial_max, partial_sum, n_elements, BLOCK_SIZE=block_size)
        softmax_reduce_kernel[(1,)](partial_max, partial_sum, global_max, global_sum, n_blocks, BLOCK_SIZE=reduce_block)
        softmax_pass2_kernel[(n_blocks,)](x_gpu, output_gpu, global_max, global_sum, n_elements, BLOCK_SIZE=block_size)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iterations):
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return np.median(times)


def main():
    block_sizes = [256, 512, 1024, 2048, 4096, 8192]
    test_sizes = [2**16, 2**20, 2**24, 2**27]  # 64K, 1M, 16M, 128M

    print(f"{'Size':>12} |", " | ".join(f"{bs:>8}" for bs in block_sizes))
    print("-" * 13 + "+" + "+".join("-" * 10 for _ in block_sizes))

    for n in test_sizes:
        results = []
        for bs in block_sizes:
            try:
                t = bench_block_size(n, bs)
                results.append(f"{t:>7.3f}ms")
            except Exception:
                results.append(f"{'ERR':>9}")
        print(f"{n:>12,} |", " | ".join(results))


if __name__ == "__main__":
    main()
