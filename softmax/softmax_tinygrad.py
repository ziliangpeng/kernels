"""
TinyGrad softmax implementation for benchmarking.

This module provides softmax using TinyGrad's built-in operation.
Interface uses NumPy arrays for compatibility with other implementations.
"""

import numpy as np
from tinygrad import Device, TinyJit
from tinygrad.tensor import Tensor


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax using TinyGrad (built-in).

    Args:
        x: 1D float32 NumPy array

    Returns:
        1D float32 NumPy array with softmax output
    """
    t = Tensor(x, device="CUDA")
    result = t.softmax()
    result.realize()
    return result.numpy()


def benchmark(x: np.ndarray, iterations: int = 100, warmup: int = 10) -> list[float]:
    """
    Benchmark softmax kernel execution time only (no D2H transfer).

    Uses TinyJIT to compile the kernel once and replay it without
    rebuilding the computation graph each iteration.

    Args:
        x: Input array (1D float32)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        List of timing results in milliseconds
    """
    import time

    # Get the CUDA device for explicit synchronization
    cuda_device = Device["CUDA"]

    # Create input tensor once - data stays on GPU
    t = Tensor(x, device="CUDA")
    t.realize()  # Ensure input is on GPU
    cuda_device.synchronize()

    # Define JIT-compiled softmax function
    # TinyJIT replays the compiled kernels without rebuilding the graph
    @TinyJit
    def jit_softmax(inp: Tensor) -> Tensor:
        return inp.softmax().realize()

    # Warmup - first calls compile and cache the kernel
    for _ in range(warmup):
        jit_softmax(t)
        cuda_device.synchronize()

    # Benchmark kernel execution only (using JIT-cached kernel)
    times = []
    for _ in range(iterations):
        cuda_device.synchronize()  # Ensure previous work is done
        start = time.perf_counter()
        jit_softmax(t)
        cuda_device.synchronize()  # Wait for kernel to complete
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return times
