#include "sum_reduce.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>

// ============================================================================
// SUM REDUCTION KERNELS
// ============================================================================

// Block-level reduction kernel using shared memory
// Each block reduces its elements to a single partial sum
__global__ void sumReductionKernel(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (or 0 if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    // Each iteration: half the threads sum pairs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes this block's result
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// Warp-optimized reduction kernel
// Uses shared memory for 256→32, then warp shuffles for 32→1
// Benchmarked ~8% faster than regular version (eliminates 5 __syncthreads barriers)
__global__ void sumReductionKernel_Warp(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Part 1: Shared memory reduction (256 → 64)
    // Stop before entering warp-level range
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Explicit stride=32 reduction (64 → 32)
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    // Part 2: Warp-level reduction for final 32 → 1
    // Only first warp (threads 0-31) participates
    if (tid < 32) {
        // Load value from shared memory into register
        float val = sdata[tid];

        // Reduce using warp shuffles (no __syncthreads needed!)
        val += __shfl_down_sync(0xffffffff, val, 16);  // 32 → 16
        val += __shfl_down_sync(0xffffffff, val, 8);   // 16 → 8
        val += __shfl_down_sync(0xffffffff, val, 4);   // 8 → 4
        val += __shfl_down_sync(0xffffffff, val, 2);   // 4 → 2
        val += __shfl_down_sync(0xffffffff, val, 1);   // 2 → 1

        // Thread 0 writes final result
        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// SUM REDUCTION WRAPPER FUNCTIONS
// ============================================================================

// Internal helper: Fully GPU recursive reduction (with optional warp optimization)
// Keeps launching kernels until only 1 element remains
static float vectorSum_GPU_internal(const float *d_input, int n, int threadsPerBlock, bool useWarpOpt) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;

    // Keep reducing until we have 1 element
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch reduction kernel (choose based on optimization flag)
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        if (useWarpOpt) {
            sumReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        } else {
            sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        }
        cudaCheckError(cudaGetLastError());

        // Free previous temp buffer (if we allocated it)
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output;
        currentSize = numBlocks;
        allocated = true;
    }

    // Copy final single element to host
    float result;
    cudaCheckError(cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup final buffer
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    return result;
}

// Fully GPU recursive reduction
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock) {
    return vectorSum_GPU_internal(d_input, n, threadsPerBlock, false);
}

// Warp-optimized version: Fully GPU recursive reduction
float vectorSum_GPU_Warp(const float *d_input, int n, int threadsPerBlock) {
    return vectorSum_GPU_internal(d_input, n, threadsPerBlock, true);
}

// Internal helper: GPU with configurable CPU threshold (with optional warp optimization)
// Reduces on GPU until size <= threshold, then finishes on CPU
static float vectorSum_Threshold_internal(const float *d_input, int n, int threadsPerBlock, int cpuThreshold, bool useWarpOpt) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;

    // Keep reducing on GPU while size > threshold
    while (currentSize > cpuThreshold) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch reduction kernel (choose based on optimization flag)
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        if (useWarpOpt) {
            sumReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        } else {
            sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        }
        cudaCheckError(cudaGetLastError());

        // Free previous buffer if we allocated it
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output;
        currentSize = numBlocks;
        allocated = true;
    }

    // Transfer remaining elements to CPU and finish reduction
    float *h_partial = (float*)malloc(currentSize * sizeof(float));
    if (!h_partial) {
        fprintf(stderr, "Failed to allocate host memory for partial results in reduction\n");
        if (allocated) {
            cudaFree((void*)d_current);
        }
        exit(EXIT_FAILURE);
    }
    cudaCheckError(cudaMemcpy(h_partial, d_current, currentSize * sizeof(float),
                               cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    float result = 0.0f;
    for (int i = 0; i < currentSize; i++) {
        result += h_partial[i];
    }

    // Cleanup
    free(h_partial);
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    return result;
}

// GPU with configurable CPU threshold
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold) {
    return vectorSum_Threshold_internal(d_input, n, threadsPerBlock, cpuThreshold, false);
}

// Warp-optimized version: GPU with configurable CPU threshold
float vectorSum_Threshold_Warp(const float *d_input, int n, int threadsPerBlock, int cpuThreshold) {
    return vectorSum_Threshold_internal(d_input, n, threadsPerBlock, cpuThreshold, true);
}
