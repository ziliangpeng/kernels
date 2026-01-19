#include "max_reduce.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// ============================================================================
// MAX REDUCTION KERNELS
// ============================================================================

// Block-level max reduction kernel using shared memory
__global__ void maxReductionKernel(const float *input, float *partialMaxs, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (or -infinity if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
    __syncthreads();

    // Tree reduction with max operator
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes this block's result
    if (tid == 0) {
        partialMaxs[blockIdx.x] = sdata[0];
    }
}

// Warp-optimized max reduction kernel
// Uses shared memory for 256→32, then warp shuffles for 32→1
__global__ void maxReductionKernel_Warp(const float *input, float *partialMaxs, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (or -infinity if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
    __syncthreads();

    // Part 1: Shared memory reduction (256 → 64) with max operator
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Explicit stride=32 reduction (64 → 32)
    if (tid < 32) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
    }
    __syncthreads();

    // Part 2: Warp-level reduction for final 32 → 1 using shuffles
    if (tid < 32) {
        float val = sdata[tid];
        // Use max shuffle instead of addition
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));

        if (tid == 0) {
            partialMaxs[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// MAX REDUCTION WRAPPER FUNCTIONS
// ============================================================================

// Internal helper: Fully GPU recursive max reduction (with optional warp optimization)
static float vectorMax_GPU_internal(const float *d_input, int n, int threadsPerBlock, bool useWarpOpt) {
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
            maxReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        } else {
            maxReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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

// Public API: Fully GPU recursive max reduction
float vectorMax_GPU(const float *d_input, int n, int threadsPerBlock) {
    return vectorMax_GPU_internal(d_input, n, threadsPerBlock, false);
}

// Public API: Warp-optimized version
float vectorMax_GPU_Warp(const float *d_input, int n, int threadsPerBlock) {
    return vectorMax_GPU_internal(d_input, n, threadsPerBlock, true);
}
