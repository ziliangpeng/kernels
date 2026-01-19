#include "sum_reduce_atomic.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Atomic-based reduction kernel
// Extremely simple but serializes - all threads contend for same result location
__global__ void sumReductionKernel_Atomic(const float *input, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(result, input[idx]);  // Single atomic add - hardware serializes
    }
}

// Atomic method: Simple single-kernel approach using atomicAdd
// All threads directly add to global result - hardware serializes access
float vectorSum_Atomic(const float *d_input, int n, int threadsPerBlock) {
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate result on device, initialize to 0
    float *d_result;
    cudaCheckError(cudaMalloc(&d_result, sizeof(float)));
    cudaCheckError(cudaMemset(d_result, 0, sizeof(float)));

    // Launch atomic kernel (single kernel, no shared memory)
    sumReductionKernel_Atomic<<<numBlocks, threadsPerBlock>>>(
        d_input, d_result, n);
    cudaCheckError(cudaGetLastError());

    // Copy result to host
    float result;
    cudaCheckError(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_result);

    return result;
}
