#include "batch_softmax_cub.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>
#include <stdexcept>

// ============================================================================
// CUB-BASED BATCH SOFTMAX: One Block Per Row with CUB Reductions
// ============================================================================
//
// This kernel uses CUB BlockReduce for all reduction operations within a row.
// Each CUDA block processes one row of the batch independently.
//
// KEY DIFFERENCE FROM NAIVE:
// - Naive: Manual tree reduction with explicit shared memory management
// - CUB: Single line BlockReduce call that handles everything optimally
//
// CUB internally:
// 1. Uses warp shuffles within each warp (32 threads -> 1 value)
// 2. Uses minimal shared memory for cross-warp reduction
// 3. Automatically handles power-of-2 vs non-power-of-2 block sizes
//
// TEMPLATE PARAMETER:
// CUB requires the block size as a template parameter (compile-time constant).
// We support common block sizes: 128, 256, 512.
//
// ============================================================================

// Template kernel for batch softmax using CUB BlockReduce
template<int BLOCK_SIZE>
__global__ void batch_softmax_cub_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    // CUB BlockReduce type definition
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

    // Allocate shared memory for CUB's temporary storage
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Pointers to this row's data
    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;

    // ========================================================================
    // PHASE 1: Find maximum value using CUB BlockReduce
    // ========================================================================

    // Each thread finds local max over its portion of the row
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    // CUB reduction: 256 threads -> 1 thread (lane 0) in one call!
    float row_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    // Broadcast to all threads via shared memory
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    row_max = shared_max;

    // ========================================================================
    // PHASE 2: Compute exp(x - max), store in output, and sum
    // ========================================================================

    // Each thread computes exp(x - max) once, stores it, and accumulates sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = expf(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate result
        thread_sum += val;
    }

    // Note: We can reuse temp_storage after __syncthreads()
    float row_sum = BlockReduce(temp_storage).Sum(thread_sum);

    // Broadcast to all threads
    __shared__ float shared_sum;
    if (tid == 0) {
        shared_sum = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum;

    // ========================================================================
    // PHASE 3: Normalize output (reuse stored exp values)
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        row_output[i] *= inv_sum;  // Normalize the stored exp values
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

CubBatchSoftmax::CubBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim), threadsPerBlock(threadsPerBlock) {
    // Validate block size (CUB requires template parameter to match)
    if (threadsPerBlock != 128 && threadsPerBlock != 256 && threadsPerBlock != 512) {
        fprintf(stderr, "Error: CUB batch softmax only supports block sizes 128, 256, or 512.\n");
        fprintf(stderr, "       Requested block size: %d\n", threadsPerBlock);
        throw std::invalid_argument("Unsupported block size for CUB batch softmax");
    }
}

void CubBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Launch one block per row, using appropriate template instantiation
    if (threadsPerBlock == 256) {
        batch_softmax_cub_kernel<256><<<batch_size, 256>>>(
            d_input, d_output, batch_size, dim);
    } else if (threadsPerBlock == 128) {
        batch_softmax_cub_kernel<128><<<batch_size, 128>>>(
            d_input, d_output, batch_size, dim);
    } else if (threadsPerBlock == 512) {
        batch_softmax_cub_kernel<512><<<batch_size, 512>>>(
            d_input, d_output, batch_size, dim);
    }
    cudaCheckError(cudaGetLastError());
}
