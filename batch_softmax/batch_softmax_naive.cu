#include "batch_softmax_naive.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// NAIVE BATCH SOFTMAX KERNEL - ONE BLOCK PER ROW
// ============================================================================
//
// This kernel assigns one CUDA block to each row of the input matrix.
// Each block computes softmax for its row independently using shared memory
// reductions.
//
// LAUNCH CONFIGURATION:
// - Grid: (batch_size, 1, 1) - blockIdx.x indexes the row
// - Block: (threadsPerBlock, 1, 1) - threads cooperatively process one row
//
// MEMORY LAYOUT:
// - Input/Output: Row-major, contiguous in memory
// - Row i starts at offset (i * dim)
// - Within a row, elements are contiguous (coalesced access when threads
//   access consecutive elements)
//
// ALGORITHM:
// Phase 1: Find maximum value in the row (parallel reduction)
// Phase 2: Compute sum of exp(x_i - max) (parallel reduction)
// Phase 3: Normalize: output_i = exp(x_i - max) / sum
//
// NUMERICAL STABILITY:
// Uses max-subtraction trick to prevent overflow in exp()
// ============================================================================

// Shared memory is allocated externally and split for max and sum reductions
__global__ void batch_softmax_naive_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Pointers to this row's data
    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Dynamic shared memory for reductions
    extern __shared__ float sdata[];

    // ========================================================================
    // PHASE 1: Find maximum value in the row
    // ========================================================================

    // Each thread finds local max over its portion of the row
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += blockSize) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    // Store in shared memory for reduction
    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction for max
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Broadcast max to all threads
    float row_max = sdata[0];
    __syncthreads();

    // ========================================================================
    // PHASE 2: Compute exp(x - max), store in output, and sum
    // ========================================================================

    // Each thread computes exp(x - max) once, stores it, and accumulates sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockSize) {
        float val = expf(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate result
        thread_sum += val;
    }

    // Store in shared memory for reduction
    sdata[tid] = thread_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Broadcast sum to all threads
    float row_sum = sdata[0];
    __syncthreads();

    // ========================================================================
    // PHASE 3: Normalize output (reuse stored exp values)
    // ========================================================================

    float inv_sum = 1.0f / row_sum;  // Multiply is faster than divide
    for (int i = tid; i < dim; i += blockSize) {
        row_output[i] *= inv_sum;  // Normalize the stored exp values
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

NaiveBatchSoftmax::NaiveBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim), threadsPerBlock(threadsPerBlock) {
    // No workspace allocation needed - uses dynamic shared memory
}

void NaiveBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Launch one block per row
    // Shared memory size: threadsPerBlock floats for reduction
    int sharedMemSize = threadsPerBlock * sizeof(float);

    batch_softmax_naive_kernel<<<batch_size, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output, batch_size, dim
    );
    cudaCheckError(cudaGetLastError());
}
