#include "softmax_online_simple.h"
#include "cuda_utils.h"
#include "elementwise/normalize.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// ONLINE SOFTMAX - SIMPLE THREAD-LEVEL IMPLEMENTATION (EDUCATIONAL)
// ============================================================================
//
// This implementation demonstrates the core online softmax algorithm in the
// clearest possible way, prioritizing understandability over performance.
//
// WHAT IS ONLINE SOFTMAX?
// -----------------------
// Traditional softmax requires multiple passes:
//   Pass 1: Find max(x)
//   Pass 2: Compute sum(exp(x - max))
//   Pass 3: Normalize (exp(x - max) / sum)
//
// Online softmax does this in a SINGLE streaming pass by maintaining
// running statistics (max, sum) that update as we see each element.
//
// THE MAGIC: When max changes, we adjust the sum to remain valid!
//
// CORE ALGORITHM:
// --------------
// m = -∞   // running max
// s = 0    // running sum
//
// for each x[i]:
//   m_old = m
//   m = max(m, x[i])                                // Update max
//   s = s * exp(m_old - m) + exp(x[i] - m)         // Adjust sum!
//
// output[i] = exp(x[i] - m) / s
//
// WHY THIS WORKS (Numerical Stability):
// -------------------------------------
// When we see a new max, all previous exponentials need to be rescaled:
//   exp(x[j] - m_old)  →  exp(x[j] - m_new)
//                      =  exp(x[j] - m_old) * exp(m_old - m_new)
//
// So we multiply the entire sum by exp(m_old - m_new) to keep it valid!
//
// KEY PROPERTY: exp(m_old - m_new) ≤ 1.0, so we never cause overflow
//
// MERGING (max, sum) PAIRS:
// ------------------------
// The tricky part is reduction - how to merge two (max, sum) pairs:
//
//   Given: (max_a, sum_a) and (max_b, sum_b)
//   Want:  (max_merged, sum_merged)
//
// Solution:
//   max_merged = max(max_a, max_b)
//   sum_merged = sum_a * exp(max_a - max_merged) +
//                sum_b * exp(max_b - max_merged)
//
// This is the SAME adjustment we use for streaming!
//
// ============================================================================

// Kernel 1: Block-Level Online Statistics
// ========================================
// Each thread maintains online (max, sum) state, then reduces to block level.

__global__ void onlineSimple_BlockStats(
    const float *input,
    float *block_maxes,
    float *block_sums,
    int n
) {
    extern __shared__ float sdata[];  // Size: blockDim.x * 2

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ========================================================================
    // PHASE 1: Thread-Level Online Accumulation
    // ========================================================================
    // Each thread processes multiple elements via grid-stride loop,
    // maintaining running (max, sum) state.

    float thread_max = -INFINITY;  // Start with smallest possible value
    float thread_sum = 0.0f;

    for (int i = idx; i < n; i += stride) {
        float x = input[i];

        // === ONLINE UPDATE (core algorithm) ===
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, x);  // Update max

        // Adjust sum when max changes:
        // - If max didn't change: exp(old_max - thread_max) = exp(0) = 1.0 (no-op)
        // - If max increased: exp(old_max - thread_max) < 1.0 (rescale down)
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(x - thread_max);
    }

    // ========================================================================
    // PHASE 2: Block-Level Reduction Using Shared Memory
    // ========================================================================
    // Now we need to merge all thread (max, sum) pairs into a single
    // (block_max, block_sum) pair.
    //
    // Shared memory layout:
    //   sdata[0 ... blockDim.x-1]:           max values
    //   sdata[blockDim.x ... 2*blockDim.x-1]: sum values

    // Store thread results in shared memory
    sdata[tid] = thread_max;
    sdata[tid + blockDim.x] = thread_sum;
    __syncthreads();

    // Tree reduction: Merge pairs at each level
    // At each step, we merge (max, sum) pairs using the same online formula
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Load two (max, sum) pairs to merge
            float max_a = sdata[tid];
            float sum_a = sdata[tid + blockDim.x];
            float max_b = sdata[tid + s];
            float sum_b = sdata[tid + s + blockDim.x];

            // === MERGE (max, sum) PAIRS (key operation) ===
            float merged_max = fmaxf(max_a, max_b);

            // Adjust both sums to the merged max:
            // sum_a was computed with max_a, so multiply by exp(max_a - merged_max)
            // sum_b was computed with max_b, so multiply by exp(max_b - merged_max)
            //
            // Special case: If max_a or max_b is -INFINITY, exp(-INF - x) = 0
            // This avoids NaN from -INFINITY - (-INFINITY)
            float merged_sum = (isinf(max_a) ? 0.0f : sum_a * expf(max_a - merged_max)) +
                              (isinf(max_b) ? 0.0f : sum_b * expf(max_b - merged_max));

            // Store merged result
            sdata[tid] = merged_max;
            sdata[tid + blockDim.x] = merged_sum;
        }
        __syncthreads();
    }

    // Thread 0 writes block result to global memory
    if (tid == 0) {
        block_maxes[blockIdx.x] = sdata[0];
        block_sums[blockIdx.x] = sdata[blockDim.x];
    }
}

// Kernel 2: Global Reduce + Normalize
// ====================================
// Reduce all block stats to global (max, sum), then normalize output.
//
// This kernel uses shared memory to:
// 1. Load block stats (grid-stride if many blocks)
// 2. Merge all (max, sum) pairs to single global (max, sum)
// 3. Launch normalization

__global__ void onlineSimple_GlobalReduce(
    const float *block_maxes,
    const float *block_sums,
    float *global_max,
    float *global_sum,
    int numBlocks
) {
    extern __shared__ float sdata[];  // Size: blockDim.x * 2

    int tid = threadIdx.x;

    // ========================================================================
    // PHASE 1: Load Block Stats with Grid-Stride Loop
    // ========================================================================
    // Each thread accumulates multiple block (max, sum) pairs if needed

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // Grid-stride loop over all block statistics
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        float block_max = block_maxes[i];
        float block_sum = block_sums[i];

        // Merge with thread's running state
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, block_max);

        // Avoid NaN from -INFINITY - (-INFINITY)
        thread_sum = (isinf(old_max) ? 0.0f : thread_sum * expf(old_max - thread_max)) +
                    (isinf(block_max) ? 0.0f : block_sum * expf(block_max - thread_max));
    }

    // ========================================================================
    // PHASE 2: Reduce Threads to Single Global (max, sum)
    // ========================================================================

    // Store in shared memory
    sdata[tid] = thread_max;
    sdata[tid + blockDim.x] = thread_sum;
    __syncthreads();

    // Tree reduction (same as Kernel 1)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float max_a = sdata[tid];
            float sum_a = sdata[tid + blockDim.x];
            float max_b = sdata[tid + s];
            float sum_b = sdata[tid + s + blockDim.x];

            float merged_max = fmaxf(max_a, max_b);
            // Avoid NaN from -INFINITY - (-INFINITY)
            float merged_sum = (isinf(max_a) ? 0.0f : sum_a * expf(max_a - merged_max)) +
                              (isinf(max_b) ? 0.0f : sum_b * expf(max_b - merged_max));

            sdata[tid] = merged_max;
            sdata[tid + blockDim.x] = merged_sum;
        }
        __syncthreads();
    }

    // Thread 0 writes final global result
    if (tid == 0) {
        global_max[0] = sdata[0];
        global_sum[0] = sdata[blockDim.x];
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate intermediate buffers
OnlineSimpleSoftmax::OnlineSimpleSoftmax(int n, int threadsPerBlock)
    : n(n), threadsPerBlock(threadsPerBlock) {
    // Calculate grid dimensions
    numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate intermediate buffers (done once, outside timing loop)
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));
}

// Execute: Pure kernel execution (ONLY this is timed in benchmarks)
void OnlineSimpleSoftmax::execute(const float *d_input, float *d_output) {
    // Shared memory size: 2x threadsPerBlock floats (for max and sum)
    size_t sharedMemSize = threadsPerBlock * 2 * sizeof(float);

    // ========================================================================
    // Kernel 1: Compute block-level online statistics
    // ========================================================================
    onlineSimple_BlockStats<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_block_maxes, d_block_sums, n);
    cudaCheckError(cudaGetLastError());

    // ========================================================================
    // Kernel 2: Reduce block stats to global (max, sum)
    // ========================================================================
    // Use single block for simplicity (could optimize for very large numBlocks)
    onlineSimple_GlobalReduce<<<1, threadsPerBlock, sharedMemSize>>>(
        d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    cudaCheckError(cudaGetLastError());

    // ========================================================================
    // Kernel 3: Normalize output using device pointer version (avoids D2H transfer)
    // ========================================================================
    softmaxNormalizeKernel_DevicePtr<<<numBlocks, threadsPerBlock>>>(
        d_input, d_global_max, d_global_sum, d_output, n);
    cudaCheckError(cudaGetLastError());
}

// Destructor: Free intermediate buffers
OnlineSimpleSoftmax::~OnlineSimpleSoftmax() {
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_OnlineSimple(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    OnlineSimpleSoftmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
