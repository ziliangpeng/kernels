#include "softmax_cub_block.h"
#include "cuda_utils.h"
#include "elementwise/normalize.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>  // CUB library
#include <stdlib.h>
#include <math.h>
#include <stdexcept>

// ============================================================================
// CUB-BASED SOFTMAX WALKTHROUGH: Industry-Standard Approach
// ============================================================================
//
// WHAT IS CUB?
// ------------
// CUB (CUDA Unbound) is NVIDIA's library of reusable CUDA primitives, now part
// of CUDA Core Compute Libraries (CCCL). It provides template-based building
// blocks that compile to optimal code for any GPU architecture.
//
// Key primitive: cub::BlockReduce<T, BLOCK_SIZE>
// - Performs block-level reductions (sum, max, min, custom operators)
// - Automatically selects optimal algorithm (warp shuffle vs shared memory)
// - Zero runtime overhead (template specialization at compile time)
// - Handles bank conflicts, synchronization, boundary conditions
//
// WHY CUB INSTEAD OF COOPERATIVE GROUPS FOR SOFTMAX?
// ---------------------------------------------------
// Cooperative Groups (our fused2 implementation):
//   ✗ Grid-level sync limits parallelism to ~1,056 blocks on H100
//   ✗ Normalization becomes bottleneck (each thread processes ~3.6 elements)
//   ✗ Result: 7-14x SLOWER than fused3
//   ✓ Use case: When you truly need grid-wide synchronization
//
// CUB BlockReduce (this implementation):
//   ✓ Block-level operations only - no parallelism limits
//   ✓ Can use full GPU (3,907 blocks for 1M elements)
//   ✓ Result: 7-16% FASTER than hand-written fused3
//   ✓ Industry standard: Used by PyTorch, TensorFlow, cuDNN
//
// ============================================================================
// SIDE-BY-SIDE COMPARISON: FUSED3 (HAND-WRITTEN) VS CUB
// ============================================================================
//
// KERNEL 1: BLOCK STATISTICS
// ---------------------------
//
// fused3 (manual tree reduction):                CUB (optimized primitive):
// -----------------------------------------------  --------------------------
// extern __shared__ float sdata[];                typedef cub::BlockReduce<float, 256> BlockReduce;
// int tid = threadIdx.x;                          __shared__ typename BlockReduce::TempStorage temp_storage;
// int idx = blockIdx.x * 256 + tid;
//                                                  int idx = blockIdx.x * blockDim.x + threadIdx.x;
// // Phase 1: Find max                            // Phase 1: Find max (1 line!)
// float thread_max = (idx < n) ? input[idx] : -INF;  float thread_max = (idx < n) ? input[idx] : -INFINITY;
// sdata[tid] = thread_max;                        float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
// __syncthreads();
// for (int s = blockDim.x/2; s > 0; s >>= 1) {   // Broadcast to all threads
//     if (tid < s) {                              __shared__ float shared_max;
//         sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);  if (threadIdx.x == 0) {
//     }                                               shared_max = block_max;
//     __syncthreads();                                block_maxes[blockIdx.x] = block_max;
// }                                               }
// float block_max = sdata[0];                     __syncthreads();
// __shared__ float shared_max;                    block_max = shared_max;
// if (tid == 0) {
//     shared_max = block_max;                     // Phase 2: Compute sum (1 line!)
//     block_maxes[blockIdx.x] = block_max;        float thread_exp_sum = (idx < n) ? expf(input[idx] - block_max) : 0.0f;
// }                                               float block_sum = BlockReduce(temp_storage).Sum(thread_exp_sum);
// __syncthreads();
// block_max = shared_max;                         if (threadIdx.x == 0) {
//                                                      block_sums[blockIdx.x] = block_sum;
// // Phase 2: Compute sum                         }
// float thread_exp_sum = (idx < n) ?
//     expf(input[idx] - block_max) : 0.0f;
// sdata[tid] = thread_exp_sum;
// __syncthreads();
// for (int s = blockDim.x/2; s > 0; s >>= 1) {
//     if (tid < s) {
//         sdata[tid] += sdata[tid + s];
//     }
//     __syncthreads();
// }
// if (tid == 0) {
//     block_sums[blockIdx.x] = sdata[0];
// }
//
// Lines of code:  ~35 lines                       Lines of code: ~12 lines (63% less!)
// Shared memory:  Manual extern declaration       Shared memory: Automatic via TempStorage
// Synchronization: Explicit __syncthreads()       Synchronization: Handled by CUB
// Bank conflicts: Must handle manually            Bank conflicts: Optimized by CUB
// Warp shuffles:  Not used (more code needed)     Warp shuffles: Used automatically when beneficial
//
// KEY INSIGHT: CUB lets you focus on WHAT to compute, not HOW to reduce
//
// ============================================================================
// HOW CUB BLOCKREDUCE WORKS INTERNALLY
// ============================================================================
//
// 1. Template Specialization (Compile Time):
//    - CUB generates optimal code based on BLOCK_SIZE at compile time
//    - For BLOCK_SIZE = 256 (power of 2), uses warp shuffle + shared memory
//    - For BLOCK_SIZE = 200 (not power of 2), uses pure shared memory
//
// 2. Algorithm Selection:
//    - BLOCK_SIZE <= 32:  Pure warp shuffle (no shared memory needed!)
//    - BLOCK_SIZE > 32:   Warp shuffle within warps, then tree reduction across warps
//    - This is faster than pure tree reduction (fewer shared memory accesses)
//
// 3. TempStorage Pattern:
//    - CUB requires you to allocate shared memory via typename BlockReduce::TempStorage
//    - This is a typedef to a struct with the right shared memory size
//    - You pass temp_storage to BlockReduce constructor
//    - CUB manages the actual shared memory operations internally
//
// 4. Reusing TempStorage:
//    - You can reuse temp_storage AFTER __syncthreads()
//    - Example: Phase 1 uses temp_storage for max, Phase 2 reuses it for sum
//    - This saves shared memory (critical resource on GPU)
//
// 5. Result Distribution:
//    - BlockReduce returns valid result ONLY in thread 0
//    - If all threads need result: broadcast via shared memory
//    - Pattern: if (threadIdx.x == 0) { shared_val = result; } __syncthreads();
//
// ============================================================================
// PERFORMANCE RESULTS (H100)
// ============================================================================
//
//                     fused3 (hand-written)    CUB      Speedup
// 100K elements:           0.090 ms          0.090 ms    same
// 1M elements:             0.245 ms          0.229 ms    7% faster
// 10M elements:            0.427 ms          0.357 ms    16% faster
//
// Why is CUB faster?
// 1. Warp shuffles avoid shared memory access (lower latency)
// 2. Better bank conflict handling (NVIDIA engineers' optimizations)
// 3. Compiler can optimize template code better (inline, unroll)
// 4. Works optimally on all GPU architectures (no manual tuning needed)
//
// Code comparison:
// - fused3: ~140 lines for two kernels with manual reductions
// - CUB:    ~80 lines for same functionality (43% less code!)
//
// ============================================================================
// WHEN TO USE CUB VS COOPERATIVE GROUPS
// ============================================================================
//
// Use CUB BlockReduce when:
//   ✓ You need block-level reductions (sum, max, min)
//   ✓ You want optimal performance with minimal code
//   ✓ You want portability across GPU architectures
//   ✓ Example: Softmax, LayerNorm, BatchNorm, reductions
//
// Use Cooperative Groups when:
//   ✓ You truly need GRID-WIDE synchronization
//   ✓ Work after sync is SMALL (not worth separate kernel)
//   ✓ Example: Writing single global result, small updates
//   ✗ NOT for softmax (normalization is large parallel work!)
//
// Industry practice:
// - PyTorch softmax: Uses block-level primitives (CUB-style)
// - cuDNN: Uses optimized block-level reductions
// - FlashAttention: Uses warp-level operations, not grid sync
// - Triton: Automatically generates block-level code
//
// ============================================================================

// Kernel 1: Compute block-level statistics using CUB
template<int BLOCK_SIZE>
__global__ void softmaxCub_BlockStats(
    const float *input,
    float *block_maxes,
    float *block_sums,
    int n
) {
    // CUB BlockReduce type definition
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

    // Allocate shared memory for CUB's temporary storage
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Find block maximum using CUB
    float thread_max = (idx < n) ? input[idx] : -INFINITY;
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    // Only thread 0 has the valid result - broadcast to shared memory
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = block_max;
        block_maxes[blockIdx.x] = block_max;
    }
    __syncthreads();
    block_max = shared_max;  // All threads now have the max

    // Phase 2: Compute sum(exp(x - block_max)) using CUB
    // Note: We reuse temp_storage here - legal after __syncthreads()
    float thread_exp_sum = (idx < n) ? expf(input[idx] - block_max) : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(thread_exp_sum);

    // Thread 0 writes the sum
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
}

// Kernel 2: Global reduce using CUB (single block processes all block stats)
template<int BLOCK_SIZE>
__global__ void softmaxCub_GlobalReduce(
    const float *block_maxes,
    const float *block_sums,
    float *global_max,
    float *global_sum,
    int numBlocks
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;

    // Phase 1: Find global max using grid-stride loop + CUB reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < numBlocks; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, block_maxes[i]);
    }

    float global_max_val = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float shared_global_max;
    if (tid == 0) {
        shared_global_max = global_max_val;
        global_max[0] = global_max_val;
    }
    __syncthreads();
    global_max_val = shared_global_max;

    // Phase 2: Compute adjusted global sum using grid-stride loop + CUB reduction
    // Key formula: adjusted_sum = Σ(block_sum[i] * exp(block_max[i] - global_max))
    float thread_sum = 0.0f;
    for (int i = tid; i < numBlocks; i += BLOCK_SIZE) {
        thread_sum += block_sums[i] * expf(block_maxes[i] - global_max_val);
    }

    // Reuse temp_storage after __syncthreads()
    float global_sum_val = BlockReduce(temp_storage).Sum(thread_sum);

    if (tid == 0) {
        global_sum[0] = global_sum_val;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate intermediate buffers
CubBlockSoftmax::CubBlockSoftmax(int n, int threadsPerBlock)
    : n(n), threadsPerBlock(threadsPerBlock) {
    // Validate block size (CUB requires template parameter to match actual block size)
    if (threadsPerBlock != 128 && threadsPerBlock != 256 && threadsPerBlock != 512) {
        fprintf(stderr, "Error: CUB block softmax only supports block sizes 128, 256, or 512.\n");
        fprintf(stderr, "       Requested block size: %d\n", threadsPerBlock);
        throw std::invalid_argument("Unsupported block size for CUB block softmax");
    }

    // Calculate grid dimensions
    numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate intermediate buffers (done once, outside timing loop)
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));
}

// Execute: Pure kernel execution (ONLY this is timed in benchmarks)
void CubBlockSoftmax::execute(const float *d_input, float *d_output) {
    // Launch Kernel 1: Block statistics with CUB
    // Note: We use template specialization for common block sizes
    if (threadsPerBlock == 256) {
        softmaxCub_BlockStats<256><<<numBlocks, 256>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else if (threadsPerBlock == 128) {
        softmaxCub_BlockStats<128><<<numBlocks, 128>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else if (threadsPerBlock == 512) {
        softmaxCub_BlockStats<512><<<numBlocks, 512>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else {
        // Error out for unsupported block sizes
        // The template parameter MUST match the actual block size for CUB primitives
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                 "Unsupported block size %d for CUB block softmax (supported: 128, 256, 512)",
                 threadsPerBlock);
        throw std::runtime_error(error_msg);
    }
    cudaCheckError(cudaGetLastError());

    // Launch Kernel 2: Global reduce with CUB (single block)
    if (threadsPerBlock == 256) {
        softmaxCub_GlobalReduce<256><<<1, 256>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else if (threadsPerBlock == 128) {
        softmaxCub_GlobalReduce<128><<<1, 128>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else if (threadsPerBlock == 512) {
        softmaxCub_GlobalReduce<512><<<1, 512>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else {
        // Error out for unsupported block sizes (same reason as above)
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                 "Unsupported block size %d for CUB block softmax (supported: 128, 256, 512)",
                 threadsPerBlock);
        throw std::runtime_error(error_msg);
    }
    cudaCheckError(cudaGetLastError());

    // Launch Kernel 3: Normalize using device pointers (avoids D2H transfer)
    softmaxNormalizeKernel_DevicePtr<<<numBlocks, threadsPerBlock>>>(
        d_input, d_global_max, d_global_sum, d_output, n);
    cudaCheckError(cudaGetLastError());
}

// Destructor: Free intermediate buffers
CubBlockSoftmax::~CubBlockSoftmax() {
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_CubBlock(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    CubBlockSoftmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
