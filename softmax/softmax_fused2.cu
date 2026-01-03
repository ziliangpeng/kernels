#include "softmax_fused2.h"
#include "softmax_fused3.h"  // Reuse softmaxFused3_BlockStats (regular launch, no grid-stride needed)
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <math.h>

namespace cg = cooperative_groups;

// ============================================================================
// COOPERATIVE GROUPS EDUCATIONAL WALKTHROUGH
// ============================================================================
// This file demonstrates CUDA Cooperative Groups with grid-level synchronization.
// It implements a 2-kernel softmax that is intentionally SLOWER than the 3-kernel
// version to teach important lessons about GPU architecture and parallelism.
//
// WHAT ARE COOPERATIVE GROUPS?
// ----------------------------
// Cooperative Groups (introduced CUDA 9.0) provide flexible thread synchronization:
//
// 1. Thread-block level: cg::thread_block - replaces __syncthreads()
// 2. Warp level: cg::thread_block_tile<32> - warp-shuffle operations
// 3. Grid level: cg::grid_group - synchronize ALL blocks in kernel (NEW!)
//
// Grid-level sync enables patterns impossible with standard CUDA:
//   - Multiple phases in single kernel (compute → sync → use results)
//   - No CPU involvement between phases
//   - Data stays in registers/shared memory across phases
//
// THE CRITICAL CONSTRAINT: All blocks must fit on GPU simultaneously
//   - Standard launch: Use as many blocks as needed (e.g., 3,907 for 1M elements)
//   - Cooperative launch: Limited by hardware (e.g., 1,056 blocks on H100)
//   - Must use cudaLaunchCooperativeKernel() API instead of <<<>>>
//
// WHEN COOPERATIVE GROUPS ARE BENEFICIAL:
// ----------------------------------------
// Research shows cooperative groups excel when:
//
// 1. ✓ Warp/block-level operations (NOT grid-level):
//    - Warp-aggregated atomics: 32x reduction in atomic operations
//    - Block reductions with shared memory
//    - Coalesced memory access patterns
//
// 2. ✓ Grid-level sync for SMALL post-sync work:
//    - Example: Normalization where you write a single result
//    - Grid sync for finding max, then write one normalized value
//    - Acceleware benchmarks: Faster than two-kernel approach
//
// 3. ✗ Grid-level sync for LARGE post-sync work (OUR CASE):
//    - Problem: Need to normalize ALL 1M elements with only 1,056 blocks
//    - Each thread must process ~3.6 elements sequentially
//    - Result: 7-14x SLOWER than full parallelism approach
//
// RESEARCH FINDINGS:
// ------------------
// Lei Mao's benchmarks (RTX 3090): Cooperative groups perform similarly to
//   traditional CUDA for reductions (~882 GB/s for both approaches)
//
// FlashDecoding++ study: Synchronized softmax has 18.8% overhead in LLaMA2-7B
//   Their solution: Avoid grid synchronization entirely with async softmax
//
// NVIDIA forums consensus: "No synchronization operations are 'free'...
//   performance would strongly depend on the specific use case"
//
// Industry practice: Modern softmax implementations (Triton, PyTorch) use
//   single-pass "online softmax" algorithm - NO cooperative groups needed
//
// ============================================================================
// OUR IMPLEMENTATION: 2-KERNEL SOFTMAX WITH COOPERATIVE GROUPS
// ============================================================================
//
// ALGORITHM: softmax(x) = exp(x[i] - max) / sum(exp(x[j] - max))
//
// KERNEL 1: Block-level statistics (REGULAR launch - full parallelism)
//   Input:  d_input[n]
//   Output: block_maxes[numBlocks], block_sums[numBlocks]
//   Launch: Standard <<<numBlocks, threadsPerBlock>>>
//
//   For 1M elements: 3,907 blocks × 256 threads
//   Each block computes:
//     - Local max across its 256 elements
//     - Local sum of exp(x - local_max)
//
// KERNEL 2: Cooperative reduce + normalize (COOPERATIVE launch - limited)
//   Input:  d_input[n], block_maxes[numBlocks], block_sums[numBlocks]
//   Output: d_output[n]
//   Launch: cudaLaunchCooperativeKernel(..., 1056_blocks, 256_threads, ...)
//
//   For 1M elements: Only 1,056 blocks × 256 threads (H100 hardware limit)
//
//   Phase 1: Reduce block_maxes → global_max
//     - Grid-stride loop: for (i = blockIdx; i < 3907; i += 1056)
//     - Block reduction in shared memory
//     - atomicMaxFloat() to global memory
//     - grid.sync() ← ALL blocks wait here
//
//   Phase 2: Reduce block_sums → global_sum (with exponential adjustment)
//     - Grid-stride loop over 3,907 block statistics
//     - Adjustment: block_sum * exp(block_max - global_max)
//     - atomicAdd() to global memory
//     - grid.sync() ← ALL blocks wait here
//
//   Phase 3: Normalize output (THE BOTTLENECK!)
//     - Grid-stride loop: for (i = idx; i < 1M; i += 270K)
//     - Each thread processes ~3.6 elements sequentially
//     - output[i] = exp(input[i] - global_max) / global_sum
//
// KEY TECHNIQUE: atomicMaxFloat()
//   CUDA has no native atomicMax for floats, so we implement with CAS:
//   do {
//     assumed = old;
//     old = atomicCAS(addr, assumed, max(value, assumed));
//   } while (assumed != old);
//
// ============================================================================
// PERFORMANCE COMPARISON
// ============================================================================
//
//                      3-Kernel Fused    2-Kernel Coop    Slowdown
// 100K elements:          0.088 ms          1.22 ms         14x
// 1M elements:            0.208 ms          1.48 ms          7x
// 10M elements:           0.439 ms          1.71 ms          4x
//
// WHY IS IT SLOWER?
// -----------------
// Kernel 1: ✓ Same speed (both use full parallelism)
//
// Kernel 2 Phase 1 & 2: ✓ Fast (reducing 3,907 blocks is quick work)
//   Grid-stride loop works well for small reductions
//
// Kernel 2 Phase 3: ✗ BOTTLENECK (normalizing 1M elements is large work)
//   - 3-kernel approach: 3,907 blocks × 256 = 1M threads (one element each)
//   - 2-kernel approach: 1,056 blocks × 256 = 270K threads (~3.6 elements each)
//   - Sequential work per thread dominates performance
//   - Kernel launch overhead saved (~0.01ms) << normalization loss (~1.2ms)
//
// The lesson: Grid-stride loops are a workaround for hardware constraints,
//            not an optimization. Full parallelism is always faster.
//
// ============================================================================
// WHAT WE LEARNED
// ============================================================================
//
// 1. Grid synchronization != free kernel fusion
//    Cooperative launch trades kernel overhead for parallelism constraints
//
// 2. Hardware limits have cascading effects
//    The 1,056 block limit affects ALL subsequent work in that kernel
//
// 3. Grid-stride loops are a compromise, not a solution
//    Good for: Small reductions where ~1K blocks suffice
//    Bad for: Large parallel work (normalization, element-wise ops)
//
// 4. Fewer kernels ≠ better performance
//    Sometimes 3 kernels with full parallelism >> 2 kernels with constraints
//
// 5. Industry avoids this pattern
//    Modern implementations use "online softmax" (single-pass, no grid sync)
//    FlashAttention: Fuses at warp level, not grid level
//    Triton: Single kernel with per-row parallelism, no cross-row sync
//
// 6. When to use cooperative groups:
//    ✓ Warp-level operations (shuffle, vote, coalesced)
//    ✓ Block-level reductions and barriers
//    ✓ Grid-level sync when post-sync work is SMALL
//    ✗ Grid-level sync when post-sync work is LARGE (like normalization)
//
// RECOMMENDED READING:
// --------------------
// - NVIDIA Cooperative Groups Blog: https://developer.nvidia.com/blog/cooperative-groups/
// - Lei Mao's Benchmarks: https://leimao.github.io/blog/CUDA-Cooperative-Groups/
// - FlashAttention paper: https://arxiv.org/abs/2307.08691
// - Triton Fused Softmax: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
//
// ============================================================================

// Helper: atomicMax for floats (no native support, use compare-and-swap)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                       __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Note: Kernel 1 is reused from softmax_fused3.h (softmaxFused3_BlockStats)
// It uses regular launch (no cooperative constraints) so no grid-stride loop needed
// This allows full parallelism: for 1M elements, use 3,907 blocks instead of 1,056!

// Kernel 2: Fused global reduce + normalize (COOPERATIVE KERNEL)
// ============================================================================
// COOPERATIVE KERNEL: Requires cudaLaunchCooperativeKernel (special launch API)
// ============================================================================
// LIMITATION: All blocks must fit on GPU simultaneously (hard constraint)
//   H100: 132 SMs × 32 blocks/SM = 4,224 max blocks
//   With 256 threads/block: ~1.08M elements max
//   For larger inputs (2M+), use 3-kernel fused softmax instead
// ============================================================================
__global__ void softmaxFused2_ReduceAndNormalize(
    const float *input,
    const float *block_maxes,
    const float *block_sums,
    float *global_max_out,
    float *global_sum_out,
    float *output,
    int n,
    int numBlocks
) {
    // Get grid group for synchronization
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Phase 1: Find global max via grid-stride loop + block reduction
    float thread_max = -INFINITY;
    for (int i = blockIdx.x * blockDim.x + tid; i < numBlocks; i += gridDim.x * blockDim.x) {
        thread_max = fmaxf(thread_max, block_maxes[i]);
    }

    // Block reduction for max
    sdata[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes block's max to global memory atomically
    if (tid == 0) {
        atomicMaxFloat(global_max_out, sdata[0]);
    }

    // Grid-wide barrier: ensure all blocks have computed max
    grid.sync();

    // All threads read the global max
    float global_max = global_max_out[0];

    // Phase 2: Compute adjusted global sum
    // Key formula: adjusted_sum = block_sum * exp(block_max - global_max)
    float thread_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < numBlocks; i += gridDim.x * blockDim.x) {
        thread_sum += block_sums[i] * expf(block_maxes[i] - global_max);
    }

    // Block reduction for sum
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block's sum to global memory atomically
    if (tid == 0) {
        atomicAdd(global_sum_out, sdata[0]);
    }

    // Grid-wide barrier: ensure all blocks have computed sum
    grid.sync();

    // All threads read the global sum
    float global_sum = global_sum_out[0];

    // Phase 3: Normalize output (grid-stride loop)
    // Each thread normalizes multiple elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

// Host function: 2-kernel fused softmax with cooperative launch
// ============================================================================
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate intermediate buffers
Fused2Softmax::Fused2Softmax(int n, int threadsPerBlock)
    : n(n), threadsPerBlock(threadsPerBlock) {
    // Query device
    int device;
    cudaCheckError(cudaGetDevice(&device));

    // Check if device supports cooperative launch
    int supportsCoopLaunch = 0;
    cudaCheckError(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device));

    if (!supportsCoopLaunch) {
        fprintf(stderr, "ERROR: Device does not support cooperative kernel launch\n");
        fprintf(stderr, "       This implementation requires CUDA compute capability 6.0+\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;
    cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

    // Kernel 1: Regular launch - use as many blocks as needed (no cooperative constraint!)
    numBlocks_K1 = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel 2: Cooperative launch - limited by hardware
    int maxBlocksPerSM_K2 = 0;
    cudaCheckError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM_K2,
        softmaxFused2_ReduceAndNormalize, threadsPerBlock, threadsPerBlock * sizeof(float)));
    numBlocks_K2 = maxBlocksPerSM_K2 * deviceProp.multiProcessorCount;

    if (numBlocks_K2 == 0) {
        fprintf(stderr, "ERROR: Cooperative launch not possible (maxBlocks = 0)\n");
        fprintf(stderr, "       Kernel may use too many resources\n");
        exit(EXIT_FAILURE);
    }

    // Allocate intermediate buffers (done once, outside timing loop)
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks_K1 * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks_K1 * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));
}

// Execute: Pure kernel execution (ONLY this is timed in benchmarks)
void Fused2Softmax::execute(const float *d_input, float *d_output) {
    // Initialize global stats (critical for correctness!)
    float init_max = -INFINITY;
    float init_sum = 0.0f;
    cudaCheckError(cudaMemcpy(d_global_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_global_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice));

    // Calculate shared memory size
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Kernel 1: Compute block statistics (REGULAR LAUNCH - full parallelism!)
    softmaxFused3_BlockStats<<<numBlocks_K1, threadsPerBlock, sharedMemSize>>>(
        d_input, d_block_maxes, d_block_sums, n);
    cudaCheckError(cudaGetLastError());

    // Kernel 2: Cooperative launch for fused reduce + normalize
    void* kernelArgs[] = {
        (void*)&d_input,
        (void*)&d_block_maxes,
        (void*)&d_block_sums,
        (void*)&d_global_max,
        (void*)&d_global_sum,
        (void*)&d_output,
        (void*)&n,
        (void*)&numBlocks_K1  // Number of block statistics to process
    };

    cudaCheckError(cudaLaunchCooperativeKernel(
        (void*)softmaxFused2_ReduceAndNormalize,
        dim3(numBlocks_K2), dim3(threadsPerBlock),
        kernelArgs, sharedMemSize, 0));
}

// Destructor: Free intermediate buffers
Fused2Softmax::~Fused2Softmax() {
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_Fused2(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    Fused2Softmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
