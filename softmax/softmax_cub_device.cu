#include "softmax_cub_device.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CUB DEVICE-LEVEL SOFTMAX: Single-Call Approach
// ============================================================================
//
// WHAT ARE CUB DEVICE-LEVEL PRIMITIVES?
// --------------------------------------
// CUB provides device-level functions that handle EVERYTHING for you:
//
// cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items)
//   - Automatically determines optimal grid/block configuration
//   - Allocates temporary storage (you provide buffer)
//   - Launches reduction kernel(s) internally
//   - Returns single result
//
// Key features:
//   ✓ Single function call (vs writing your own kernel)
//   ✓ Handles all edge cases (odd sizes, small inputs, etc.)
//   ✓ Optimized for all GPU architectures
//   ✗ Less control over memory access patterns
//   ✗ Multiple kernel launches internally (may be slower)
//
// IMPLEMENTATION STRATEGY
// -----------------------
// We use a 2-kernel approach:
//
// Kernel 1: cub::DeviceReduce::Max
//   - Find global maximum in single call
//   - CUB handles kernel launch, grid config, everything
//
// Kernel 2: Custom fused kernel
//   - Compute exp(x - max) / sum(exp(x - max)) in single pass
//   - Each thread: accumulate local sum, then normalize
//   - Use atomicAdd for global sum accumulation
//
// WHY NOT USE CUB FOR SUM TOO?
// -----------------------------
// We COULD do this:
//   1. cub::DeviceReduce::Max to find max
//   2. Custom kernel to compute exp(x - max) and store in temp array
//   3. cub::DeviceReduce::Sum to find sum of exps
//   4. Custom kernel to normalize
//
// But this would be SLOWER because:
//   - 4 kernels vs 2 kernels
//   - 3 full passes over data (max, exp, sum, normalize)
//   - Temp array allocation (extra memory bandwidth)
//
// Our approach fuses exp-sum-normalize into one kernel, saving 2 passes.
//
// EXPECTED PERFORMANCE VS BLOCK-LEVEL CUB
// ----------------------------------------
// Block-level CUB (3 kernels):
//   ✓ Each block computes local stats once (good cache locality)
//   ✓ Single pass over input data per kernel
//   ✓ Typical: 0.058ms for 100K elements
//
// Device-level CUB (2 kernels):
//   ✗ DeviceReduce::Max may launch multiple kernels internally
//   ✗ Less control over memory access patterns
//   ? Expected: Similar or slightly slower
//
// The main advantage is CODE SIMPLICITY, not performance.
//
// ============================================================================

// Kernel 2: Compute sum(exp(x - global_max)) using atomic add
// Each block computes local sum and atomically adds to global sum
__global__ void softmaxCubDevice_ExpSum(
    const float *input,
    const float *d_global_max,
    float *global_sum,
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Read global max from device memory once per thread
    float global_max = *d_global_max;

    // Phase 1: Compute block-level sum(exp(x - global_max))
    float thread_sum = 0.0f;
    if (idx < n) {
        thread_sum = expf(input[idx] - global_max);
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

    // Thread 0 atomically adds block sum to global sum
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// Kernel 3: Normalize output using device pointers with shared memory optimization
__global__ void softmaxCubDevice_Normalize(
    const float *input,
    const float *d_global_max,
    const float *d_global_sum,
    float *output,
    int n
) {
    // Use shared memory to cache global values (read once per block instead of per thread)
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    if (threadIdx.x == 0) {
        s_max_val = *d_global_max;
        s_sum_val = *d_global_sum;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx] - s_max_val) / s_sum_val;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate workspace (temp storage, output buffers)
CubDeviceSoftmax::CubDeviceSoftmax(int n, int threadsPerBlock)
    : n(n), threadsPerBlock(threadsPerBlock) {
    // Calculate grid dimensions
    numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate output buffers
    cudaCheckError(cudaMalloc(&d_max_out, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));

    // Determine temporary device storage requirements for CUB DeviceReduce
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    // First call: determine required temp storage size
    // Note: We use a dummy input pointer here (CUB doesn't dereference it in the sizing call)
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, (const float*)nullptr, d_max_out, n);

    // Allocate temporary storage (done once, outside timing loop)
    cudaCheckError(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

// Execute: Pure kernel execution (ONLY this is timed in benchmarks)
void CubDeviceSoftmax::execute(const float *d_input, float *d_output) {
    // ========================================================================
    // KERNEL 1: Use CUB DeviceReduce::Max to find global maximum
    // ========================================================================

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_max_out, n);
    cudaCheckError(cudaGetLastError());

    // ========================================================================
    // KERNEL 2: Compute sum(exp(x - global_max))
    // ========================================================================

    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Initialize global sum to 0 using cudaMemset (more efficient than cudaMemcpy)
    cudaCheckError(cudaMemset(d_global_sum, 0, sizeof(float)));

    // Launch kernel to compute exp-sum (reads d_max_out directly from device memory)
    softmaxCubDevice_ExpSum<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_max_out, d_global_sum, n);
    cudaCheckError(cudaGetLastError());

    // ========================================================================
    // KERNEL 3: Normalize output
    // ========================================================================
    // Note: Implicit synchronization happens between kernel launches
    // No need for explicit cudaDeviceSynchronize()

    softmaxCubDevice_Normalize<<<numBlocks, threadsPerBlock>>>(
        d_input, d_max_out, d_global_sum, d_output, n);
    cudaCheckError(cudaGetLastError());
}

// Destructor: Free workspace
CubDeviceSoftmax::~CubDeviceSoftmax() {
    cudaFree(d_temp_storage);
    cudaFree(d_max_out);
    cudaFree(d_global_sum);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_CubDevice(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    CubDeviceSoftmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
