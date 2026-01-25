#include "batch_softmax_autotuner.h"
#include "batch_softmax_naive.h"
#include "batch_softmax_multi_warp.h"
#include "batch_softmax_cub.h"
#include "batch_softmax_cudnn.h"
#include "batch_softmax_online_multi_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <float.h>

// ============================================================================
// BATCH SOFTMAX AUTOTUNER IMPLEMENTATION
// ============================================================================
//
// General autotuning strategy:
// 1. Try each config value in the provided list
// 2. For each config, run warmup iterations to get GPU into steady state
// 3. Time multiple iterations and compute average
// 4. Track the fastest config
// 5. Return the best config and its timing
//
// ============================================================================

// Generic autotuning helper function
// Takes a vector of configs and a factory function that creates kernels
template<typename KernelFactory>
AutotuneResult autotune_generic(
    const std::vector<int>& configs,
    KernelFactory factory,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    AutotuneResult result;
    result.best_time_ms = FLT_MAX;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int config : configs) {
        BatchSoftmaxKernel* kernel = nullptr;
        try {
            kernel = factory(config);
        } catch (...) {
            // Skip configs that fail to instantiate
            continue;
        }

        if (!kernel) continue;

        // Warmup runs
        for (int i = 0; i < warmup_iters; i++) {
            kernel->execute(d_input, d_output);
        }
        cudaDeviceSynchronize();

        // Timed runs
        cudaEventRecord(start);
        for (int i = 0; i < timed_iters; i++) {
            kernel->execute(d_input, d_output);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start, stop);
        float avg_ms = total_ms / timed_iters;

        if (avg_ms < result.best_time_ms) {
            result.best_time_ms = avg_ms;
            result.best_config = config;
        }

        delete kernel;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

// Autotune the naive batch softmax kernel
AutotuneResult autotune_naive(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    std::vector<int> configs = {64, 128, 256, 512};

    auto factory = [batch_size, dim](int threadsPerBlock) -> BatchSoftmaxKernel* {
        return new NaiveBatchSoftmax(batch_size, dim, threadsPerBlock);
    };

    return autotune_generic(configs, factory, d_input, d_output, warmup_iters, timed_iters);
}

// Autotune the multi_warp batch softmax kernel
AutotuneResult autotune_multi_warp(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    std::vector<int> configs = {4, 8, 16};

    auto factory = [batch_size, dim](int num_warps) -> BatchSoftmaxKernel* {
        return new MultiWarpBatchSoftmax(batch_size, dim, num_warps);
    };

    return autotune_generic(configs, factory, d_input, d_output, warmup_iters, timed_iters);
}

// Autotune the CUB batch softmax kernel
AutotuneResult autotune_cub(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    std::vector<int> configs = {128, 256, 512};

    auto factory = [batch_size, dim](int threadsPerBlock) -> BatchSoftmaxKernel* {
        return new CubBatchSoftmax(batch_size, dim, threadsPerBlock);
    };

    return autotune_generic(configs, factory, d_input, d_output, warmup_iters, timed_iters);
}

// Autotune the cuDNN batch softmax kernel
AutotuneResult autotune_cudnn(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    // 0 = CUDNN_SOFTMAX_FAST, 1 = CUDNN_SOFTMAX_ACCURATE
    std::vector<int> configs = {0, 1};

    auto factory = [batch_size, dim](int algorithm) -> BatchSoftmaxKernel* {
        return new CudnnBatchSoftmax(batch_size, dim, algorithm);
    };

    return autotune_generic(configs, factory, d_input, d_output, warmup_iters, timed_iters);
}

// Autotune the online multi-warp batch softmax kernel
AutotuneResult autotune_online_multi_warp(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters,
    int timed_iters
) {
    std::vector<int> configs = {4, 8, 16};

    auto factory = [batch_size, dim](int num_warps) -> BatchSoftmaxKernel* {
        return new OnlineMultiWarpBatchSoftmax(batch_size, dim, num_warps);
    };

    return autotune_generic(configs, factory, d_input, d_output, warmup_iters, timed_iters);
}
