#ifndef BATCH_SOFTMAX_AUTOTUNER_H
#define BATCH_SOFTMAX_AUTOTUNER_H

#include "batch_softmax_kernel.h"
#include <vector>

// ============================================================================
// BATCH SOFTMAX AUTOTUNER
// ============================================================================
//
// This module provides runtime autotuning for batch softmax kernels.
// For kernels with tunable hyperparameters, the autotuner tries all
// supported configurations and returns the best one for a given shape.
//
// TUNABLE KERNELS:
// ----------------
// 1. naive:      threadsPerBlock (64, 128, 256, 512)
// 2. multi_warp: num_warps (4, 8, 16)
// 3. cub:        threadsPerBlock (128, 256, 512)
//
// AUTOTUNING PROCESS:
// -------------------
// For each config:
// 1. Create kernel with that config
// 2. Run warmup iterations to stabilize GPU state
// 3. Run timed iterations and measure elapsed time
// 4. Track the fastest config
//
// Returns the best config value (e.g., best threadsPerBlock or num_warps)
//
// ============================================================================

// Autotune result containing best config and timing info
struct AutotuneResult {
    int best_config;      // The config value that performed best
    float best_time_ms;   // Time in milliseconds for best config

    AutotuneResult() : best_config(0), best_time_ms(0.0f) {}
    AutotuneResult(int config, float time) : best_config(config), best_time_ms(time) {}
};

// Autotune the naive batch softmax kernel
// Returns the best threadsPerBlock value (64, 128, 256, or 512)
AutotuneResult autotune_naive(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters = 3,
    int timed_iters = 10
);

// Autotune the multi_warp batch softmax kernel
// Returns the best num_warps value (4, 8, or 16)
AutotuneResult autotune_multi_warp(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters = 3,
    int timed_iters = 10
);

// Autotune the CUB batch softmax kernel
// Returns the best threadsPerBlock value (128, 256, or 512)
AutotuneResult autotune_cub(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters = 3,
    int timed_iters = 10
);

// Autotune the cuDNN batch softmax kernel
// Returns the best algorithm: 0 = FAST, 1 = ACCURATE
AutotuneResult autotune_cudnn(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters = 3,
    int timed_iters = 10
);

// Autotune the online multi-warp batch softmax kernel
// Returns the best num_warps value (4, 8, or 16)
AutotuneResult autotune_online_multi_warp(
    int batch_size,
    int dim,
    const float* d_input,
    float* d_output,
    int warmup_iters = 3,
    int timed_iters = 10
);

#endif  // BATCH_SOFTMAX_AUTOTUNER_H
