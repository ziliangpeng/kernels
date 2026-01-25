#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "batch_softmax_naive.h"
#include "batch_softmax_warp.h"
#include "batch_softmax_online_warp.h"
#include "batch_softmax_multi_warp.h"
#include "batch_softmax_cub.h"
#include "batch_softmax_cudnn.h"
#include "batch_softmax_hybrid.h"
#include "batch_softmax_online_multi_warp.h"
#include "batch_softmax_autotuner.h"
#include "vector_init.h"
#include "common/benchmark/benchmark_utils.h"

// ============================================================================
// BENCHMARK MODE: Data Structures and Constants
// ============================================================================

// Performance Summary (tested on NVIDIA GPU):
// ============================================================================
// | Size      | Fastest Kernel      | Time (ms) | Notes                      |
// |-----------|---------------------|-----------|----------------------------|
// | 64x64     | tie (all ~0.006)    | 0.006     | All kernels similar        |
// | 64x256    | cudnn               | 0.005     | cuDNN optimal for small    |
// | 64x1K     | multi_warp/hybrid   | 0.006     | Vectorization helps        |
// | 256x256   | tie (most ~0.006)   | 0.006     | All kernels similar        |
// | 256x1K    | multi_warp/hybrid   | 0.006     | Vectorization helps        |
// | 1Kx256    | cudnn               | 0.006     | cuDNN optimal              |
// | 1Kx1K     | multi_warp/hybrid   | 0.008     | Vectorization helps        |
// | 4Kx256    | cudnn               | 0.006     | cuDNN optimal for batches  |
// | 4Kx1K     | multi_warp/hybrid   | 0.014     | 256 threads + float4       |
// | 8Kx512    | cudnn               | 0.011     | cuDNN optimal              |
// | 256x4K    | multi_warp/hybrid   | 0.008     | Vectorization shines       |
// | 256x8K    | multi_warp/hybrid   | 0.010     | Vectorization shines       |
// | 1Kx4K     | multi_warp/hybrid   | 0.013     | Vectorization shines       |
// | 512x16K   | multi_warp/hybrid   | 0.040     | 7x faster than warp        |
// | 128x32K   | multi_warp/hybrid   | 0.015     | 6x faster than warp        |
// ============================================================================
//
// Key Findings:
// 1. multi_warp/hybrid: Best for large dims (4K+) due to float4 vectorization
// 2. cudnn: Best for high batch counts with small-medium dims
// 3. warp/online_warp: Struggle with large dims (only 32 threads per row)
// 4. hybrid delegates to multi_warp for dim > 64, making it a good default
//
// Recommendation: Use "hybrid" as default - it adapts to input size automatically

// Benchmark configuration
const char* BENCHMARK_METHODS[] = {
    "naive",
    "naive_tuned",
    "warp",
    "online_warp",
    "multi_warp",
    "multi_warp_tuned",
    "online_multi_warp",
    "online_multi_warp_tuned",
    "cub",
    "cub_tuned",
    "cudnn",
    "cudnn_tuned",
    "hybrid"
};
const int NUM_METHODS = 13;

// Benchmark sizes: (batch_size, dim) pairs
struct BenchmarkSize {
    int batch;
    int dim;
    const char* label;
};

const BenchmarkSize BENCHMARK_SIZES[] = {
    // Standard sizes
    {64, 64, "64x64"},
    {64, 256, "64x256"},
    {64, 1024, "64x1K"},
    {256, 256, "256x256"},
    {256, 1024, "256x1K"},
    {1024, 256, "1Kx256"},
    {1024, 1024, "1Kx1K"},
    {4096, 256, "4Kx256"},
    {4096, 1024, "4Kx1K"},
    {8192, 512, "8Kx512"},
    // Large dims - test vectorization benefits
    {256, 4096, "256x4K"},
    {256, 8192, "256x8K"},
    {1024, 4096, "1Kx4K"},
    {512, 16384, "512x16K"},
    {128, 32768, "128x32K"},
};
const int NUM_SIZES = sizeof(BENCHMARK_SIZES) / sizeof(BENCHMARK_SIZES[0]);

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -b, --batch N             Set batch size (default: 64)\n");
    printf("  -d, --dim N               Set dimension per row (default: 1024)\n");
    printf("  -t, --threads N           Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Method name or 'all' (default: hybrid)\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  naive:           One block per row, shared memory reductions\n");
    printf("  naive_tuned:     Naive with autotuned threadsPerBlock (64-512)\n");
    printf("  warp:            One warp per row, warp shuffle reductions\n");
    printf("  online_warp:     Single-pass online algorithm with warp shuffles (32 threads)\n");
    printf("  multi_warp:      Multiple warps per row with vectorized loads (float4)\n");
    printf("  multi_warp_tuned: Multi-warp with autotuned warps (4, 8, or 16)\n");
    printf("  online_multi_warp: Online softmax + multi-warp + vectorization\n");
    printf("  online_multi_warp_tuned: Online multi-warp with autotuned warps\n");
    printf("  cub:             NVIDIA CUB BlockReduce-based implementation\n");
    printf("  cub_tuned:       CUB with autotuned threadsPerBlock (128-512)\n");
    printf("  cudnn:           cuDNN library implementation (ACCURATE algorithm)\n");
    printf("  cudnn_tuned:     cuDNN with autotuned algorithm (FAST or ACCURATE)\n");
    printf("  hybrid:          Adaptive kernel selection based on dimension size\n");
    printf("\nSpecial method:\n");
    printf("  all:         Run comprehensive benchmark across all methods and sizes\n");
    printf("\nExample usage:\n");
    printf("  %s --method all                         # Performance benchmark\n", program_name);
    printf("  %s --batch 256 --dim 4096 --method multi_warp  # Single test\n", program_name);
}

// ============================================================================
// CPU Reference Implementation for Verification
// ============================================================================

void cpu_batch_softmax(const float *input, float *output, int batch_size, int dim) {
    for (int row = 0; row < batch_size; row++) {
        const float *row_in = input + row * dim;
        float *row_out = output + row * dim;

        // Find max
        float max_val = row_in[0];
        for (int i = 1; i < dim; i++) {
            max_val = fmaxf(max_val, row_in[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            row_out[i] = expf(row_in[i] - max_val);
            sum += row_out[i];
        }

        // Normalize
        for (int i = 0; i < dim; i++) {
            row_out[i] /= sum;
        }
    }
}

// ============================================================================
// Verification Helper
// ============================================================================

bool verify_output(const float *gpu_output, const float *cpu_output, int batch_size, int dim, float tolerance = 1e-5f) {
    int total = batch_size * dim;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int i = 0; i < total; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    if (max_diff > tolerance) {
        int row = max_diff_idx / dim;
        int col = max_diff_idx % dim;
        printf("  Verification FAILED: max diff %.6f at [%d, %d] (GPU: %.6f, CPU: %.6f)\n",
               max_diff, row, col, gpu_output[max_diff_idx], cpu_output[max_diff_idx]);
        return false;
    }
    return true;
}

// ============================================================================
// BENCHMARK MODE: Main Benchmark Function
// ============================================================================

void benchmark_all_methods(int threadsPerBlock) {
    printf("\n=============================================================================\n");
    printf("                    RUNNING COMPREHENSIVE BENCHMARK\n");
    printf("=============================================================================\n");
    printf("Methods to test: %d\n", NUM_METHODS);
    printf("Sizes to test: %d\n", NUM_SIZES);
    printf("Iterations per test: 100\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("=============================================================================\n\n");

    // Convert to vectors for the utility functions
    std::vector<std::string> method_names(BENCHMARK_METHODS, BENCHMARK_METHODS + NUM_METHODS);
    std::vector<std::string> size_labels;
    for (int i = 0; i < NUM_SIZES; i++) {
        size_labels.push_back(BENCHMARK_SIZES[i].label);
    }

    // Allocate result arrays
    std::vector<std::vector<BenchmarkResult>> all_results(NUM_METHODS, std::vector<BenchmarkResult>(NUM_SIZES));

    BenchmarkConfig config;
    config.warmup_iterations = 10;
    config.timed_iterations = 100;

    // Run benchmarks for each method and size
    for (int m = 0; m < NUM_METHODS; m++) {
        const char *method = BENCHMARK_METHODS[m];
        printf("Testing method: %s\n", method);

        for (int s = 0; s < NUM_SIZES; s++) {
            int batch_size = BENCHMARK_SIZES[s].batch;
            int dim = BENCHMARK_SIZES[s].dim;
            int total = batch_size * dim;
            printf("  Size: %s (%d x %d = %d elements)... ", BENCHMARK_SIZES[s].label, batch_size, dim, total);
            fflush(stdout);

            // Allocate host memory
            float *h_input;
            allocateAndInitVector(&h_input, total);

            // Allocate device memory
            float *d_input, *d_output;
            cudaError_t err = cudaMalloc(&d_input, total * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                continue;
            }

            err = cudaMalloc(&d_output, total * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                cudaFree(d_input);
                continue;
            }

            cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

            // Try to instantiate kernel
            BatchSoftmaxKernel *kernel = nullptr;
            try {
                if (strcmp(method, "naive") == 0) {
                    kernel = new NaiveBatchSoftmax(batch_size, dim, threadsPerBlock);
                } else if (strcmp(method, "naive_tuned") == 0) {
                    AutotuneResult result = autotune_naive(batch_size, dim, d_input, d_output);
                    kernel = new NaiveBatchSoftmax(batch_size, dim, result.best_config);
                } else if (strcmp(method, "warp") == 0) {
                    kernel = new WarpBatchSoftmax(batch_size, dim, threadsPerBlock);
                } else if (strcmp(method, "online_warp") == 0) {
                    kernel = new OnlineWarpBatchSoftmax(batch_size, dim, threadsPerBlock);
                } else if (strcmp(method, "multi_warp") == 0) {
                    // Default to 8 warps (256 threads)
                    kernel = new MultiWarpBatchSoftmax(batch_size, dim, 8);
                } else if (strcmp(method, "multi_warp_tuned") == 0) {
                    AutotuneResult result = autotune_multi_warp(batch_size, dim, d_input, d_output);
                    kernel = new MultiWarpBatchSoftmax(batch_size, dim, result.best_config);
                } else if (strcmp(method, "online_multi_warp") == 0) {
                    // Default to 8 warps (256 threads)
                    kernel = new OnlineMultiWarpBatchSoftmax(batch_size, dim, 8);
                } else if (strcmp(method, "online_multi_warp_tuned") == 0) {
                    AutotuneResult result = autotune_online_multi_warp(batch_size, dim, d_input, d_output);
                    kernel = new OnlineMultiWarpBatchSoftmax(batch_size, dim, result.best_config);
                } else if (strcmp(method, "cub") == 0) {
                    kernel = new CubBatchSoftmax(batch_size, dim, threadsPerBlock);
                } else if (strcmp(method, "cub_tuned") == 0) {
                    AutotuneResult result = autotune_cub(batch_size, dim, d_input, d_output);
                    kernel = new CubBatchSoftmax(batch_size, dim, result.best_config);
                } else if (strcmp(method, "cudnn") == 0) {
                    // Default to ACCURATE algorithm (1)
                    kernel = new CudnnBatchSoftmax(batch_size, dim, 1);
                } else if (strcmp(method, "cudnn_tuned") == 0) {
                    AutotuneResult result = autotune_cudnn(batch_size, dim, d_input, d_output);
                    kernel = new CudnnBatchSoftmax(batch_size, dim, result.best_config);
                } else if (strcmp(method, "hybrid") == 0) {
                    kernel = new HybridBatchSoftmax(batch_size, dim, threadsPerBlock);
                }

                if (!kernel) {
                    printf("SKIPPED (unknown method)\n");
                    all_results[m][s] = BenchmarkResult("Unknown method");
                } else {
                    // Run performance benchmark with per-iteration timing
                    auto kernel_fn = [&]() { kernel->execute(d_input, d_output); };
                    TimingStats stats = get_timing_stats(kernel_fn, config);

                    all_results[m][s] = BenchmarkResult(stats.p50_ms, stats.p90_ms);
                    printf("%.3f/%.3f ms\n", stats.p50_ms, stats.p90_ms);

                    delete kernel;
                }
            } catch (const std::exception &e) {
                printf("SKIPPED (exception: %s)\n", e.what());
                all_results[m][s] = BenchmarkResult(std::string("Exception: ") + e.what());
                if (kernel) delete kernel;
            } catch (...) {
                printf("SKIPPED (unknown exception)\n");
                all_results[m][s] = BenchmarkResult("Unknown exception");
                if (kernel) delete kernel;
            }

            // Cleanup
            freeHostVector(h_input);
            cudaFree(d_input);
            cudaFree(d_output);
        }
        printf("\n");
    }

    // Print results using utility functions
    print_benchmark_header("BATCH SOFTMAX PERFORMANCE BENCHMARK", size_labels, config.timed_iterations);
    for (int m = 0; m < NUM_METHODS; m++) {
        print_benchmark_row(BENCHMARK_METHODS[m], all_results[m]);
    }
    print_benchmark_footer(method_names, all_results);
    print_top_fastest(method_names, all_results, size_labels, 3, {});
}

// ============================================================================
// Single Method Benchmark
// ============================================================================

void batch_softmax_op(int batch_size, int dim, int threadsPerBlock, const char *method) {
    int total = batch_size * dim;
    printf("Batch Softmax: %d rows x %d columns = %d elements\n", batch_size, dim, total);
    printf("Method: %s\n", method);
    printf("Threads per block: %d\n", threadsPerBlock);

    // Allocate and initialize input
    float *h_input;
    allocateAndInitVector(&h_input, total);

    // Allocate host output for verification
    float *h_output = (float*)malloc(total * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        freeHostVector(h_input);
        return;
    }
    float *h_output_cpu = (float*)malloc(total * sizeof(float));
    if (!h_output_cpu) {
        fprintf(stderr, "Failed to allocate host memory for CPU output\n");
        freeHostVector(h_input);
        free(h_output);
        return;
    }

    // Compute CPU reference
    cpu_batch_softmax(h_input, h_output_cpu, batch_size, dim);

    // Allocate device vectors
    float *d_input, *d_output;
    allocateDeviceVector(&d_input, total);
    allocateDeviceVector(&d_output, total);
    transferToDevice(d_input, h_input, total);

    // Configure timing
    const int num_iterations = 100;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (!timings) {
        fprintf(stderr, "Failed to allocate memory for timings\n");
        freeHostVector(h_input);
        free(h_output);
        free(h_output_cpu);
        freeDeviceVector(d_input);
        freeDeviceVector(d_output);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running batch softmax %d times to collect statistics...\n", num_iterations);

    // Instantiate kernel
    BatchSoftmaxKernel *kernel = nullptr;
    if (strcmp(method, "naive") == 0) {
        kernel = new NaiveBatchSoftmax(batch_size, dim, threadsPerBlock);
    } else if (strcmp(method, "naive_tuned") == 0) {
        printf("Autotuning naive kernel...\n");
        AutotuneResult result = autotune_naive(batch_size, dim, d_input, d_output);
        printf("Best config: %d threads (%.3f ms)\n", result.best_config, result.best_time_ms);
        kernel = new NaiveBatchSoftmax(batch_size, dim, result.best_config);
    } else if (strcmp(method, "warp") == 0) {
        kernel = new WarpBatchSoftmax(batch_size, dim, threadsPerBlock);
    } else if (strcmp(method, "online_warp") == 0) {
        kernel = new OnlineWarpBatchSoftmax(batch_size, dim, threadsPerBlock);
    } else if (strcmp(method, "multi_warp") == 0) {
        // Default to 8 warps (256 threads)
        kernel = new MultiWarpBatchSoftmax(batch_size, dim, 8);
    } else if (strcmp(method, "multi_warp_tuned") == 0) {
        printf("Autotuning multi_warp kernel...\n");
        AutotuneResult result = autotune_multi_warp(batch_size, dim, d_input, d_output);
        printf("Best config: %d warps (%.3f ms)\n", result.best_config, result.best_time_ms);
        kernel = new MultiWarpBatchSoftmax(batch_size, dim, result.best_config);
    } else if (strcmp(method, "online_multi_warp") == 0) {
        // Default to 8 warps (256 threads)
        kernel = new OnlineMultiWarpBatchSoftmax(batch_size, dim, 8);
    } else if (strcmp(method, "online_multi_warp_tuned") == 0) {
        printf("Autotuning online_multi_warp kernel...\n");
        AutotuneResult result = autotune_online_multi_warp(batch_size, dim, d_input, d_output);
        printf("Best config: %d warps (%.3f ms)\n", result.best_config, result.best_time_ms);
        kernel = new OnlineMultiWarpBatchSoftmax(batch_size, dim, result.best_config);
    } else if (strcmp(method, "cub") == 0) {
        kernel = new CubBatchSoftmax(batch_size, dim, threadsPerBlock);
    } else if (strcmp(method, "cub_tuned") == 0) {
        printf("Autotuning cub kernel...\n");
        AutotuneResult result = autotune_cub(batch_size, dim, d_input, d_output);
        printf("Best config: %d threads (%.3f ms)\n", result.best_config, result.best_time_ms);
        kernel = new CubBatchSoftmax(batch_size, dim, result.best_config);
    } else if (strcmp(method, "cudnn") == 0) {
        // Default to ACCURATE algorithm (1)
        kernel = new CudnnBatchSoftmax(batch_size, dim, 1);
    } else if (strcmp(method, "cudnn_tuned") == 0) {
        printf("Autotuning cudnn kernel...\n");
        AutotuneResult result = autotune_cudnn(batch_size, dim, d_input, d_output);
        const char* algo_name = (result.best_config == 0) ? "FAST" : "ACCURATE";
        printf("Best config: %s (%.3f ms)\n", algo_name, result.best_time_ms);
        kernel = new CudnnBatchSoftmax(batch_size, dim, result.best_config);
    } else if (strcmp(method, "hybrid") == 0) {
        kernel = new HybridBatchSoftmax(batch_size, dim, threadsPerBlock);
    }

    if (kernel) {
        // Time kernel execution
        for (int i = 0; i < num_iterations; i++) {
            cudaEventRecord(start);
            kernel->execute(d_input, d_output);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timings[i], start, stop);
        }
        delete kernel;
    } else {
        fprintf(stderr, "Unknown method: %s\n", method);
        freeHostVector(h_input);
        free(h_output);
        free(h_output_cpu);
        freeDeviceVector(d_input);
        freeDeviceVector(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(timings);
        return;
    }

    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Verify correctness
    transferFromDevice(h_output, d_output, total);
    printf("\nVerifying correctness against CPU reference...\n");
    if (verify_output(h_output, h_output_cpu, batch_size, dim)) {
        printf("  Verification PASSED\n");
    }

    // Cleanup
    freeDeviceVector(d_input);
    freeDeviceVector(d_output);
    freeHostVector(h_input);
    free(h_output);
    free(h_output_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nBatch Softmax completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    const char *method = "hybrid";  // Default method (adaptive selection)
    int batch_size = 64;
    int dim = 1024;
    int threadsPerBlock = 256;

    static struct option long_options[] = {
        {"batch",   required_argument, 0, 'b'},
        {"dim",     required_argument, 0, 'd'},
        {"threads", required_argument, 0, 't'},
        {"method",  required_argument, 0, 'm'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "b:d:t:m:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'b':
                batch_size = atoi(optarg);
                if (batch_size <= 0) {
                    fprintf(stderr, "Error: batch size must be positive\n");
                    return 1;
                }
                break;
            case 'd':
                dim = atoi(optarg);
                if (dim <= 0) {
                    fprintf(stderr, "Error: dimension must be positive\n");
                    return 1;
                }
                break;
            case 't':
                threadsPerBlock = atoi(optarg);
                if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
                    fprintf(stderr, "Error: threads must be between 1 and 1024\n");
                    return 1;
                }
                break;
            case 'm':
                method = optarg;
                {
                    // Validate method against BENCHMARK_METHODS array
                    bool is_valid = (strcmp(method, "all") == 0);
                    if (!is_valid) {
                        for (int i = 0; i < NUM_METHODS; ++i) {
                            if (strcmp(method, BENCHMARK_METHODS[i]) == 0) {
                                is_valid = true;
                                break;
                            }
                        }
                    }
                    if (!is_valid) {
                        fprintf(stderr, "Error: unknown method '%s'. Use --help for available methods.\n", method);
                        return 1;
                    }
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Initialize random seed
    srand(time(NULL));

    // Run batch softmax operation
    if (strcmp(method, "all") == 0) {
        // Benchmark mode: test all methods across all sizes
        benchmark_all_methods(threadsPerBlock);
    } else {
        // Single method mode
        batch_softmax_op(batch_size, dim, threadsPerBlock, method);
    }

    return 0;
}
