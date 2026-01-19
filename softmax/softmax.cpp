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
#include "softmax_naive.h"
#include "softmax_multipass.h"
#include "softmax_fused3.h"
#include "softmax_fused2.h"
#include "softmax_fused1.h"
#include "softmax_online.h"
#include "softmax_online_simple.h"
#include "softmax_online_warp.h"
#include "softmax_cub_block.h"
#include "softmax_cub_device.h"
#include "softmax_cudnn.h"
#include "softmax_tiny.h"
#include "softmax_small.h"
#include "vector_init.h"
#include "common/benchmark/benchmark_utils.h"

// ============================================================================
// BENCHMARK MODE: Data Structures and Constants
// ============================================================================

// Benchmark configuration
const char* BENCHMARK_METHODS[] = {
    "naive",
    "multi",
    "fused3",
    "fused2",
    "online_simple",
    "online_warp",
    "cub_block",
    "cub_device",
    "cudnn",
    "tiny",
    "small"
};
const int NUM_METHODS = 11;

const int BENCHMARK_SIZES[] = {16, 32, 64, 256, 512, 1<<10, 1<<13, 1<<16, 1<<18, 1<<20, 1<<23};  // 16, 32, 64, 256, 512, 1K, 8K, 64K, 256K, 1M, 8M
const int NUM_SIZES = sizeof(BENCHMARK_SIZES) / sizeof(BENCHMARK_SIZES[0]);
const char* SIZE_LABELS[] = {"16", "32", "64", "256", "512", "1K", "8K", "64K", "256K", "1M", "8M"};

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N              Set array size (default: 1048576)\n");
    printf("  -b, --block-size N        Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Softmax method: 'naive', 'multi', 'fused3', 'fused2', 'fused1', 'online', 'online_simple', 'online_warp', 'cub_block', 'cub_device', 'cudnn', 'tiny', or 'small' (default: online_warp)\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  naive:         Naive exp(x)/sum (unstable - demonstrates overflow)\n");
    printf("  multi:         Multi-pass stable (max -> exp-sum -> normalize)\n");
    printf("  fused3:        3-kernel fused (block stats -> global reduce -> normalize) [IMPLEMENTED]\n");
    printf("  fused2:        2-kernel fused (cooperative groups, grid sync) [IMPLEMENTED]\n");
    printf("  fused1:        1-kernel fused (single kernel, grid sync, cooperative groups) [SKELETON]\n");
    printf("  online:        Single-pass online algorithm (streaming max/sum) [SKELETON]\n");
    printf("  online_simple: 2-kernel online softmax (thread-level, educational) [IMPLEMENTED]\n");
    printf("  online_warp:   1-kernel online softmax (warp-level, cooperative, performance) [IMPLEMENTED]\n");
    printf("  cub_block:     3-kernel with CUB block-level primitives [IMPLEMENTED]\n");
    printf("  cub_device:    CUB device-level primitives (single-call reductions) [IMPLEMENTED]\n");
    printf("  cudnn:         NVIDIA cuDNN library (industry-standard) [IMPLEMENTED]\n");
    printf("  tiny:          Single-warp kernel (32 threads, warp shuffles only, optimal for <=1K) [IMPLEMENTED]\n");
    printf("  small:         Single-block kernel (256 threads, hybrid reduction, optimal for 1K-8K) [IMPLEMENTED]\n");
    printf("\nSpecial method:\n");
    printf("  all:           Run comprehensive benchmark across all methods and sizes\n");
    printf("                 Tests sizes: 16, 32, 64, 256, 512, 1K, 8K, 64K, 256K, 1M, 8M\n");
    printf("                 Iterations: 100 per test\n");
    printf("                 Output: Formatted performance table\n");
    printf("\nExample usage:\n");
    printf("  %s --method all                # Performance benchmark\n", program_name);
    printf("  %s --method all -b 512         # Custom block size\n", program_name);
}

// ============================================================================
// BENCHMARK MODE: Main Benchmark Function
// ============================================================================

// Main benchmark function: test all methods across all sizes
void benchmark_all_methods(int threadsPerBlock) {
    printf("\n=============================================================================\n");
    printf("                    RUNNING COMPREHENSIVE BENCHMARK\n");
    printf("=============================================================================\n");
    printf("Methods to test: %d\n", NUM_METHODS);
    printf("Sizes to test: %d (16, 32, 64, 256, 512, 1K, 8K, 64K, 256K, 1M, 8M)\n", NUM_SIZES);
    printf("Iterations per test: 100\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("=============================================================================\n\n");

    // Convert to vectors for the utility functions
    std::vector<std::string> method_names(BENCHMARK_METHODS, BENCHMARK_METHODS + NUM_METHODS);
    std::vector<std::string> size_labels(SIZE_LABELS, SIZE_LABELS + NUM_SIZES);

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
            int n = BENCHMARK_SIZES[s];
            printf("  Size: %s (%d elements)... ", SIZE_LABELS[s], n);
            fflush(stdout);

            // Allocate host memory
            float *h_input;
            allocateAndInitVector(&h_input, n);

            // Allocate device memory
            float *d_input, *d_output;
            cudaError_t err = cudaMalloc(&d_input, n * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                continue;
            }

            err = cudaMalloc(&d_output, n * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                cudaFree(d_input);
                continue;
            }

            cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

            // Try to instantiate kernel
            SoftmaxKernel *kernel = nullptr;
            try {
                if (strcmp(method, "naive") == 0) {
                    kernel = new NaiveSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "multi") == 0) {
                    // Multi is a legacy method without class-based API, skip it
                    kernel = nullptr;
                } else if (strcmp(method, "fused3") == 0) {
                    kernel = new Fused3Softmax(n, threadsPerBlock);
                } else if (strcmp(method, "fused2") == 0) {
                    kernel = new Fused2Softmax(n, threadsPerBlock);
                } else if (strcmp(method, "online_simple") == 0) {
                    kernel = new OnlineSimpleSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "online_warp") == 0) {
                    kernel = new OnlineWarpSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "cub_block") == 0) {
                    kernel = new CubBlockSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "cub_device") == 0) {
                    kernel = new CubDeviceSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "cudnn") == 0) {
                    kernel = new CudnnSoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "tiny") == 0) {
                    kernel = new TinySoftmax(n, threadsPerBlock);
                } else if (strcmp(method, "small") == 0) {
                    kernel = new SmallSoftmax(n, threadsPerBlock);
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
    print_benchmark_header("SOFTMAX PERFORMANCE BENCHMARK", size_labels, config.timed_iterations);
    for (int m = 0; m < NUM_METHODS; m++) {
        print_benchmark_row(BENCHMARK_METHODS[m], all_results[m]);
    }
    print_benchmark_footer(method_names, all_results);
    print_top_fastest(method_names, all_results, size_labels, 3, {"naive"});
}

// ============================================================================
// Single Method Benchmark
// ============================================================================

void softmax_op(int n, int threadsPerBlock, const char *method) {
    printf("Softmax of %d elements\n", n);
    printf("Method: %s\n", method);

    // Allocate and initialize input vector
    float *h_input;
    allocateAndInitVector(&h_input, n);

    // Allocate device vectors
    float *d_input, *d_output;
    allocateDeviceVector(&d_input, n);
    allocateDeviceVector(&d_output, n);
    transferToDevice(d_input, h_input, n);

    // Configure timing
    const int num_iterations = 100;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (!timings) {
        fprintf(stderr, "Failed to allocate memory for timings\n");
        freeHostVector(h_input);
        freeDeviceVector(d_input);
        freeDeviceVector(d_output);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running softmax %d times to collect statistics...\n", num_iterations);

    // Methods with class-based API for accurate profiling (allocate workspace ONCE)
    SoftmaxKernel *kernel = nullptr;

    if (strcmp(method, "cudnn") == 0) {
        kernel = new CudnnSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "cub_block") == 0) {
        kernel = new CubBlockSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "cub_device") == 0) {
        kernel = new CubDeviceSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "tiny") == 0) {
        kernel = new TinySoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "small") == 0) {
        kernel = new SmallSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "fused3") == 0) {
        kernel = new Fused3Softmax(n, threadsPerBlock);
    } else if (strcmp(method, "fused2") == 0) {
        kernel = new Fused2Softmax(n, threadsPerBlock);
    } else if (strcmp(method, "naive") == 0) {
        kernel = new NaiveSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "online_simple") == 0) {
        kernel = new OnlineSimpleSoftmax(n, threadsPerBlock);
    } else if (strcmp(method, "online_warp") == 0) {
        kernel = new OnlineWarpSoftmax(n, threadsPerBlock);
    }

    if (kernel) {
        // Class-based API: Time ONLY kernel execution (no setup/teardown overhead)
        for (int i = 0; i < num_iterations; i++) {
            cudaEventRecord(start);
            kernel->execute(d_input, d_output);  // Pure kernel execution!
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timings[i], start, stop);
        }
        delete kernel;  // Destructor automatically frees resources
    } else {
        // Legacy methods (include allocation overhead - not yet refactored)
        for (int i = 0; i < num_iterations; i++) {
            cudaEventRecord(start);

            if (strcmp(method, "multi") == 0) {
                softmax_MultiPass(d_input, d_output, n, threadsPerBlock);
            } else if (strcmp(method, "fused1") == 0) {
                softmax_Fused1(d_input, d_output, n, threadsPerBlock);
            } else if (strcmp(method, "online") == 0) {
                softmax_Online(d_input, d_output, n, threadsPerBlock);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timings[i], start, stop);
        }
    }
    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Cleanup
    freeDeviceVector(d_input);
    freeDeviceVector(d_output);
    freeHostVector(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nSoftmax completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    const char *method = "online_warp";  // Default method
    int n = 1 << 20;  // Default: 1 million elements
    int threadsPerBlock = 256;  // Default: 256 threads per block (optimal)

    static struct option long_options[] = {
        {"size",       required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"method",     required_argument, 0, 'm'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:b:m:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                if (n <= 0) {
                    fprintf(stderr, "Error: size must be positive\n");
                    return 1;
                }
                break;
            case 'b':
                threadsPerBlock = atoi(optarg);
                if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
                    fprintf(stderr, "Error: block-size must be between 1 and 1024\n");
                    return 1;
                }
                break;
            case 'm':
                method = optarg;
                if (strcmp(method, "naive") != 0 && strcmp(method, "multi") != 0 &&
                    strcmp(method, "fused3") != 0 && strcmp(method, "fused2") != 0 &&
                    strcmp(method, "fused1") != 0 && strcmp(method, "online") != 0 &&
                    strcmp(method, "online_simple") != 0 && strcmp(method, "online_warp") != 0 &&
                    strcmp(method, "cub_block") != 0 && strcmp(method, "cub_device") != 0 &&
                    strcmp(method, "cudnn") != 0 && strcmp(method, "tiny") != 0 &&
                    strcmp(method, "small") != 0 && strcmp(method, "all") != 0) {
                    fprintf(stderr, "Error: method must be 'naive', 'multi', 'fused3', 'fused2', 'fused1', 'online', 'online_simple', 'online_warp', 'cub_block', 'cub_device', 'cudnn', 'tiny', 'small', or 'all'\n");
                    return 1;
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

    // Run softmax operation
    if (strcmp(method, "all") == 0) {
        // Benchmark mode: test all methods across all sizes
        benchmark_all_methods(threadsPerBlock);
    } else {
        // Single method mode (original behavior)
        softmax_op(n, threadsPerBlock, method);
    }

    return 0;
}
