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

// Result structures
struct TimingStats {
    float p50_ms;  // Median (50th percentile)
    float p90_ms;  // 90th percentile
};

struct BenchmarkResult {
    float p50_time_ms;       // Median (50th percentile), -1.0 if skipped/failed
    float p90_time_ms;       // 90th percentile, -1.0 if skipped/failed
    bool skipped;
    std::string skip_reason;
};

struct VerificationResult {
    bool passed;
    double max_rel_error;    // -1.0 if skipped
    double sum_error;        // -1.0 if skipped
    bool has_nan_inf;
    bool skipped;
};

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N              Set array size (default: 1048576)\n");
    printf("  -b, --block-size N        Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Softmax method: 'naive', 'multi', 'fused3', 'fused2', 'fused1', 'online', 'online_simple', 'online_warp', 'cub_block', 'cub_device', 'cudnn', 'tiny', or 'small' (default: online_warp)\n");
    printf("  -v, --verify              Enable result verification\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  naive:         Naive exp(x)/sum (unstable - demonstrates overflow)\n");
    printf("  multi:         Multi-pass stable (max → exp-sum → normalize)\n");
    printf("  fused3:        3-kernel fused (block stats → global reduce → normalize) [IMPLEMENTED]\n");
    printf("  fused2:        2-kernel fused (cooperative groups, grid sync) [IMPLEMENTED]\n");
    printf("  fused1:        1-kernel fused (single kernel, grid sync, cooperative groups) [SKELETON]\n");
    printf("  online:        Single-pass online algorithm (streaming max/sum) [SKELETON]\n");
    printf("  online_simple: 2-kernel online softmax (thread-level, educational) [IMPLEMENTED]\n");
    printf("  online_warp:   1-kernel online softmax (warp-level, cooperative, performance) [IMPLEMENTED]\n");
    printf("  cub_block:     3-kernel with CUB block-level primitives [IMPLEMENTED]\n");
    printf("  cub_device:    CUB device-level primitives (single-call reductions) [IMPLEMENTED]\n");
    printf("  cudnn:         NVIDIA cuDNN library (industry-standard) [IMPLEMENTED]\n");
    printf("  tiny:          Single-warp kernel (32 threads, warp shuffles only, optimal for ≤1K) [IMPLEMENTED]\n");
    printf("  small:         Single-block kernel (256 threads, hybrid reduction, optimal for 1K-8K) [IMPLEMENTED]\n");
    printf("\nSpecial method:\n");
    printf("  all:           Run comprehensive benchmark across all methods and sizes\n");
    printf("                 Tests sizes: 16, 32, 64, 256, 512, 1K, 8K, 64K, 256K, 1M, 8M\n");
    printf("                 Iterations: 100 per test\n");
    printf("                 Output: Formatted performance table\n");
    printf("\n                 When combined with --verify:\n");
    printf("                   - Validates correctness against CPU reference\n");
    printf("                   - Prints accuracy comparison table\n");
    printf("\nExample usage:\n");
    printf("  %s --method all                # Performance benchmark only\n", program_name);
    printf("  %s --method all --verify       # Performance + verification\n", program_name);
    printf("  %s --method all -b 512         # Custom block size\n", program_name);
}

// CPU reference implementation for verification (numerically stable)
void softmax_cpu_reference(const float *input, float *output, int n) {
    // Find max
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute exp-sum (use double for better accuracy)
    double sum_exp = 0.0;
    for (int i = 0; i < n; i++) {
        sum_exp += exp(input[i] - max_val);
    }

    // Normalize
    for (int i = 0; i < n; i++) {
        output[i] = exp(input[i] - max_val) / sum_exp;
    }
}

// ============================================================================
// BENCHMARK MODE: Helper Functions
// ============================================================================

// Helper: Get per-iteration timing statistics using CUDA events
// Uses per-iteration event pairs without CPU sync between iterations,
// then extracts P50 (median) and P90 percentiles for accurate kernel timing
TimingStats get_timing_stats(SoftmaxKernel *kernel, const float *d_input, float *d_output,
                             int n, int num_iterations) {
    // Allocate event pairs for each iteration
    std::vector<cudaEvent_t> starts(num_iterations), stops(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&stops[i]);
    }

    // Warmup: run a few iterations to ensure kernel is compiled and caches are warm
    for (int i = 0; i < 10; i++) {
        kernel->execute(d_input, d_output);
    }
    cudaDeviceSynchronize();  // No events recorded yet, use device sync

    // Queue all iterations with per-iteration events (no sync between)
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(starts[i]);
        kernel->execute(d_input, d_output);
        cudaEventRecord(stops[i]);
    }

    // Single sync at the end - wait for last event
    cudaEventSynchronize(stops[num_iterations - 1]);

    // Extract individual timings
    std::vector<float> timings(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventElapsedTime(&timings[i], starts[i], stops[i]);
    }

    // Sort for percentile calculation
    std::sort(timings.begin(), timings.end());

    TimingStats stats;
    stats.p50_ms = timings[num_iterations / 2];            // 50th percentile (median)
    stats.p90_ms = timings[(num_iterations * 90) / 100];   // 90th percentile

    // Cleanup
    for (int i = 0; i < num_iterations; i++) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    return stats;
}

// Helper: Run verification for a method
VerificationResult verify_method(SoftmaxKernel *kernel, const float *d_input,
                                 float *d_output, const float *h_input,
                                 float *h_output, const float *h_expected, int n) {
    VerificationResult result;
    result.skipped = false;
    result.passed = false;
    result.max_rel_error = -1.0;
    result.sum_error = -1.0;
    result.has_nan_inf = false;

    // Run kernel once
    kernel->execute(d_input, d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.skipped = true;
        return result;
    }

    // Transfer output
    err = cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        result.skipped = true;
        return result;
    }

    // Check for NaN/Inf
    for (int i = 0; i < n; i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) {
            result.has_nan_inf = true;
            result.passed = false;
            return result;
        }
    }

    // Compare against CPU reference
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double sum_gpu = 0.0;

    for (int i = 0; i < n; i++) {
        double abs_error = fabs(h_output[i] - h_expected[i]);
        double rel_error = abs_error / (fabs(h_expected[i]) + 1e-10);

        if (abs_error > max_abs_error) max_abs_error = abs_error;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
        sum_gpu += h_output[i];
    }

    double sum_error = fabs(sum_gpu - 1.0);

    result.max_rel_error = max_rel_error;
    result.sum_error = sum_error;
    result.passed = (max_rel_error < 1e-4 && sum_error < 1e-4);

    return result;
}

// ============================================================================
// BENCHMARK MODE: Table Printing Functions
// ============================================================================

void print_performance_table(BenchmarkResult results[][NUM_SIZES]) {
    printf("\n");
    printf("=======================================================================================\n");
    printf("                         SOFTMAX PERFORMANCE BENCHMARK\n");
    printf("=======================================================================================\n");
    printf("Iterations per test: 100\n");
    printf("Metric: P50/P90 execution time (ms) - median and 90th percentile\n");
    printf("Method: Per-iteration CUDA event timing (no CPU sync between iterations)\n\n");

    // Print header
    printf("%-15s", "Method");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf(" | %-13s", SIZE_LABELS[s]);
    }
    printf(" |\n");

    // Print separator
    printf("%-15s", "---------------");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf("-|--------------");
    }
    printf("-|\n");

    // Print data rows (P50/P90 format)
    for (int m = 0; m < NUM_METHODS; m++) {
        printf("%-15s", BENCHMARK_METHODS[m]);
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped) {
                printf(" | %-13s", "SKIPPED");
            } else if (results[m][s].p50_time_ms < 0) {
                printf(" | %-13s", "FAILED");
            } else {
                printf(" | %5.3f/%5.3f", results[m][s].p50_time_ms, results[m][s].p90_time_ms);
            }
        }
        printf(" |\n");
    }

    // Print skipped reasons
    printf("\nSKIPPED/FAILED reasons:\n");
    for (int m = 0; m < NUM_METHODS; m++) {
        bool has_skip = false;
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped || results[m][s].p50_time_ms < 0) {
                if (!has_skip) {
                    printf("- %s: %s\n", BENCHMARK_METHODS[m], results[m][s].skip_reason.c_str());
                    has_skip = true;
                }
            }
        }
    }

    // Find top 3 fastest per size (excluding naive method which is numerically unstable)
    printf("\nTOP 3 FASTEST per size by P50 (excluding naive):\n");
    for (int s = 0; s < NUM_SIZES; s++) {
        // Create array of (method_index, time) pairs
        struct MethodTime {
            int method_idx;
            float time;
        };
        MethodTime method_times[NUM_METHODS];
        int count = 0;

        // Collect valid methods
        for (int m = 0; m < NUM_METHODS; m++) {
            // Skip naive method - it's numerically unstable
            if (strcmp(BENCHMARK_METHODS[m], "naive") == 0) continue;

            if (!results[m][s].skipped && results[m][s].p50_time_ms > 0) {
                method_times[count].method_idx = m;
                method_times[count].time = results[m][s].p50_time_ms;
                count++;
            }
        }

        // Sort by time (simple selection sort for top 3)
        for (int i = 0; i < count && i < 3; i++) {
            for (int j = i + 1; j < count; j++) {
                if (method_times[j].time < method_times[i].time) {
                    MethodTime temp = method_times[i];
                    method_times[i] = method_times[j];
                    method_times[j] = temp;
                }
            }
        }

        // Print top 3
        printf("- %-5s: ", SIZE_LABELS[s]);
        int print_count = (count < 3) ? count : 3;
        for (int i = 0; i < print_count; i++) {
            if (i > 0) printf(", ");
            printf("%s (%.3f ms)", BENCHMARK_METHODS[method_times[i].method_idx], method_times[i].time);
        }
        printf("\n");
    }
    printf("=======================================================================================\n");
}

void print_verification_table(VerificationResult results[][NUM_SIZES]) {
    printf("\n");
    printf("=============================================================================\n");
    printf("                    SOFTMAX VERIFICATION RESULTS\n");
    printf("=============================================================================\n");
    printf("Reference: CPU softmax (numerically stable)\n");
    printf("Threshold: Max relative error < 1e-4, Sum error < 1e-4\n\n");

    // Print header
    printf("%-15s", "Method");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf(" | %-13s", SIZE_LABELS[s]);
    }
    printf(" |\n");

    // Print separator
    printf("%-15s", "---------------");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf("-|--------------");
    }
    printf("-|\n");

    // Print data rows
    for (int m = 0; m < NUM_METHODS; m++) {
        printf("%-15s", BENCHMARK_METHODS[m]);
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped) {
                printf(" | %-13s", "SKIPPED");
            } else if (results[m][s].has_nan_inf) {
                printf(" | %-13s", "FAIL (NaN)");
            } else if (results[m][s].passed) {
                printf(" | PASS (%.2e)", results[m][s].max_rel_error);
            } else {
                printf(" | FAIL (%.2e)", results[m][s].max_rel_error);
            }
        }
        printf(" |\n");
    }

    printf("\nLegend:\n");
    printf("- PASS (error): Verification passed, shows max relative error\n");
    printf("- FAIL (NaN):   Output contains NaN or Inf\n");
    printf("- FAIL (error): Error exceeds threshold\n");
    printf("- SKIPPED:      Method failed to execute\n");
    printf("=============================================================================\n");
}

// Main benchmark function: test all methods across all sizes
void benchmark_all_methods(int threadsPerBlock, bool verify) {
    printf("\n=============================================================================\n");
    printf("                    RUNNING COMPREHENSIVE BENCHMARK\n");
    printf("=============================================================================\n");
    printf("Methods to test: %d\n", NUM_METHODS);
    printf("Sizes to test: %d (16, 32, 64, 256, 512, 1K, 8K, 64K, 256K, 1M, 8M)\n", NUM_SIZES);
    printf("Iterations per test: 100\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    if (verify) {
        printf("Verification: ENABLED\n");
    }
    printf("=============================================================================\n\n");

    // Allocate result arrays
    BenchmarkResult perf_results[NUM_METHODS][NUM_SIZES];
    VerificationResult verify_results[NUM_METHODS][NUM_SIZES];

    // Initialize all results as skipped by default
    for (int m = 0; m < NUM_METHODS; m++) {
        for (int s = 0; s < NUM_SIZES; s++) {
            perf_results[m][s].p50_time_ms = -1.0f;
            perf_results[m][s].p90_time_ms = -1.0f;
            perf_results[m][s].skipped = true;
            perf_results[m][s].skip_reason = "Not run";

            verify_results[m][s].skipped = true;
            verify_results[m][s].passed = false;
            verify_results[m][s].max_rel_error = -1.0;
            verify_results[m][s].sum_error = -1.0;
            verify_results[m][s].has_nan_inf = false;
        }
    }

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
            float *h_output = (float*)malloc(n * sizeof(float));
            float *h_expected = nullptr;

            if (!h_output) {
                printf("SKIPPED (host allocation failed)\n");
                perf_results[m][s].skip_reason = "Host allocation failed";
                freeHostVector(h_input);
                continue;
            }

            // Allocate device memory
            float *d_input, *d_output;
            cudaError_t err = cudaMalloc(&d_input, n * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                perf_results[m][s].skip_reason = "Device allocation failed";
                freeHostVector(h_input);
                free(h_output);
                continue;
            }

            err = cudaMalloc(&d_output, n * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                perf_results[m][s].skip_reason = "Device allocation failed";
                freeHostVector(h_input);
                free(h_output);
                cudaFree(d_input);
                continue;
            }

            cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

            // Compute CPU reference if verification enabled
            if (verify) {
                h_expected = (float*)malloc(n * sizeof(float));
                if (h_expected) {
                    softmax_cpu_reference(h_input, h_expected, n);
                }
            }

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
                    perf_results[m][s].skip_reason = "Unknown method";
                } else {
                    // Run performance benchmark with per-iteration timing
                    TimingStats stats = get_timing_stats(kernel, d_input, d_output, n, 100);

                    perf_results[m][s].p50_time_ms = stats.p50_ms;
                    perf_results[m][s].p90_time_ms = stats.p90_ms;
                    perf_results[m][s].skipped = false;
                    perf_results[m][s].skip_reason = "";

                    // Run verification if enabled
                    if (verify && h_expected) {
                        VerificationResult vr = verify_method(kernel, d_input, d_output,
                                                              h_input, h_output, h_expected, n);
                        verify_results[m][s] = vr;

                        if (vr.has_nan_inf) {
                            printf("%.3f/%.3f ms [FAIL: NaN/Inf]\n", stats.p50_ms, stats.p90_ms);
                        } else if (vr.passed) {
                            printf("%.3f/%.3f ms [PASS: %.2e]\n", stats.p50_ms, stats.p90_ms, vr.max_rel_error);
                        } else if (!vr.skipped) {
                            printf("%.3f/%.3f ms [FAIL: %.2e]\n", stats.p50_ms, stats.p90_ms, vr.max_rel_error);
                        } else {
                            printf("%.3f/%.3f ms [verify skipped]\n", stats.p50_ms, stats.p90_ms);
                        }
                    } else {
                        printf("%.3f/%.3f ms\n", stats.p50_ms, stats.p90_ms);
                    }

                    delete kernel;
                }
            } catch (const std::exception &e) {
                printf("SKIPPED (exception: %s)\n", e.what());
                perf_results[m][s].skip_reason = std::string("Exception: ") + e.what();
                if (kernel) delete kernel;
            } catch (...) {
                printf("SKIPPED (unknown exception)\n");
                perf_results[m][s].skip_reason = "Unknown exception";
                if (kernel) delete kernel;
            }

            // Cleanup
            freeHostVector(h_input);
            free(h_output);
            if (h_expected) free(h_expected);
            cudaFree(d_input);
            cudaFree(d_output);
        }
        printf("\n");
    }

    // Print results
    printf("\n");
    print_performance_table(perf_results);

    if (verify) {
        printf("\n");
        print_verification_table(verify_results);
    }
}

// Softmax operation
void softmax_op(int n, int threadsPerBlock, bool verify, const char *method) {
    printf("Softmax of %d elements\n", n);
    printf("Method: %s\n", method);

    if (verify) {
        printf("Verification enabled\n");
    }

    // Allocate and initialize input vector
    float *h_input;
    allocateAndInitVector(&h_input, n);

    // Allocate output vector on host
    float *h_output = (float*)malloc(n * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host output memory\n");
        exit(EXIT_FAILURE);
    }

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
        free(h_output);
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

    // Transfer result back for verification
    if (verify) {
        // Run one final time to get a clean result (not timed)
        if (strcmp(method, "cudnn") == 0) {
            // Use class-based API
            CudnnSoftmax cudnn_kernel(n, threadsPerBlock);
            cudnn_kernel.execute(d_input, d_output);
        } else if (strcmp(method, "naive") == 0) {
            softmax_Naive(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "multi") == 0) {
            softmax_MultiPass(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused3") == 0) {
            softmax_Fused3(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused2") == 0) {
            softmax_Fused2(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused1") == 0) {
            softmax_Fused1(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "online") == 0) {
            softmax_Online(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "tiny") == 0) {
            softmax_Tiny(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "small") == 0) {
            softmax_Small(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "online_simple") == 0) {
            softmax_OnlineSimple(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "online_warp") == 0) {
            softmax_OnlineWarp(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "cub_block") == 0) {
            softmax_CubBlock(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "cub_device") == 0) {
            softmax_CubDevice(d_input, d_output, n, threadsPerBlock);
        }
        cudaDeviceSynchronize();

        transferFromDevice(h_output, d_output, n);

        printf("\nVerifying results...\n");

        // Calculate expected result on CPU
        float *h_expected = (float*)malloc(n * sizeof(float));
        if (!h_expected) {
            fprintf(stderr, "Failed to allocate CPU reference memory\n");
            exit(EXIT_FAILURE);
        }
        softmax_cpu_reference(h_input, h_expected, n);

        // Check for NaN/Inf (common with naive method for large inputs)
        bool has_nan_inf = false;
        for (int i = 0; i < n; i++) {
            if (isnan(h_output[i]) || isinf(h_output[i])) {
                has_nan_inf = true;
                break;
            }
        }

        if (has_nan_inf) {
            printf("✗ Verification FAILED: Output contains NaN or Inf\n");
            printf("  This is expected for naive method with large input values\n");
            printf("  (demonstrates numerical instability)\n");
        } else {
            // Compare GPU vs CPU
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            double sum_gpu = 0.0;

            for (int i = 0; i < n; i++) {
                double abs_error = fabs(h_output[i] - h_expected[i]);
                double rel_error = abs_error / (fabs(h_expected[i]) + 1e-10);

                if (abs_error > max_abs_error) max_abs_error = abs_error;
                if (rel_error > max_rel_error) max_rel_error = rel_error;

                sum_gpu += h_output[i];
            }

            // Check if sum is approximately 1.0 (probability distribution property)
            double sum_error = fabs(sum_gpu - 1.0);

            if (max_rel_error < 1e-4 && sum_error < 1e-4) {
                printf("✓ Verification PASSED\n");
                printf("  Max relative error: %.2e\n", max_rel_error);
                printf("  Sum(output) = %.6f (expected 1.0, error: %.2e)\n", sum_gpu, sum_error);
            } else {
                printf("✗ Verification FAILED\n");
                printf("  Max absolute error: %.6e\n", max_abs_error);
                printf("  Max relative error: %.2e\n", max_rel_error);
                printf("  Sum(output) = %.6f (expected 1.0, error: %.2e)\n", sum_gpu, sum_error);
            }
        }

        free(h_expected);
    }

    // Cleanup
    freeDeviceVector(d_input);
    freeDeviceVector(d_output);
    freeHostVector(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nSoftmax completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool verify = false;
    const char *method = "online_warp";  // Default method
    int n = 1 << 20;  // Default: 1 million elements
    int threadsPerBlock = 256;  // Default: 256 threads per block (optimal)

    static struct option long_options[] = {
        {"size",       required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"method",     required_argument, 0, 'm'},
        {"verify",     no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:b:m:vh", long_options, NULL)) != -1) {
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
            case 'v':
                verify = true;
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
        benchmark_all_methods(threadsPerBlock, verify);
    } else {
        // Single method mode (original behavior)
        softmax_op(n, threadsPerBlock, verify, method);
    }

    return 0;
}
