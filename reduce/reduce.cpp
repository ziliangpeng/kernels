#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "reduce/sum_reduce.h"
#include "reduce/sum_reduce_atomic.h"
#include "vector_init.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N              Set array size (default: 1048576)\n");
    printf("  -b, --block-size N        Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Reduction method: 'gpu', 'threshold', or 'atomic' (default: threshold)\n");
    printf("  -t, --cpu-threshold N     CPU threshold for 'threshold' method (default: 1000)\n");
    printf("  -w, --warp-opt            Use warp shuffle optimization (requires block size >= 64 and power of 2)\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  gpu:       Fully GPU recursive reduction (reduce until 1 element)\n");
    printf("  threshold: GPU reduction until size <= threshold, then CPU final sum\n");
    printf("  atomic:    Single kernel using atomicAdd (simple but serializes)\n");
    printf("\nOptimization:\n");
    printf("  --warp-opt: Uses warp shuffles instead of __syncthreads for final 32→1\n");
    printf("              Requires block size >= 64 and power of 2 (64, 128, 256, 512, 1024)\n");
    printf("              Only applies to gpu/threshold methods\n");
    printf("              Expected 8-10%% speedup\n");
}

// SUM reduction operation
void sum_op(int n, int threadsPerBlock, const char *method, int cpuThreshold, bool useWarpOpt) {
    printf("Vector sum reduction of %d elements\n", n);
    printf("Method: %s", method);
    if (strcmp(method, "threshold") == 0) {
        printf(" (cpu-threshold: %d)", cpuThreshold);
    }
    if (useWarpOpt) {
        printf(" [warp-optimized]");
    }
    printf("\n");

    // Allocate and initialize input vector
    float *h_input;
    allocateAndInitVector(&h_input, n);

    float *d_input;
    allocateDeviceVector(&d_input, n);
    transferToDevice(d_input, h_input, n);

    // Configure timing
    const int num_iterations = 1000;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (!timings) {
        fprintf(stderr, "Failed to allocate memory for timings\n");
        freeHostVector(h_input);
        freeDeviceVector(d_input);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running reduction %d times to collect statistics...\n", num_iterations);

    // Time the reduction operation (excludes input allocation/transfer)
    float result = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start);

        if (strcmp(method, "gpu") == 0) {
            if (useWarpOpt) {
                result = vectorSum_GPU_Warp(d_input, n, threadsPerBlock);
            } else {
                result = vectorSum_GPU(d_input, n, threadsPerBlock);
            }
        } else if (strcmp(method, "threshold") == 0) {
            if (useWarpOpt) {
                result = vectorSum_Threshold_Warp(d_input, n, threadsPerBlock, cpuThreshold);
            } else {
                result = vectorSum_Threshold(d_input, n, threadsPerBlock, cpuThreshold);
            }
        } else if (strcmp(method, "atomic") == 0) {
            result = vectorSum_Atomic(d_input, n, threadsPerBlock);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timings[i], start, stop);
    }
    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    printf("\nSum result: %f\n", result);

    // Cleanup
    freeDeviceVector(d_input);
    freeHostVector(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(timings);

    printf("\nReduction completed successfully!\n");
}

// Helper function to check if a number is a power of 2
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool useWarpOpt = false;  // Warp shuffle optimization flag
    const char *method = "threshold";  // Default method
    int n = 1 << 20;  // Default: 1 million elements
    int threadsPerBlock = 256;  // Default: 256 threads per block
    int cpuThreshold = 1000;  // Default CPU threshold

    static struct option long_options[] = {
        {"size",          required_argument, 0, 'n'},
        {"block-size",    required_argument, 0, 'b'},
        {"method",        required_argument, 0, 'm'},
        {"cpu-threshold", required_argument, 0, 't'},
        {"warp-opt",      no_argument,       0, 'w'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:b:m:t:wh", long_options, NULL)) != -1) {
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
                if (strcmp(method, "gpu") != 0 && strcmp(method, "threshold") != 0 && strcmp(method, "atomic") != 0) {
                    fprintf(stderr, "Error: method must be 'gpu', 'threshold', or 'atomic'\n");
                    return 1;
                }
                break;
            case 't':
                cpuThreshold = atoi(optarg);
                if (cpuThreshold <= 0) {
                    fprintf(stderr, "Error: cpu-threshold must be positive\n");
                    return 1;
                }
                break;
            case 'w':
                useWarpOpt = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Validate warp optimization requirements
    if (useWarpOpt && (threadsPerBlock < 64 || !isPowerOfTwo(threadsPerBlock))) {
        fprintf(stderr, "Error: --warp-opt requires block size >= 64 and power of 2\n");
        fprintf(stderr, "       Current block size: %d\n", threadsPerBlock);
        fprintf(stderr, "       Valid sizes: 64, 128, 256, 512, 1024\n");
        return 1;
    }

    // Initialize random seed
    srand(time(NULL));

    // Run sum reduction operation
    sum_op(n, threadsPerBlock, method, cpuThreshold, useWarpOpt);

    return 0;
}
