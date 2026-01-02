#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "softmax_kernels.h"
#include "vector_init.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N              Set array size (default: 1048576)\n");
    printf("  -b, --block-size N        Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Softmax method: 'naive', 'multi', 'fused', 'fused2', 'fused1', or 'online' (default: multi)\n");
    printf("  -v, --verify              Enable result verification\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  naive:   Naive exp(x)/sum (unstable - demonstrates overflow)\n");
    printf("  multi:   Multi-pass stable (max → exp-sum → normalize)\n");
    printf("  fused:   3-kernel fused (block stats → global reduce → normalize) [IMPLEMENTED]\n");
    printf("  fused2:  2-kernel fused (block stats → fused reduce+normalize) [SKELETON]\n");
    printf("  fused1:  1-kernel fused (single kernel, grid sync, cooperative groups) [SKELETON]\n");
    printf("  online:  Single-pass online algorithm (streaming max/sum) [SKELETON]\n");
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
    const int num_iterations = 1000;
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

    // Time the softmax operation (excludes input allocation/transfer)
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start);

        if (strcmp(method, "naive") == 0) {
            softmax_Naive(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "multi") == 0) {
            softmax_MultiPass(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused") == 0) {
            softmax_Fused(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused2") == 0) {
            softmax_Fused2(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused1") == 0) {
            softmax_Fused1(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "online") == 0) {
            softmax_Online(d_input, d_output, n, threadsPerBlock);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timings[i], start, stop);
    }
    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Transfer result back for verification
    if (verify) {
        // Run one final time to get a clean result (not timed)
        if (strcmp(method, "naive") == 0) {
            softmax_Naive(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "multi") == 0) {
            softmax_MultiPass(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused") == 0) {
            softmax_Fused(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused2") == 0) {
            softmax_Fused2(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "fused1") == 0) {
            softmax_Fused1(d_input, d_output, n, threadsPerBlock);
        } else if (strcmp(method, "online") == 0) {
            softmax_Online(d_input, d_output, n, threadsPerBlock);
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
    const char *method = "multi";  // Default method
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
                    strcmp(method, "fused") != 0 && strcmp(method, "fused2") != 0 &&
                    strcmp(method, "fused1") != 0 && strcmp(method, "online") != 0) {
                    fprintf(stderr, "Error: method must be 'naive', 'multi', 'fused', 'fused2', 'fused1', or 'online'\n");
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
    softmax_op(n, threadsPerBlock, verify, method);

    return 0;
}
