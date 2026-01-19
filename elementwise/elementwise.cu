#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "elementwise/vector_add.h"
#include "elementwise/vector_mul.h"
#include "elementwise/vector_fma.h"
#include "vector_init.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N         Set array size (default: 1048576)\n");
    printf("  -b, --block-size N   Set threads per block (default: 256)\n");
    printf("  -m, --mode MODE      Computation mode: 'add' or 'vma' (default: add)\n");
    printf("  -f, --fused          Use fused kernel for VMA (default: separate)\n");
    printf("  -V, --vectorized     Use float4 vectorized kernel (requires --fused, VMA mode)\n");
    printf("  -h, --help           Show this help message\n");
    printf("\nModes:\n");
    printf("  add: c = a + b (2 inputs, 1 output)\n");
    printf("  vma: d = (a * b) + c (3 inputs, 1 output)\n");
    printf("       --fused: use single FMA kernel\n");
    printf("       --fused --vectorized: use float4 FMA kernel (4 elements/thread)\n");
    printf("       (default): use separate mul then add kernels\n");
}

// ADD operation: c = a + b
void add_op(int n, int threadsPerBlock) {
    printf("Vector operation: add mode on %d elements\n", n);

    // Allocate and initialize host vectors
    float *h_a, *h_b, *h_c;
    allocateAndInitVector(&h_a, n);
    allocateAndInitVector(&h_b, n);
    allocateAndInitVector(&h_c, n);

    // Allocate device vectors
    float *d_a, *d_b, *d_c;
    allocateDeviceVector(&d_a, n);
    allocateDeviceVector(&d_b, n);
    allocateDeviceVector(&d_c, n);

    // Transfer input vectors to device
    transferToDevice(d_a, h_a, n);
    transferToDevice(d_b, h_b, n);

    // Configure kernel launch parameters
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing
    cudaEvent_t kernel_start, kernel_stop;
    cudaCheckError(cudaEventCreate(&kernel_start));
    cudaCheckError(cudaEventCreate(&kernel_stop));

    // Run kernel multiple times to collect statistics
    const int num_iterations = 1000;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (timings == NULL) {
        fprintf(stderr, "Failed to allocate timings array\n");
        exit(EXIT_FAILURE);
    }

    printf("Running kernel %d times to collect statistics...\n", num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(kernel_start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        cudaEventRecord(kernel_stop);

        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&timings[i], kernel_start, kernel_stop);
    }
    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Transfer result back to host
    transferFromDevice(h_c, d_c, n);
    printf("Vector operation completed successfully!\n");

    // Cleanup
    freeDeviceVector(d_a);
    freeDeviceVector(d_b);
    freeDeviceVector(d_c);
    freeHostVector(h_a);
    freeHostVector(h_b);
    freeHostVector(h_c);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    free(timings);
}

// VMA operation: d = (a * b) + c
void vma_op(int n, int threadsPerBlock, bool fused, bool vectorized) {
    printf("Vector operation: vma mode on %d elements\n", n);
    if (fused && vectorized) {
        printf("Using fused FMA kernel with float4 vectorization\n");
    } else if (fused) {
        printf("Using fused FMA kernel\n");
    } else {
        printf("Using separate multiply + add kernels\n");
    }

    // Allocate and initialize host vectors
    float *h_a, *h_b, *h_c, *h_d;
    allocateAndInitVector(&h_a, n);
    allocateAndInitVector(&h_b, n);
    allocateAndInitVector(&h_c, n);
    allocateAndInitVector(&h_d, n);

    // Allocate device vectors
    float *d_a, *d_b, *d_c, *d_d;
    allocateDeviceVector(&d_a, n);
    allocateDeviceVector(&d_b, n);
    allocateDeviceVector(&d_c, n);
    allocateDeviceVector(&d_d, n);

    // Transfer input vectors to device
    transferToDevice(d_a, h_a, n);
    transferToDevice(d_b, h_b, n);
    transferToDevice(d_c, h_c, n);

    // Configure kernel launch parameters
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing
    cudaEvent_t kernel_start, kernel_stop;
    cudaCheckError(cudaEventCreate(&kernel_start));
    cudaCheckError(cudaEventCreate(&kernel_stop));

    // Run kernel multiple times to collect statistics
    const int num_iterations = 1000;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (timings == NULL) {
        fprintf(stderr, "Failed to allocate timings array\n");
        exit(EXIT_FAILURE);
    }

    // Allocate temp buffer once for separate mode (reuse across iterations)
    float *d_temp = NULL;
    if (!fused) {
        allocateDeviceVector(&d_temp, n);
    }

    printf("Running kernel %d times to collect statistics...\n", num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(kernel_start);

        if (fused && vectorized) {
            // VMA vectorized fused: each thread processes 4 elements
            int n_vec = n / 4;
            int blocksPerGrid_vec = (n_vec + threadsPerBlock - 1) / threadsPerBlock;
            vectorFMA_float4<<<blocksPerGrid_vec, threadsPerBlock>>>(d_a, d_b, d_c, d_d, n);
        } else if (fused) {
            // VMA fused mode: d = (a * b) + c
            vectorFMA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_d, n);
        } else {
            // VMA separate mode: temp = a * b, then d = temp + c
            // No sync needed - kernels auto-order on same stream
            vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_temp, n);
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_c, d_d, n);
        }

        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&timings[i], kernel_start, kernel_stop);
    }
    cudaCheckError(cudaGetLastError());

    // Free temp buffer if allocated
    if (d_temp) {
        freeDeviceVector(d_temp);
    }

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Transfer result back to host
    transferFromDevice(h_d, d_d, n);
    printf("Vector operation completed successfully!\n");

    // Cleanup
    freeDeviceVector(d_a);
    freeDeviceVector(d_b);
    freeDeviceVector(d_c);
    freeDeviceVector(d_d);
    freeHostVector(h_a);
    freeHostVector(h_b);
    freeHostVector(h_c);
    freeHostVector(h_d);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    free(timings);
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool fused = false;
    bool vectorized = false;
    const char *mode = "add";  // Default mode
    int n = 1 << 20;  // Default: 1 million elements
    // Default: 256 threads per block (NVIDIA recommended, 8 warps, optimal for most kernels)
    int threadsPerBlock = 256;

    static struct option long_options[] = {
        {"size",       required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"mode",       required_argument, 0, 'm'},
        {"fused",      no_argument,       0, 'f'},
        {"vectorized", no_argument,       0, 'V'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:b:m:fVh", long_options, NULL)) != -1) {
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
                mode = optarg;
                if (strcmp(mode, "add") != 0 && strcmp(mode, "vma") != 0) {
                    fprintf(stderr, "Error: mode must be 'add' or 'vma'\n");
                    return 1;
                }
                break;
            case 'f':
                fused = true;
                break;
            case 'V':
                vectorized = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Validate flags
    if (vectorized && !fused) {
        fprintf(stderr, "Error: --vectorized requires --fused flag\n");
        return 1;
    }

    if (vectorized && strcmp(mode, "vma") != 0) {
        fprintf(stderr, "Error: --vectorized only supported for VMA mode\n");
        return 1;
    }

    if (vectorized && n % 4 != 0) {
        fprintf(stderr, "Error: --vectorized requires array size to be multiple of 4\n");
        return 1;
    }

    // Initialize random seed
    srand(time(NULL));

    // Call appropriate operation function
    if (strcmp(mode, "add") == 0) {
        add_op(n, threadsPerBlock);
    } else if (strcmp(mode, "vma") == 0) {
        vma_op(n, threadsPerBlock, fused, vectorized);
    } else {
        fprintf(stderr, "Error: operation '%s' is not supported\n", mode);
        return 1;
    }

    return 0;
}
