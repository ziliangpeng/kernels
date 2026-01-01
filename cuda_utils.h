#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to check CUDA errors
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Calculate and print timing statistics (min, max, mean, median, P90, P95, P99)
// Takes ownership of timings array and frees it
void calculate_and_print_statistics(float *timings, int num_iterations);

#endif
