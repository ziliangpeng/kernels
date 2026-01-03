#include "vector_init.h"
#include "cuda_utils.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

void allocateAndInitVector(float **h_vec, int n) {
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    *h_vec = (float*)malloc(bytes);
    if (!*h_vec) {
        fprintf(stderr, "Failed to allocate host memory (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }

    // Initialize with random values in range [0, 100]
    // This range will cause overflow in naive softmax (exp(x) overflows for x > 88)
    // but stable methods (multi, fused) should handle it correctly
    for (int i = 0; i < n; i++) {
        (*h_vec)[i] = 100.0f * (float)rand() / RAND_MAX;
    }
}

void allocateDeviceVector(float **d_vec, int n) {
    size_t bytes = n * sizeof(float);
    cudaCheckError(cudaMalloc(d_vec, bytes));
}

void transferToDevice(float *d_vec, const float *h_vec, int n) {
    size_t bytes = n * sizeof(float);
    cudaCheckError(cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice));
}

void transferFromDevice(float *h_vec, const float *d_vec, int n) {
    size_t bytes = n * sizeof(float);
    cudaCheckError(cudaMemcpy(h_vec, d_vec, bytes, cudaMemcpyDeviceToHost));
}

void freeHostVector(float *h_vec) {
    free(h_vec);
}

void freeDeviceVector(float *d_vec) {
    cudaCheckError(cudaFree(d_vec));
}
