#include "matrix_init.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Allocate and initialize two square matrices with random values in [0, 1]
void allocateAndInitMatrices(float **h_A, float **h_B, int N) {
    // Allocate host memory for both matrices
    *h_A = (float*)malloc(N * N * sizeof(float));
    *h_B = (float*)malloc(N * N * sizeof(float));

    // Check for allocation failure
    if (!*h_A || !*h_B) {
        // Clean up partial allocations
        if (*h_A) free(*h_A);
        if (*h_B) free(*h_B);
        *h_A = *h_B = nullptr;
        fprintf(stderr, "Failed to allocate %dx%d matrices (%zu bytes each)\n",
                N, N, N * N * sizeof(float));
        return;
    }

    // Initialize random seed (use static variable to only seed once)
    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }

    // Initialize matrices with random values in [0, 1]
    for (int i = 0; i < N * N; i++) {
        (*h_A)[i] = (float)rand() / RAND_MAX;
        (*h_B)[i] = (float)rand() / RAND_MAX;
    }
}

// CPU reference implementation using double precision for accuracy
// Computes C = A × B where all matrices are N×N, row-major layout
void matmul_cpu_reference(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Use double precision for accumulation to minimize error
            double sum = 0.0;

            // Dot product of row i from A and column j from B
            for (int k = 0; k < N; k++) {
                sum += (double)A[i * N + k] * (double)B[k * N + j];
            }

            // Store result (cast back to float)
            C[i * N + j] = (float)sum;
        }
    }
}
