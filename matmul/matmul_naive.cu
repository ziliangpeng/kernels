#include "matmul_naive.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Naive matrix multiplication kernel
// Each thread computes one output element C[row][col]
__global__ void matmulNaiveKernel(const float *A, const float *B, float *C, int N) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (important for non-multiple-of-blockDim sizes)
    if (row < N && col < N) {
        float sum = 0.0f;

        // Dot product of row from A and column from B
        // A[row][k] = A[row * N + k]
        // B[k][col] = B[k * N + col]
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        // Store result
        C[row * N + col] = sum;
    }
}

// Constructor: Store configuration
MatmulNaive::MatmulNaive(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace allocation needed for naive kernel
}

// Execute: Launch kernel with 2D grid of 2D blocks
void MatmulNaive::execute(const float *d_A, const float *d_B, float *d_C) {
    // Configure 2D blocks and grid
    dim3 threads(blockDim, blockDim);  // e.g., 16×16 = 256 threads
    dim3 blocks((N + blockDim - 1) / blockDim,
                (N + blockDim - 1) / blockDim);

    // Launch kernel
    matmulNaiveKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    cudaCheckError(cudaGetLastError());
}

// Destructor: Nothing to free
MatmulNaive::~MatmulNaive() {
    // No workspace to free
}
