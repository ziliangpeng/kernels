#include "matmul_coalesced.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Global Memory Coalescing kernel (Kernel 2)
// Key insight: Change thread indexing so consecutive threads access consecutive
// memory locations in B matrix, enabling coalesced memory access.
//
// In naive kernel:
//   row = blockIdx.y * blockDim.y + threadIdx.y
//   col = blockIdx.x * blockDim.x + threadIdx.x
// This causes threads in the same warp to access B[k][col] with consecutive cols,
// which is fine, but we can do better by ensuring all memory accesses are optimal.
//
// The key fix: ensure thread indexing maps to memory layout correctly.
// For row-major matrices, consecutive threads should access consecutive columns.

__global__ void matmulCoalescedKernel(const float *A, const float *B, float *C, int N) {
    // Use 1D block indexing mapped to 2D output
    // This ensures consecutive threads access consecutive memory
    const int BLOCKSIZE = 32;

    // Thread position within block
    int threadCol = threadIdx.x % BLOCKSIZE;
    int threadRow = threadIdx.x / BLOCKSIZE;

    // Block position in output grid
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Global position
    int row = blockRow * BLOCKSIZE + threadRow;
    int col = blockCol * BLOCKSIZE + threadCol;

    // Boundary check
    if (row < N && col < N) {
        float sum = 0.0f;

        // Pointers to the start of the row in A and start of matrix B
        const float *A_row = A + row * N;
        const float *B_col = B + col;

        // Compute dot product
        for (int k = 0; k < N; k++) {
            // A[row][k] - threads in same warp access different rows (not coalesced for A)
            // B[k][col] - threads in same warp access consecutive cols (coalesced!)
            sum += A_row[k] * B_col[k * N];
        }

        C[row * N + col] = sum;
    }
}

MatmulCoalesced::MatmulCoalesced(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void MatmulCoalesced::execute(const float *d_A, const float *d_B, float *d_C) {
    const int BLOCKSIZE = 32;

    // Using 1D blocks with BLOCKSIZE*BLOCKSIZE threads
    dim3 threads(BLOCKSIZE * BLOCKSIZE);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (N + BLOCKSIZE - 1) / BLOCKSIZE);

    matmulCoalescedKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

MatmulCoalesced::~MatmulCoalesced() {
    // No workspace to free
}
