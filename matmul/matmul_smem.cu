#include "matmul_smem.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Shared Memory Tiling kernel (Kernel 3)
// Key optimization: Load tiles of A and B into shared memory to reduce
// global memory accesses. Each element from GMEM is reused multiple times.

#define SMEM_TILE 32

__global__ void matmulSmemKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[SMEM_TILE][SMEM_TILE];
    __shared__ float Bs[SMEM_TILE][SMEM_TILE];

    // Thread position within the block (using 2D thread indexing conceptually)
    // We launch with 32x32 = 1024 threads, but CUDA limits to 1024 so this is exactly at limit
    // Each thread handles one element in the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block position in output grid
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Global row and column indices for this thread's output element
    int row = blockRow * SMEM_TILE + ty;
    int col = blockCol * SMEM_TILE + tx;

    // Accumulator for this thread's output element
    float sum = 0.0f;

    // Loop over tiles along the K dimension
    for (int tileIdx = 0; tileIdx < N; tileIdx += SMEM_TILE) {
        // Load tile of A into shared memory
        // As[ty][tx] = A[row][tileIdx + tx]
        int A_col = tileIdx + tx;
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        // Bs[ty][tx] = B[tileIdx + ty][col]
        int B_row = tileIdx + ty;
        if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Ensure all threads have loaded their data
        __syncthreads();

        // Compute partial dot product using shared memory
        #pragma unroll
        for (int k = 0; k < SMEM_TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Ensure all threads have finished using the tiles before loading new ones
        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

MatmulSmem::MatmulSmem(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void MatmulSmem::execute(const float *d_A, const float *d_B, float *d_C) {
    // Use 2D blocks with SMEM_TILE x SMEM_TILE threads
    dim3 threads(SMEM_TILE, SMEM_TILE);  // 32x32 = 1024 threads
    dim3 blocks((N + SMEM_TILE - 1) / SMEM_TILE,
                (N + SMEM_TILE - 1) / SMEM_TILE);

    matmulSmemKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

MatmulSmem::~MatmulSmem() {
    // No workspace to free
}
