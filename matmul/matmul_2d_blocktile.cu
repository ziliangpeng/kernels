#include "matmul_2d_blocktile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// 2D Block Tiling kernel (Kernel 5)
// Each thread computes a TM x TN = 8x8 = 64 element tile of C
// Uses outer product formulation for maximum register reuse

#define BM_2D 128
#define BN_2D 128
#define BK_2D 8
#define TM_2D 8
#define TN_2D 8

// Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256
#define NUM_THREADS_2D ((BM_2D / TM_2D) * (BN_2D / TN_2D))

__global__ void matmul2DBlocktileKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[BM_2D][BK_2D];
    __shared__ float Bs[BK_2D][BN_2D];

    // Thread position in the output tile grid
    const int threadCol = threadIdx.x % (BN_2D / TN_2D);  // 0-15
    const int threadRow = threadIdx.x / (BN_2D / TN_2D);  // 0-15

    // Block position in output
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to the block's starting position
    A += blockRow * BM_2D * N;
    B += blockCol * BN_2D;
    C += blockRow * BM_2D * N + blockCol * BN_2D;

    // Thread results: each thread computes TM x TN elements
    float threadResults[TM_2D][TN_2D] = {{0.0f}};

    // Registers for A and B values
    float regA[TM_2D];
    float regB[TN_2D];

    // Calculate load indices
    // We have 256 threads and need to load BM_2D * BK_2D = 128 * 8 = 1024 elements for A
    // Each thread loads 1024/256 = 4 elements
    const int strideA = NUM_THREADS_2D / BK_2D;  // 256 / 8 = 32 rows per iteration
    const int strideB = NUM_THREADS_2D / BN_2D;  // 256 / 128 = 2 rows per iteration

    const int innerRowA = threadIdx.x / BK_2D;
    const int innerColA = threadIdx.x % BK_2D;
    const int innerRowB = threadIdx.x / BN_2D;
    const int innerColB = threadIdx.x % BN_2D;

    // Loop over K dimension in chunks of BK
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_2D) {
        // Load tile of A into shared memory (each thread loads multiple elements)
        for (int loadOffset = 0; loadOffset < BM_2D; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM_2D + row < N && tileIdx + innerColA < N) {
                As[row][innerColA] = A[row * N + innerColA];
            } else {
                As[row][innerColA] = 0.0f;
            }
        }

        // Load tile of B into shared memory (each thread loads multiple elements)
        for (int loadOffset = 0; loadOffset < BK_2D; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN_2D + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = 0.0f;
            }
        }

        __syncthreads();

        // Move pointers for next iteration
        A += BK_2D;
        B += BK_2D * N;

        // Compute using outer product
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_2D; dotIdx++) {
            // Load A column into registers
            #pragma unroll
            for (int i = 0; i < TM_2D; i++) {
                regA[i] = As[threadRow * TM_2D + i][dotIdx];
            }

            // Load B row into registers
            #pragma unroll
            for (int j = 0; j < TN_2D; j++) {
                regB[j] = Bs[dotIdx][threadCol * TN_2D + j];
            }

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM_2D; i++) {
                #pragma unroll
                for (int j = 0; j < TN_2D; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM_2D; i++) {
        #pragma unroll
        for (int j = 0; j < TN_2D; j++) {
            int globalRow = blockRow * BM_2D + threadRow * TM_2D + i;
            int globalCol = blockCol * BN_2D + threadCol * TN_2D + j;
            if (globalRow < N && globalCol < N) {
                C[(threadRow * TM_2D + i) * N + threadCol * TN_2D + j] = threadResults[i][j];
            }
        }
    }
}

Matmul2DBlocktile::Matmul2DBlocktile(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void Matmul2DBlocktile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_2D);
    dim3 blocks((N + BN_2D - 1) / BN_2D,
                (N + BM_2D - 1) / BM_2D);

    matmul2DBlocktileKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

Matmul2DBlocktile::~Matmul2DBlocktile() {
    // No workspace to free
}
