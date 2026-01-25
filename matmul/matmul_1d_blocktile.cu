#include "matmul_1d_blocktile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// 1D Block Tiling kernel (Kernel 4)
// Each thread computes TM elements in a column of C, increasing arithmetic intensity.
// This reduces the number of threads needed and improves register utilization.

#define BM_1D 64
#define BN_1D 64
#define BK_1D 8
#define TM_1D 8

// Number of threads per block: (BM/TM) * BN = 8 * 64 = 512
#define NUM_THREADS_1D ((BM_1D / TM_1D) * BN_1D)

__global__ void matmul1DBlocktileKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[BM_1D][BK_1D];
    __shared__ float Bs[BK_1D][BN_1D];

    // Thread indices
    // Each thread is responsible for loading and computing
    const int threadCol = threadIdx.x % BN_1D;
    const int threadRow = threadIdx.x / BN_1D;

    // Block position in output
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to the block's starting position
    A += blockRow * BM_1D * N;
    B += blockCol * BN_1D;
    C += blockRow * BM_1D * N + blockCol * BN_1D;

    // Thread results: each thread computes TM elements in a column
    float threadResults[TM_1D] = {0.0f};

    // Load indices for shared memory
    // For As: we need to load BM_1D * BK_1D elements = 64 * 8 = 512
    // For Bs: we need to load BK_1D * BN_1D elements = 8 * 64 = 512
    const int innerRowA = threadIdx.x / BK_1D;
    const int innerColA = threadIdx.x % BK_1D;
    const int innerRowB = threadIdx.x / BN_1D;
    const int innerColB = threadIdx.x % BN_1D;

    // Loop over K dimension in chunks of BK
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_1D) {
        // Load tile of A into shared memory
        if (blockRow * BM_1D + innerRowA < N && tileIdx + innerColA < N) {
            As[innerRowA][innerColA] = A[innerRowA * N + innerColA];
        } else {
            As[innerRowA][innerColA] = 0.0f;
        }

        // Load tile of B into shared memory
        if (tileIdx + innerRowB < N && blockCol * BN_1D + innerColB < N) {
            Bs[innerRowB][innerColB] = B[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB][innerColB] = 0.0f;
        }

        __syncthreads();

        // Move pointers for next iteration
        A += BK_1D;
        B += BK_1D * N;

        // Compute TM elements
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_1D; dotIdx++) {
            // Load B element once (same for all TM results)
            float tmpB = Bs[dotIdx][threadCol];
            #pragma unroll
            for (int resIdx = 0; resIdx < TM_1D; resIdx++) {
                // Each thread computes TM consecutive rows
                threadResults[resIdx] += As[threadRow * TM_1D + resIdx][dotIdx] * tmpB;
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int resIdx = 0; resIdx < TM_1D; resIdx++) {
        int globalRow = blockRow * BM_1D + threadRow * TM_1D + resIdx;
        int globalCol = blockCol * BN_1D + threadCol;
        if (globalRow < N && globalCol < N) {
            C[(threadRow * TM_1D + resIdx) * N + threadCol] = threadResults[resIdx];
        }
    }
}

Matmul1DBlocktile::Matmul1DBlocktile(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void Matmul1DBlocktile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_1D);
    dim3 blocks((N + BN_1D - 1) / BN_1D,
                (N + BM_1D - 1) / BM_1D);

    matmul1DBlocktileKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

Matmul1DBlocktile::~Matmul1DBlocktile() {
    // No workspace to free
}
