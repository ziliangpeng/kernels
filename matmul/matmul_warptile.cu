#include "matmul_warptile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Warp Tiling kernel (Kernel 10)
// Adds warp-level tiling between block and thread levels
// Three-level hierarchy: Block -> Warp -> Thread

#define BM_WARP 128
#define BN_WARP 128
#define BK_WARP 16
#define WM 64       // Warp tile M dimension
#define WN 64       // Warp tile N dimension
#define TM_WARP 8   // Thread tile M dimension
#define TN_WARP 4   // Thread tile N dimension
#define WARP_SIZE 32

// Warps per block: (BM/WM) * (BN/WN) = 2 * 2 = 4
#define WARPS_PER_BLOCK_X (BN_WARP / WN)  // 2
#define WARPS_PER_BLOCK_Y (BM_WARP / WM)  // 2
#define NUM_WARPS (WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y)  // 4

// Threads per warp tile: (WM/TM) * (WN/TN) = 8 * 16 = 128
// But we only have 32 threads per warp, so each thread computes multiple tiles
// Threads per warp compute: (WM*WN) / (TM*TN*32) = 4096 / 1024 = 4 tiles each? No...
// Let's recalculate:
// - Warp produces WM x WN = 64 x 64 = 4096 elements
// - Each thread produces TM x TN = 8 x 4 = 32 elements
// - So we need 4096/32 = 128 threads, but warp has only 32
// - So each thread computes (128/32) = 4 sets of TM x TN

#define WARP_ITER_M (WM / (TM_WARP * (WARP_SIZE / (WN / TN_WARP))))  // iterations in M
#define WARP_ITER_N (WN / TN_WARP / (WARP_SIZE / (WN / TN_WARP)))   // No extra iter in N

// Simpler approach: each thread in warp handles multiple TM x TN tiles
// Warp layout: 8 threads in N direction (handling 8 * TN = 32 cols)
//              4 threads in M direction (handling 4 * TM = 32 rows)
// But we need 64x64, so each thread does 2x2 = 4 tiles in M direction
#define WARP_THREAD_M 4   // Threads in M direction per warp
#define WARP_THREAD_N 8   // Threads in N direction per warp (4*8 = 32)
#define WARP_SUBTILE_M 2  // Each thread computes this many TM-tiles in M
#define WARP_SUBTILE_N 2  // Each thread computes this many TN-tiles in N

// Verify: WARP_THREAD_M * TM_WARP * WARP_SUBTILE_M = 4 * 8 * 2 = 64 = WM ✓
// Verify: WARP_THREAD_N * TN_WARP * WARP_SUBTILE_N = 8 * 4 * 2 = 64 = WN ✓

// Threads per block: NUM_WARPS * WARP_SIZE = 4 * 32 = 128
#define NUM_THREADS_WARP (NUM_WARPS * WARP_SIZE)

__global__ void matmulWarptileKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory: As transposed
    __shared__ float As[BK_WARP][BM_WARP];
    __shared__ float Bs[BK_WARP][BN_WARP];

    // Warp and thread indices
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Warp position in block tile
    const int warpRow = warpId / WARPS_PER_BLOCK_X;  // 0-1
    const int warpCol = warpId % WARPS_PER_BLOCK_X;  // 0-1

    // Thread position within warp tile
    const int threadRowInWarp = laneId / WARP_THREAD_N;  // 0-3
    const int threadColInWarp = laneId % WARP_THREAD_N;  // 0-7

    // Block position in output
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to block's starting position
    A += blockRow * BM_WARP * N;
    B += blockCol * BN_WARP;
    C += blockRow * BM_WARP * N + blockCol * BN_WARP;

    // Thread results: each thread computes WARP_SUBTILE_M * WARP_SUBTILE_N tiles of TM x TN
    float threadResults[WARP_SUBTILE_M * TM_WARP][WARP_SUBTILE_N * TN_WARP] = {{0.0f}};

    // Registers for A and B values
    float regA[WARP_SUBTILE_M * TM_WARP];
    float regB[WARP_SUBTILE_N * TN_WARP];

    // Load indices - we have 128 threads
    // A tile: BM * BK = 128 * 16 = 2048 elements, 2048/128 = 16 per thread
    // B tile: BK * BN = 16 * 128 = 2048 elements, 2048/128 = 16 per thread
    const int strideA = NUM_THREADS_WARP / BK_WARP;  // 128 / 16 = 8
    const int strideB = NUM_THREADS_WARP / BN_WARP;  // 128 / 128 = 1

    const int innerRowA = threadIdx.x / BK_WARP;
    const int innerColA = threadIdx.x % BK_WARP;
    const int innerRowB = threadIdx.x / BN_WARP;
    const int innerColB = threadIdx.x % BN_WARP;

    // Loop over K dimension
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_WARP) {
        // Load A tile (transposed into As[BK][BM])
        for (int loadOffset = 0; loadOffset < BM_WARP; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM_WARP + row < N && tileIdx + innerColA < N) {
                As[innerColA][row] = A[row * N + innerColA];
            } else {
                As[innerColA][row] = 0.0f;
            }
        }

        // Load B tile
        for (int loadOffset = 0; loadOffset < BK_WARP; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN_WARP + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = 0.0f;
            }
        }

        __syncthreads();

        A += BK_WARP;
        B += BK_WARP * N;

        // Compute using warp tiling
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_WARP; dotIdx++) {
            // Load A values for all subtiles this thread computes
            #pragma unroll
            for (int subtileM = 0; subtileM < WARP_SUBTILE_M; subtileM++) {
                #pragma unroll
                for (int i = 0; i < TM_WARP; i++) {
                    // Position in warp tile + subtile offset + thread offset + element
                    int asRow = warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) +
                               threadRowInWarp * TM_WARP + i;
                    regA[subtileM * TM_WARP + i] = As[dotIdx][asRow];
                }
            }

            // Load B values for all subtiles
            #pragma unroll
            for (int subtileN = 0; subtileN < WARP_SUBTILE_N; subtileN++) {
                #pragma unroll
                for (int j = 0; j < TN_WARP; j++) {
                    int bsCol = warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) +
                               threadColInWarp * TN_WARP + j;
                    regB[subtileN * TN_WARP + j] = Bs[dotIdx][bsCol];
                }
            }

            // Outer product for all subtiles
            #pragma unroll
            for (int i = 0; i < WARP_SUBTILE_M * TM_WARP; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_SUBTILE_N * TN_WARP; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int subtileM = 0; subtileM < WARP_SUBTILE_M; subtileM++) {
        #pragma unroll
        for (int i = 0; i < TM_WARP; i++) {
            int globalRow = blockRow * BM_WARP + warpRow * WM +
                           subtileM * (WM / WARP_SUBTILE_M) +
                           threadRowInWarp * TM_WARP + i;

            if (globalRow < N) {
                #pragma unroll
                for (int subtileN = 0; subtileN < WARP_SUBTILE_N; subtileN++) {
                    #pragma unroll
                    for (int j = 0; j < TN_WARP; j++) {
                        int globalCol = blockCol * BN_WARP + warpCol * WN +
                                       subtileN * (WN / WARP_SUBTILE_N) +
                                       threadColInWarp * TN_WARP + j;

                        if (globalCol < N) {
                            int localRow = warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) +
                                          threadRowInWarp * TM_WARP + i;
                            int localCol = warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) +
                                          threadColInWarp * TN_WARP + j;
                            C[localRow * N + localCol] = threadResults[subtileM * TM_WARP + i][subtileN * TN_WARP + j];
                        }
                    }
                }
            }
        }
    }
}

MatmulWarptile::MatmulWarptile(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void MatmulWarptile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_WARP);
    dim3 blocks((N + BN_WARP - 1) / BN_WARP,
                (N + BM_WARP - 1) / BM_WARP);

    matmulWarptileKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

MatmulWarptile::~MatmulWarptile() {
    // No workspace to free
}
