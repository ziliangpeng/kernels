#include "matmul_warptile_typed.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>

// Warp-tiling matmul (rung 10) with dtype-templated storage.
// Strict mirror of matmul_warptile.cu (BM=128 BN=128 BK=16 WM=64 WN=64
// TM=8 TN=4, 128 threads, transposed As). Only differences: SMEM arrays hold
// Traits::T, loads convert to float once (Traits::to_float) before entering
// the identical regA/regB outer-product nest, and inputs are converted to
// 16-bit storage once outside the timed region.

#define BM_WARP 128
#define BN_WARP 128
#define BK_WARP 16
#define WM 64
#define WN 64
#define TM_WARP 8
#define TN_WARP 4
#define WARP_SIZE 32

#define WARPS_PER_BLOCK_X (BN_WARP / WN)
#define WARPS_PER_BLOCK_Y (BM_WARP / WM)
#define NUM_WARPS (WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y)

#define WARP_THREAD_M 4
#define WARP_THREAD_N 8
#define WARP_SUBTILE_M 2
#define WARP_SUBTILE_N 2

#define NUM_THREADS_WARP (NUM_WARPS * WARP_SIZE)

template <typename Traits>
__global__ void convertWarpT(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
__global__ void matmulWarptileTypedKernel(const typename Traits::T *A,
                                          const typename Traits::T *B,
                                          float *C, int N) {
    __shared__ typename Traits::T As[BK_WARP][BM_WARP];  // transposed [k][m]
    __shared__ typename Traits::T Bs[BK_WARP][BN_WARP];

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int warpRow = warpId / WARPS_PER_BLOCK_X;
    const int warpCol = warpId % WARPS_PER_BLOCK_X;

    const int threadRowInWarp = laneId / WARP_THREAD_N;
    const int threadColInWarp = laneId % WARP_THREAD_N;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_WARP * N;
    B += blockCol * BN_WARP;
    C += blockRow * BM_WARP * N + blockCol * BN_WARP;

    float threadResults[WARP_SUBTILE_M * TM_WARP][WARP_SUBTILE_N * TN_WARP] = {{0.0f}};
    float regA[WARP_SUBTILE_M * TM_WARP];
    float regB[WARP_SUBTILE_N * TN_WARP];

    const int strideA = NUM_THREADS_WARP / BK_WARP;
    const int strideB = NUM_THREADS_WARP / BN_WARP;

    const int innerRowA = threadIdx.x / BK_WARP;
    const int innerColA = threadIdx.x % BK_WARP;
    const int innerRowB = threadIdx.x / BN_WARP;
    const int innerColB = threadIdx.x % BN_WARP;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_WARP) {
        for (int loadOffset = 0; loadOffset < BM_WARP; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM_WARP + row < N && tileIdx + innerColA < N) {
                As[innerColA][row] = A[row * N + innerColA];
            } else {
                As[innerColA][row] = Traits::zero();
            }
        }

        for (int loadOffset = 0; loadOffset < BK_WARP; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN_WARP + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = Traits::zero();
            }
        }

        __syncthreads();

        A += BK_WARP;
        B += BK_WARP * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_WARP; dotIdx++) {
            #pragma unroll
            for (int subtileM = 0; subtileM < WARP_SUBTILE_M; subtileM++) {
                #pragma unroll
                for (int i = 0; i < TM_WARP; i++) {
                    int asRow = warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) +
                               threadRowInWarp * TM_WARP + i;
                    regA[subtileM * TM_WARP + i] = Traits::to_float(As[dotIdx][asRow]);
                }
            }

            #pragma unroll
            for (int subtileN = 0; subtileN < WARP_SUBTILE_N; subtileN++) {
                #pragma unroll
                for (int j = 0; j < TN_WARP; j++) {
                    int bsCol = warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) +
                               threadColInWarp * TN_WARP + j;
                    regB[subtileN * TN_WARP + j] = Traits::to_float(Bs[dotIdx][bsCol]);
                }
            }

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

template <typename Traits>
MatmulWarptileTyped<Traits>::MatmulWarptileTyped(int N, int blockDim)
    : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
void MatmulWarptileTyped<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertWarpT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertWarpT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    dim3 threads(NUM_THREADS_WARP);
    dim3 blocks((N + BN_WARP - 1) / BN_WARP,
                (N + BM_WARP - 1) / BM_WARP);
    matmulWarptileTypedKernel<Traits><<<blocks, threads>>>(d_A16, d_B16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

template <typename Traits>
MatmulWarptileTyped<Traits>::~MatmulWarptileTyped() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiation: FP16 (the settled 16-bit format).
template class MatmulWarptileTyped<DTypeTraitsHalf>;
