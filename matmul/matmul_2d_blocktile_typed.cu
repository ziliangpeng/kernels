#include "matmul_2d_blocktile_typed.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>

// 2D block-tiling matmul with FP16 storage + FP32 accumulation (rung 5).
//
// Strict mirror of matmul_2d_blocktile.cu (BM=128 BN=128 BK=8 TM=TN=8,
// 256 threads, strided GMEM loads, outer-product compute with regA/regB
// register reuse). A/B tiles staged in SMEM as FP16; each element converted
// ONCE per dotIdx into regA[i]/regB[j] and reused TM*TN=64 times in the
// outer-product FMA nest — the cvt tax is amortized 64x better than naive.

#define BM_2DT 128
#define BN_2DT 128
#define BK_2DT 8
#define TM_2DT 8
#define TN_2DT 8

#define NUM_THREADS_2DT ((BM_2DT / TM_2DT) * (BN_2DT / TN_2DT))

template <typename Traits>
__global__ void convertFp32ToKernel2D(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
__global__ void matmul2DBlocktileTypedKernel(const typename Traits::T *A,
                                             const typename Traits::T *B,
                                             float *C, int N) {
    __shared__ typename Traits::T As[BM_2DT][BK_2DT];
    __shared__ typename Traits::T Bs[BK_2DT][BN_2DT];

    const int threadCol = threadIdx.x % (BN_2DT / TN_2DT);
    const int threadRow = threadIdx.x / (BN_2DT / TN_2DT);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_2DT * N;
    B += blockCol * BN_2DT;
    C += blockRow * BM_2DT * N + blockCol * BN_2DT;

    float threadResults[TM_2DT][TN_2DT] = {{0.0f}};
    float regA[TM_2DT];
    float regB[TN_2DT];

    const int strideA = NUM_THREADS_2DT / BK_2DT;
    const int strideB = NUM_THREADS_2DT / BN_2DT;

    const int innerRowA = threadIdx.x / BK_2DT;
    const int innerColA = threadIdx.x % BK_2DT;
    const int innerRowB = threadIdx.x / BN_2DT;
    const int innerColB = threadIdx.x % BN_2DT;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_2DT) {
        for (int loadOffset = 0; loadOffset < BM_2DT; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM_2DT + row < N && tileIdx + innerColA < N) {
                As[row][innerColA] = A[row * N + innerColA];
            } else {
                As[row][innerColA] = Traits::zero();
            }
        }

        for (int loadOffset = 0; loadOffset < BK_2DT; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN_2DT + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = Traits::zero();
            }
        }

        __syncthreads();

        A += BK_2DT;
        B += BK_2DT * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_2DT; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM_2DT; i++) {
                regA[i] = Traits::to_float(As[threadRow * TM_2DT + i][dotIdx]);
            }
            #pragma unroll
            for (int j = 0; j < TN_2DT; j++) {
                regB[j] = Traits::to_float(Bs[dotIdx][threadCol * TN_2DT + j]);
            }
            #pragma unroll
            for (int i = 0; i < TM_2DT; i++) {
                #pragma unroll
                for (int j = 0; j < TN_2DT; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM_2DT; i++) {
        #pragma unroll
        for (int j = 0; j < TN_2DT; j++) {
            int globalRow = blockRow * BM_2DT + threadRow * TM_2DT + i;
            int globalCol = blockCol * BN_2DT + threadCol * TN_2DT + j;
            if (globalRow < N && globalCol < N) {
                C[(threadRow * TM_2DT + i) * N + threadCol * TN_2DT + j] = threadResults[i][j];
            }
        }
    }
}

template <typename Traits>
Matmul2DBlocktileTyped<Traits>::Matmul2DBlocktileTyped(int N, int blockDim)
    : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
void Matmul2DBlocktileTyped<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertFp32ToKernel2D<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertFp32ToKernel2D<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    dim3 threads(NUM_THREADS_2DT);
    dim3 blocks((N + BN_2DT - 1) / BN_2DT,
                (N + BM_2DT - 1) / BM_2DT);

    matmul2DBlocktileTypedKernel<Traits><<<blocks, threads>>>(d_A16, d_B16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

template <typename Traits>
Matmul2DBlocktileTyped<Traits>::~Matmul2DBlocktileTyped() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiation: FP16 only (16-bit format settled 2026-09-26).
template class Matmul2DBlocktileTyped<DTypeTraitsHalf>;
