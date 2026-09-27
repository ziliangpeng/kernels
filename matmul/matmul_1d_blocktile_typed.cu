#include "matmul_1d_blocktile_typed.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>

// 1D block-tiling matmul with FP16 storage + FP32 accumulation
// (rung 4 of the multi-precision ladder, default config — no autotune).
//
// Identical structure to matmul_1d_blocktile.cu: BM=64 BN=64 BK=8 TM=8,
// 512 threads/block, each thread computes TM=8 column outputs, B element
// loaded from SMEM once and reused across TM FMAs. A/B tiles in SMEM are
// FP16 (GMEM->SMEM traffic halves); compute converts on SMEM read.

#define BM_1DT 64
#define BN_1DT 64
#define BK_1DT 8
#define TM_1DT 8

#define NUM_THREADS_1DT ((BM_1DT / TM_1DT) * BN_1DT)

template <typename Traits>
__global__ void convertFp32ToKernel1D(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
__global__ void matmul1DBlocktileTypedKernel(const typename Traits::T *A,
                                             const typename Traits::T *B,
                                             float *C, int N) {
    __shared__ typename Traits::T As[BM_1DT][BK_1DT];
    __shared__ typename Traits::T Bs[BK_1DT][BN_1DT];

    const int threadCol = threadIdx.x % BN_1DT;
    const int threadRow = threadIdx.x / BN_1DT;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_1DT * N;
    B += blockCol * BN_1DT;
    C += blockRow * BM_1DT * N + blockCol * BN_1DT;

    float threadResults[TM_1DT] = {0.0f};

    const int innerRowA = threadIdx.x / BK_1DT;
    const int innerColA = threadIdx.x % BK_1DT;
    const int innerRowB = threadIdx.x / BN_1DT;
    const int innerColB = threadIdx.x % BN_1DT;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_1DT) {
        if (blockRow * BM_1DT + innerRowA < N && tileIdx + innerColA < N) {
            As[innerRowA][innerColA] = A[innerRowA * N + innerColA];
        } else {
            As[innerRowA][innerColA] = Traits::zero();
        }

        if (tileIdx + innerRowB < N && blockCol * BN_1DT + innerColB < N) {
            Bs[innerRowB][innerColB] = B[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB][innerColB] = Traits::zero();
        }

        __syncthreads();

        A += BK_1DT;
        B += BK_1DT * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_1DT; dotIdx++) {
            float tmpB = Traits::to_float(Bs[dotIdx][threadCol]);
            #pragma unroll
            for (int resIdx = 0; resIdx < TM_1DT; resIdx++) {
                threadResults[resIdx] +=
                    Traits::to_float(As[threadRow * TM_1DT + resIdx][dotIdx]) * tmpB;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int resIdx = 0; resIdx < TM_1DT; resIdx++) {
        int globalRow = blockRow * BM_1DT + threadRow * TM_1DT + resIdx;
        int globalCol = blockCol * BN_1DT + threadCol;
        if (globalRow < N && globalCol < N) {
            C[(threadRow * TM_1DT + resIdx) * N + threadCol] = threadResults[resIdx];
        }
    }
}

template <typename Traits>
Matmul1DBlocktileTyped<Traits>::Matmul1DBlocktileTyped(int N, int blockDim)
    : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
void Matmul1DBlocktileTyped<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertFp32ToKernel1D<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertFp32ToKernel1D<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    dim3 threads(NUM_THREADS_1DT);
    dim3 blocks((N + BN_1DT - 1) / BN_1DT,
                (N + BM_1DT - 1) / BM_1DT);

    matmul1DBlocktileTypedKernel<Traits><<<blocks, threads>>>(d_A16, d_B16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

template <typename Traits>
Matmul1DBlocktileTyped<Traits>::~Matmul1DBlocktileTyped() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiation: FP16 only (16-bit format settled 2026-09-26).
template class Matmul1DBlocktileTyped<DTypeTraitsHalf>;
