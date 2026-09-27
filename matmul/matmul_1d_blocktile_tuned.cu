#include "matmul_1d_blocktile_tuned.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// Templatized 1D block-tiling kernel (rung 4, autotune space).
//
// Same algorithm as matmul_1d_blocktile.cu but with BM/BN/BK/TM as template
// parameters, and dtype-templated storage via Traits (FP32 control + FP16).
// 1D constraint carried over from the original: BM = BN = BK * TM (one
// thread column spans the full BK tile width), threads = BN * BM / TM.

template <typename Traits>
__global__ void convert1D(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits, int BM, int BN, int BK, int TM>
__global__ void matmul1DTunedKernel(const typename Traits::T *A,
                                    const typename Traits::T *B,
                                    float *C, int N) {
    __shared__ typename Traits::T As[BM][BK];
    __shared__ typename Traits::T Bs[BK][BN];

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float threadResults[TM] = {0.0f};

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        if (blockIdx.y * BM + innerRowA < N && tileIdx + innerColA < N) {
            As[innerRowA][innerColA] = A[innerRowA * N + innerColA];
        } else {
            As[innerRowA][innerColA] = Traits::zero();
        }
        if (tileIdx + innerRowB < N && blockIdx.x * BN + innerColB < N) {
            Bs[innerRowB][innerColB] = B[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB][innerColB] = Traits::zero();
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float tmpB = Traits::to_float(Bs[dotIdx][threadCol]);
            #pragma unroll
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                threadResults[resIdx] +=
                    Traits::to_float(As[threadRow * TM + resIdx][dotIdx]) * tmpB;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int resIdx = 0; resIdx < TM; resIdx++) {
        int globalRow = blockIdx.y * BM + threadRow * TM + resIdx;
        int globalCol = blockIdx.x * BN + threadCol;
        if (globalRow < N && globalCol < N) {
            C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
        }
    }
}

#include "matmul_1d_configs.inc"

// ---------------------------------------------------------------------------
// Host-side dispatch

template <typename Traits>
static bool launch1DTuned(const typename Traits::T *A, const typename Traits::T *B,
                          float *C, int N, int BM, int BN, int BK, int TM) {
#include "matmul_1d_launch.inc"
    return false;  // unsupported config
}

template <typename Traits>
void Matmul1DBlocktileTuned<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convert1D<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convert1D<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    launch1DTuned<Traits>(d_A16, d_B16, d_C, N, cfg.BM, cfg.BN, cfg.BK, cfg.TM);
}

template <typename Traits>
void Matmul1DBlocktileTuned<Traits>::autotune(const float *d_A, const float *d_B, int N,
                                              int num_iterations) {
    // Convert inputs once for all configs.
    {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convert1D<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convert1D<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    float *d_C;
    cudaCheckError(cudaMalloc(&d_C, (size_t)N * N * sizeof(float)));

    float *d_probe = nullptr;
    cudaCheckError(cudaMalloc(&d_probe, (size_t)N * N * sizeof(float)));

    printf("# 1D blocktile autotune (%s storage, N=%d): %d configs\n",
           Traits::name, N, NUM_1D_CONFIGS);

    double best = -1.0;
    for (int ci = 0; ci < NUM_1D_CONFIGS; ci++) {
        const Cfg1D &c = kConfigs1D[ci];

        // warmup
        for (int w = 0; w < 3; w++)
            launch1DTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM);
        cudaCheckError(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));
        cudaCheckError(cudaEventRecord(start));
        for (int it = 0; it < num_iterations; it++)
            launch1DTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(stop));

        double tflops = 2.0 * N * N * N * num_iterations / (ms * 1e-3) / 1e12;
        printf("%d,%d,%d,%d,%d,%.1f\n", ci, c.BM, c.BN, c.BK, c.TM, tflops * 1000);
        fflush(stdout);
        if (tflops > best) {
            best = tflops;
            cfg = c;
        }
    }

    printf("# BEST %s BM=%d BN=%d BK=%d TM=%d: %.1f GFLOPS (%.2f TFLOPS)\n",
           Traits::name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, best * 1000, best);

    cudaFree(d_C);
    cudaFree(d_probe);
}

template <typename Traits>
Matmul1DBlocktileTuned<Traits>::Matmul1DBlocktileTuned(int N, int blockDim, const Cfg1D &c)
    : N(N), blockDim(blockDim), cfg(c) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
Matmul1DBlocktileTuned<Traits>::~Matmul1DBlocktileTuned() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiations: FP32 control + FP16.
template class Matmul1DBlocktileTuned<DTypeTraitsFloat>;
template class Matmul1DBlocktileTuned<DTypeTraitsHalf>;
