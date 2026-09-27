#include "matmul_2d_blocktile_tuned.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

// Templatized 2D block-tiling kernel (rung 5, autotune space), FP16 storage
// + FP32 accumulation. Config space mirrors archive-cudakernels
// Matmul2DBlocktileAuto (19 candidates; FP32 winner was (128,128,16,16,8) at
// 33.8T on H100). ABAB lesson applied: autotune finds the config; production
// numbers come from a hard-coded build of the winner.

template <typename Traits>
__global__ void convert2DT(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul2DTunedKernel(const typename Traits::T *A,
                                    const typename Traits::T *B,
                                    float *C, int N) {
    __shared__ typename Traits::T As[BM][BK];
    __shared__ typename Traits::T Bs[BK][BN];

    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr int strideA = NUM_THREADS / BK;
    constexpr int strideB = NUM_THREADS / BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    float threadResults[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM + row < N && tileIdx + innerColA < N) {
                As[row][innerColA] = A[row * N + innerColA];
            } else {
                As[row][innerColA] = Traits::zero();
            }
        }
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = Traits::zero();
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = Traits::to_float(As[threadRow * TM + i][dotIdx]);
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = Traits::to_float(Bs[dotIdx][threadCol * TN + j]);
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalRow = blockRow * BM + threadRow * TM + i;
            int globalCol = blockCol * BN + threadCol * TN + j;
            if (globalRow < N && globalCol < N) {
                C[(threadRow * TM + i) * N + threadCol * TN + j] = threadResults[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config space: 19 candidates mirroring archive-cudakernels CANDIDATES_2D.
struct Cfg2DEnum { int BM, BN, BK, TM, TN; };
static const Cfg2DEnum kConfigs2D[] = {
    {128, 128,  8,  8,  8},   // [ 0] typed-kernel default — 256 thr
    { 64,  64,  8,  8,  8},   // [ 1] smaller block
    {128, 128,  8, 16,  8},   // [ 2] bigger TM
    {128, 128,  8,  8, 16},   // [ 3] bigger TN
    {128, 128, 16,  8,  8},   // [ 4] deeper BK
    {128,  64,  8,  8,  8},   // [ 5] tall
    { 64, 128,  8,  8,  8},   // [ 6] wide
    {128, 128,  8,  4,  4},   // [ 7] small thread tile — 1024 thr
    {256, 128,  8, 16,  8},   // [ 8] bigger block (tall)
    {128, 256,  8,  8, 16},   // [ 9] bigger block (wide)
    {128, 128, 16, 16,  8},   // [10] FP32 historical winner (33.8T)
    {256, 128, 16, 16,  8},   // [11] push #10 taller
    {128, 128, 32, 16,  8},   // [12] BK=32
    {128, 128, 16,  8, 16},   // [13] TM<->TN swap of #10
    {256, 256, 16, 16,  8},   // [14] big square
    {256, 128, 24, 16,  8},   // [15] BK=24 tall
    {256, 128, 32, 16,  8},   // [16] BK=32 tall
    {128, 128, 24, 16,  8},   // [17] BK=24
    {256, 256,  8, 16,  8},   // [18] big square, shallow BK
};
static const int NUM_2D_CONFIGS = sizeof(kConfigs2D) / sizeof(kConfigs2D[0]);

// Host-side dispatch (if-chain over explicit instantiations).
template <typename Traits>
static bool launch2DTuned(const typename Traits::T *A, const typename Traits::T *B,
                          float *C, int N, int BM, int BN, int BK, int TM, int TN) {
    int threads = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define L2D(_BM, _BN, _BK, _TM, _TN) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) { \
            matmul2DTunedKernel<Traits, _BM, _BN, _BK, _TM, _TN><<<grid, threads>>>(A, B, C, N); \
            return true; \
        }
    L2D(128, 128,  8,  8,  8)
    L2D( 64,  64,  8,  8,  8)
    L2D(128, 128,  8, 16,  8)
    L2D(128, 128,  8,  8, 16)
    L2D(128, 128, 16,  8,  8)
    L2D(128,  64,  8,  8,  8)
    L2D( 64, 128,  8,  8,  8)
    L2D(128, 128,  8,  4,  4)
    L2D(256, 128,  8, 16,  8)
    L2D(128, 256,  8,  8, 16)
    L2D(128, 128, 16, 16,  8)
    L2D(256, 128, 16, 16,  8)
    L2D(128, 128, 32, 16,  8)
    L2D(128, 128, 16,  8, 16)
    L2D(256, 256, 16, 16,  8)
    L2D(256, 128, 24, 16,  8)
    L2D(256, 128, 32, 16,  8)
    L2D(128, 128, 24, 16,  8)
    L2D(256, 256,  8, 16,  8)
    #undef L2D
    return false;
}

// ---------------------------------------------------------------------------
// Class implementation

template <typename Traits>
Matmul2DBlocktileTuned<Traits>::Matmul2DBlocktileTuned(int N, int blockDim, const Cfg2D &c)
    : N(N), blockDim(blockDim), cfg(c) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
Matmul2DBlocktileTuned<Traits>::~Matmul2DBlocktileTuned() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

template <typename Traits>
void Matmul2DBlocktileTuned<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convert2DT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convert2DT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    launch2DTuned<Traits>(d_A16, d_B16, d_C, N, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN);
}

template <typename Traits>
void Matmul2DBlocktileTuned<Traits>::autotune(const float *d_A, const float *d_B, int N,
                                              int num_iterations) {
    // Convert inputs once for all configs.
    {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convert2DT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convert2DT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    float *d_probe;
    cudaCheckError(cudaMalloc(&d_probe, (size_t)N * N * sizeof(float)));

    printf("# 2D blocktile autotune (%s storage, N=%d): %d configs\n",
           Traits::name, N, NUM_2D_CONFIGS);

    double best = -1.0;
    for (int ci = 0; ci < NUM_2D_CONFIGS; ci++) {
        const Cfg2DEnum &c = kConfigs2D[ci];

        usleep(150000);  // per-config cool-down (power-state parity, 2026-09-26 lesson)

        bool ok = true;
        for (int w = 0; w < 10; w++)
            ok = launch2DTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN) && ok;
        if (!ok) {
            printf("%d,%d,%d,%d,%d,%d,LAUNCH_FAIL\n", ci, c.BM, c.BN, c.BK, c.TM, c.TN);
            fflush(stdout);
            continue;
        }
        cudaCheckError(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));
        cudaCheckError(cudaEventRecord(start));
        for (int it = 0; it < num_iterations; it++)
            launch2DTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(stop));

        double tflops = 2.0 * N * N * N * num_iterations / (ms * 1e-3) / 1e12;
        printf("%d,%d,%d,%d,%d,%d,%.1f\n", ci, c.BM, c.BN, c.BK, c.TM, c.TN, tflops * 1000);
        fflush(stdout);
        if (tflops > best) {
            best = tflops;
            cfg = Cfg2D{c.BM, c.BN, c.BK, c.TM, c.TN};
        }
    }

    printf("# BEST %s BM=%d BN=%d BK=%d TM=%d TN=%d: %.1f GFLOPS (%.2f TFLOPS)\n",
           Traits::name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN, best * 1000, best);

    cudaFree(d_probe);
}

// Explicit instantiations: FP32 control + FP16.
template class Matmul2DBlocktileTuned<DTypeTraitsFloat>;
template class Matmul2DBlocktileTuned<DTypeTraitsHalf>;
