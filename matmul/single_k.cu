// single_k.cu — TRUE single-instance builds, compile-time selected via -DKSEL.
//   KSEL=0: FP32 anchor  (128,128,16,16,4,64,32) — the 39.3T-vs-31.9T puzzle
//   KSEL=1: FP16 anchor  (128,128,16,16,4,64,32) — is 29.2T taxed or real?
//   KSEL=2: FP16 winner  (128,128,16,8,4,32,64)  — expect ~32.5 (stable so far)
// Each binary contains exactly ONE kernel instantiation. Verify vs CPU reference
// at N=1024 first (the 39.3T line has never been correctness-checked), then
// ABAB perf at N=4096.

#ifndef KSEL
#define KSEL 0
#endif

#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

template <typename Traits, int BM, int BN, int BK, int TM, int TN, int WARP_M, int WARP_N>
__global__ void warpK(const typename Traits::T * __restrict__ A,
                      const typename Traits::T * __restrict__ B,
                      float * __restrict__ C, int N) {
    __shared__ typename Traits::T As[BK][BM + 1];
    __shared__ typename Traits::T Bs[BK][BN];
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_THREAD_M = 4;
    constexpr int WARP_THREAD_N = 8;
    constexpr int WARPS_X = BN / WARP_N;
    constexpr int WARPS_Y = BM / WARP_M;
    constexpr int NUM_THREADS = WARPS_X * WARPS_Y * WARP_SIZE;
    constexpr int STM = WARP_M / (WARP_THREAD_M * TM);
    constexpr int STN = WARP_N / (WARP_THREAD_N * TN);
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpRow = warpId / WARPS_X;
    const int warpCol = warpId % WARPS_X;
    const int thrRow = laneId / WARP_THREAD_N;
    const int thrCol = laneId % WARP_THREAD_N;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;
    float acc[STM * TM][STN * TN];
    #pragma unroll
    for (int i = 0; i < STM * TM; i++)
        #pragma unroll
        for (int j = 0; j < STN * TN; j++)
            acc[i][j] = 0.0f;
    float regA[STM * TM];
    float regB[STN * TN];
    constexpr int strideA = NUM_THREADS / BK;
    constexpr int strideB = NUM_THREADS / BN;
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    for (int tileK = 0; tileK < N; tileK += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            int r = innerRowA + off;
            if (blockRow * BM + r < N && tileK + innerColA < N)
                As[innerColA][r] = A[r * N + innerColA];
            else
                As[innerColA][r] = Traits::zero();
        }
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            int r = innerRowB + off;
            if (tileK + r < N && blockCol * BN + innerColB < N)
                Bs[r][innerColB] = B[r * N + innerColB];
            else
                Bs[r][innerColB] = Traits::zero();
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int d = 0; d < BK; d++) {
            #pragma unroll
            for (int sm = 0; sm < STM; sm++)
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    regA[sm * TM + i] = Traits::to_float(As[d][warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i]);
            #pragma unroll
            for (int sn = 0; sn < STN; sn++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    regB[sn * TN + j] = Traits::to_float(Bs[d][warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j]);
            #pragma unroll
            for (int i = 0; i < STM * TM; i++)
                #pragma unroll
                for (int j = 0; j < STN * TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int sm = 0; sm < STM; sm++)
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int lr = warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i;
            int gr = blockRow * BM + lr;
            if (gr < N) {
                #pragma unroll
                for (int sn = 0; sn < STN; sn++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        int lc = warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j;
                        int gc = blockCol * BN + lc;
                        if (gc < N)
                            C[lr * N + lc] = acc[sm * TM + i][sn * TN + j];
                    }
            }
        }
}

#if KSEL == 0
  #define KNAME "fp32-anchor(128,128,16,16,4,64,32)"
  using Tr = DTypeTraitsFloat;
  #define KBM 128
  #define KBN 128
  #define KBK 16
  #define KTM 16
  #define KTN 4
  #define KWM 64
  #define KWN 32
  #define VTHRESH 1e-4
#elif KSEL == 1
  #define KNAME "fp16-anchor(128,128,16,16,4,64,32)"
  using Tr = DTypeTraitsHalf;
  #define KBM 128
  #define KBN 128
  #define KBK 16
  #define KTM 16
  #define KTN 4
  #define KWM 64
  #define KWN 32
  #define VTHRESH 5e-3
#else
  #define KNAME "fp16-winner(128,128,16,8,4,32,64)"
  using Tr = DTypeTraitsHalf;
  #define KBM 128
  #define KBN 128
  #define KBK 16
  #define KTM 8
  #define KTN 4
  #define KWM 32
  #define KWN 64
  #define VTHRESH 5e-3
#endif

static double median(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv) {
    int Nperf = argc > 1 ? atoi(argv[1]) : 4096;
    cudaFree(0);

    // ---------- verify vs CPU at N=1024 ----------
    const int Nv = 1024;
    std::vector<float> hA((size_t)Nv * Nv), hB((size_t)Nv * Nv), ref((size_t)Nv * Nv);
    for (int i = 0; i < Nv * Nv; i++) {
        hA[i] = ((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        hB[i] = ((i * 2654435761 + 987654321) % 1000) / 1000.0f - 0.5f;
    }
    for (int i = 0; i < Nv; i++)
        for (int k = 0; k < Nv; k++) {
            float a = hA[i * Nv + k];
            for (int j = 0; j < Nv; j++)
                ref[i * Nv + j] += a * hB[k * Nv + j];
        }
    float *dA, *dB, *dC, *dA16, *dB16;
    cudaMalloc(&dA, (size_t)Nv * Nv * sizeof(float));
    cudaMalloc(&dB, (size_t)Nv * Nv * sizeof(float));
    cudaMalloc(&dC, (size_t)Nv * Nv * sizeof(float));
    cudaMemcpy(dA, hA.data(), (size_t)Nv * Nv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), (size_t)Nv * Nv * sizeof(float), cudaMemcpyHostToDevice);
    dA16 = dA; dB16 = dB;  // FP32 traits use the same buffers
    if (!std::is_same<Tr, DTypeTraitsFloat>::value) {
        cudaMalloc(&dA16, (size_t)Nv * Nv * sizeof(Tr::T));
        cudaMalloc(&dB16, (size_t)Nv * Nv * sizeof(Tr::T));
        // convert on host (simple)
        std::vector<Tr::T> h16((size_t)Nv * Nv);
        for (int i = 0; i < Nv * Nv; i++) { h16[i] = Tr::from_float(hA[i]); }
        cudaMemcpy(dA16, h16.data(), (size_t)Nv * Nv * sizeof(Tr::T), cudaMemcpyHostToDevice);
        for (int i = 0; i < Nv * Nv; i++) { h16[i] = Tr::from_float(hB[i]); }
        cudaMemcpy(dB16, h16.data(), (size_t)Nv * Nv * sizeof(Tr::T), cudaMemcpyHostToDevice);
    }
    {
        dim3 threads(256);
        dim3 grid((Nv + 127) / 128, (Nv + 127) / 128);
        warpK<Tr, KBM, KBN, KBK, KTM, KTN, KWM, KWN><<<grid, threads>>>(dA16, dB16, dC, Nv);
        cudaDeviceSynchronize();
        cudaCheckError(cudaGetLastError());
    }
    std::vector<float> c((size_t)Nv * Nv);
    cudaMemcpy(c.data(), dC, (size_t)Nv * Nv * sizeof(float), cudaMemcpyDeviceToHost);
    double maxrel = 0.0;
    for (size_t i = 0; i < c.size(); i++) {
        double denom = std::max(1.0, (double)std::fabs(ref[i]));
        maxrel = std::max(maxrel, std::fabs(c[i] - ref[i]) / denom);
    }
    printf("# KSEL=%d %s  verify-vs-CPU maxrel=%.3e  %s\n", KSEL, KNAME, maxrel,
           maxrel < VTHRESH ? "PASS" : "FAIL");

    // ---------- perf at Nperf ----------
    size_t bytes = (size_t)Nperf * Nperf;
    float *pA, *pB, *pC, *pA16, *pB16;
    cudaMalloc(&pA, bytes * sizeof(float));
    cudaMalloc(&pB, bytes * sizeof(float));
    cudaMalloc(&pC, bytes * sizeof(float));
    cudaMemcpy(pA, hA.data(), (size_t)Nv * Nv * sizeof(float), cudaMemcpyHostToDevice);  // placeholder copy (values irrelevant for perf)
    cudaDeviceSynchronize();
    pA16 = pA; pB16 = pB;
    if (!std::is_same<Tr, DTypeTraitsFloat>::value) {
        cudaMalloc(&pA16, bytes * sizeof(Tr::T));
        cudaMalloc(&pB16, bytes * sizeof(Tr::T));
        // device-side fill: launch the kernel pattern on dummy data; values don't matter
        // simple: reuse verify buffers content? Just cudaMemset to a pattern then run.
        cudaMemset(pA16, 0x3c, bytes * sizeof(Tr::T));
        cudaMemset(pB16, 0x3c, bytes * sizeof(Tr::T));
    }
    dim3 threads(256);
    dim3 grid((Nperf + 127) / 128, (Nperf + 127) / 128);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int w = 0; w < 10; w++)
        warpK<Tr, KBM, KBN, KBK, KTM, KTN, KWM, KWN><<<grid, threads>>>(pA16, pB16, pC, Nperf);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int it = 0; it < 100; it++)
        warpK<Tr, KBM, KBN, KBK, KTM, KTN, KWM, KWN><<<grid, threads>>>(pA16, pB16, pC, Nperf);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    printf("# KSEL=%d %s  perf N=%d: %.4f ms/iter = %.2f TFLOPS\n", KSEL, KNAME, Nperf,
           ms / 100.0f, 2.0 * Nperf * Nperf * Nperf / (ms / 100.0f) / 1e9);
    return 0;
}
