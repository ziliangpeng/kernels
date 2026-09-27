// single_warp16.cu — single-instance rebuild of the FP16 warptile autotune
// winner (128,128,16,8,4,WM=32,WN=64), plus FP32 control at the SAME config.
//
// Why: the 136-instance sweep TU carries a 26.6% multi-instance codegen tax
// (job 219310/219311). The FP16 sweep numbers (winner 32.39T) are therefore
// ~26% low in absolute terms. This binary instantiates exactly TWO kernels
// (FP16 winner + FP32 same-config control) — the tax at this instance count
// is negligible — and runs them ABAB x7 in one process for a clean,
// untaxed dtype pair.

#include "matmul_kernel.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <vector>
#include <algorithm>

template <typename Traits, int BM, int BN, int BK, int TM, int TN, int WARP_M, int WARP_N>
__global__ void matmulWarptileTunedKernel(const typename Traits::T * __restrict__ A,
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

template <typename Traits>
__global__ void convertSW(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

static double median(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static unsigned sm_clock() {
    nvmlDevice_t dev; unsigned clock = 0;
    if (nvmlInit() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS)
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clock);
    return clock;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 4096;
    int reps = argc > 2 ? atoi(argv[2]) : 7;
    cudaFree(0);

    float *h_A = (float *)malloc((size_t)N * N * sizeof(float));
    float *h_B = (float *)malloc((size_t)N * N * sizeof(float));
    for (long long i = 0; i < (long long)N * N; i++) {
        h_A[i] = ((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        h_B[i] = ((i * 2654435761 + 987654321) % 1000) / 1000.0f - 0.5f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)N * N * sizeof(float));
    cudaMalloc(&d_B, (size_t)N * N * sizeof(float));
    cudaMalloc(&d_C, (size_t)N * N * sizeof(float));
    cudaMemcpy(d_A, h_A, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice);

    // FP16 storage conversion (one-shot, outside timing)
    __half *d_A16, *d_B16;
    cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half));
    cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half));
    {
        int t = 256; long long b = ((long long)N * N + t - 1) / t;
        convertSW<DTypeTraitsHalf><<<(unsigned)b, t>>>(d_A, d_A16, N * N);
        convertSW<DTypeTraitsHalf><<<(unsigned)b, t>>>(d_B, d_B16, N * N);
        cudaDeviceSynchronize();
    }

    // Winner config (128,128,16,8,4,32,64): WARPS_X=2 * WARPS_Y=4 * 32 = 256 thr
    dim3 threads(256);
    dim3 grid((N + 127) / 128, (N + 127) / 128);
    using K16 = matmulWarptileTunedKernel<DTypeTraitsHalf, 128, 128, 16, 8, 4, 32, 64>;
    using K32 = matmulWarptileTunedKernel<DTypeTraitsFloat, 128, 128, 16, 8, 4, 32, 64>;

    // verify: FP16 vs FP32-reference (quantization diff expected, threshold 5e-3 rel)
    K32<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    std::vector<float> c32((size_t)N * N);
    cudaMemcpy(c32.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
    K16<<<grid, threads>>>(d_A16, d_B16, d_C, N);
    cudaDeviceSynchronize();
    std::vector<float> c16((size_t)N * N);
    cudaMemcpy(c16.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
    double maxrel = 0.0;
    for (size_t i = 0; i < c32.size(); i++) {
        double denom = std::max(1.0, (double)std::fabs(c32[i]));
        maxrel = std::max(maxrel, std::fabs(c16[i] - c32[i]) / denom);
    }
    printf("# verify f16-vs-f32ref maxrel = %.3e  %s\n", maxrel, maxrel < 5e-3 ? "PASS" : "FAIL");

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);
    auto timeit = [&](bool is16, int iters) {
        for (int w = 0; w < 10; w++) {
            if (is16) K16<<<grid, threads>>>(d_A16, d_B16, d_C, N);
            else K32<<<grid, threads>>>(d_A, d_B, d_C, N);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(es);
        for (int i = 0; i < iters; i++) {
            if (is16) K16<<<grid, threads>>>(d_A16, d_B16, d_C, N);
            else K32<<<grid, threads>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        float ms; cudaEventElapsedTime(&ms, es, ee);
        return ms / iters;
    };

    std::vector<double> r32(reps), r16(reps);
    printf("# rep,fp32_ms,fp16_ms,sm_clock\n");
    for (int r = 0; r < reps; r++) {
        r32[r] = timeit(false, 100);
        r16[r] = timeit(true, 100);
        printf("R,%d,%.4f,%.4f,%u\n", r, r32[r], r16[r], sm_clock());
        fflush(stdout);
    }
    double t32 = median(r32), t16 = median(r16);
    double f32 = 2.0 * N * N * N / t32 / 1e9, f16 = 2.0 * N * N * N / t16 / 1e9;
    printf("# SUMMARY single-instance winner config (128,128,16,8,4,32,64)\n");
    printf("# FP32 %.4f ms = %.2f TFLOPS   FP16 %.4f ms = %.2f TFLOPS   f16/f32 = %.4f\n",
           t32, f32, t16, f16, f16 / f32);
    return 0;
}
