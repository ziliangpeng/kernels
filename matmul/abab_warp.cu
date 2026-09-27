// abab_warp.cu — same-process A/B: archive warptile kernel vs our tuned template.
//
// A = archive matmulWarptileKernelT<128,128,16,16,4,64,32> (the kernel inside
//     the ported MatmulWarptileAuto; measured 39.07T on node 19)
// B = our MatmulWarptileTuned template at the same config (measured 31.09T
//     in-sweep on node 54)
//
// Same process, same GPU, alternating ABAB x 7, batched + single CUDA-event
// timing, SM clock sampled. Settles whether the 26% gap is codegen (A >> B
// everywhere) or node/sweep methodology (A == B).

#include "matmul_kernel.h"
#include "matmul_warptile_tuned.h"
#include "dtype_traits.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>

// ---------------------------------------------------------------------------
// A: the archive kernel, copied VERBATIM from archive-cudakernels
// matmul_warptile.cu matmulWarptileKernelT (float, no Traits).
template<int BM, int BN, int BK, int TM, int TN, int WARP_M, int WARP_N>
__global__ void archiveWarptileKernel(const float * __restrict__ A,
                                       const float * __restrict__ B,
                                       float *C, int N) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];

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
                As[innerColA][r] = 0.0f;
        }
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            int r = innerRowB + off;
            if (tileK + r < N && blockCol * BN + innerColB < N)
                Bs[r][innerColB] = B[r * N + innerColB];
            else
                Bs[r][innerColB] = 0.0f;
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
                    regA[sm * TM + i] = As[d][warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i];
            #pragma unroll
            for (int sn = 0; sn < STN; sn++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    regB[sn * TN + j] = Bs[d][warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j];
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

// ---------------------------------------------------------------------------
// B-side launcher: reuse the tuned template's dispatch (via a direct call to
// the tuned class with the config installed) — but for a pure kernel A/B we
// launch the tuned kernel directly through the same hard-coded path the sweep
// uses. To keep the comparison purely about KERNELS (not dispatch), we wrap
// both in MatmulKernel adapters that each launch exactly one kernel.

struct AK : MatmulKernel {
    int N;
    AK(int N_) : N(N_) {}
    void execute(const float *d_A, const float *d_B, float *d_C) override {
        dim3 threads(128);  // (128/64)*(128/32)*32 = 4 warps * 32
        dim3 grid((N + 127) / 128, (N + 127) / 128);
        archiveWarptileKernel<128, 128, 16, 16, 4, 64, 32><<<grid, threads>>>(d_A, d_B, d_C, N);
        cudaCheckError(cudaGetLastError());
    }
};

struct BK_ : MatmulKernel {
    int N;
    __half *dA = nullptr, *dB = nullptr;  // FP32 traits storage (float), reuse buffers
    BK_(int N_) : N(N_) {}
    void execute(const float *d_A, const float *d_B, float *d_C) override {
        // FP32 control of the tuned template: launch with float buffers directly.
        // The tuned kernel is exported via launchWTuned which needs Traits::T*.
        // For the FP32 control the "converted" buffers ARE float copies; to keep
        // this A/B pure we allocate float copies once and pass them.
        if (!dA) {
            cudaMalloc(&dA, (size_t)N * N * sizeof(float));
            cudaMalloc(&dB, (size_t)N * N * sizeof(float));
            cudaMemcpy(dA, d_A, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dB, d_B, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        }
        extern void launchWTunedF32Direct(const float *A, const float *B, float *C, int N);
        launchWTunedF32Direct((const float *)dA, (const float *)dB, d_C, N);
    }
};

static double median(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double iqr(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return v[(3 * n) / 4] - v[n / 4];
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

    MatmulKernel *A = new AK(N);
    MatmulKernel *B = new BK_(N);

    // verify identical
    A->execute(d_A, d_B, d_C); cudaDeviceSynchronize();
    std::vector<float> cA((size_t)N * N);
    cudaMemcpy(cA.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
    B->execute(d_A, d_B, d_C); cudaDeviceSynchronize();
    std::vector<float> cB((size_t)N * N);
    cudaMemcpy(cB.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
    double maxdiff = 0.0;
    for (size_t i = 0; i < cA.size(); i++) maxdiff = std::max(maxdiff, (double)std::fabs(cA[i] - cB[i]));
    printf("# verify A-vs-B maxdiff = %.3e  %s\n", maxdiff, maxdiff < 1e-6 ? "IDENTICAL" : "DIFFER");

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);
    auto time_batched = [&](MatmulKernel *k, int iters) {
        for (int w = 0; w < 10; w++) k->execute(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(es);
        for (int i = 0; i < iters; i++) k->execute(d_A, d_B, d_C);
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        float ms; cudaEventElapsedTime(&ms, es, ee);
        return ms / iters;
    };

    std::vector<double> rA(reps), rB(reps);
    printf("# rep,A_ms,B_ms,sm_clock\n");
    for (int r = 0; r < reps; r++) {
        rA[r] = time_batched(A, 100);
        rB[r] = time_batched(B, 100);
        printf("R,%d,%.4f,%.4f,%u\n", r, rA[r], rB[r], sm_clock());
        fflush(stdout);
    }
    double a = median(rA), b = median(rB);
    printf("# SUMMARY A archive-kernel %.4f ms (%.2f TFLOPS)  B tuned-template %.4f ms (%.2f TFLOPS)  B/A=%.4f\n",
           a, 2.0 * N * N * N / a / 1e9, b, 2.0 * N * N * N / b / 1e9, b / a);
    printf("# IQR A %.5f  B %.5f\n", iqr(rA), iqr(rB));
    return 0;
}
