#include "matmul_warptile_tuned.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

// Templatized warp-tiling kernel (rung 10, autotune space), dtype-templated.
// Mirrors archive-cudakernels matmulWarptileKernelT structure: fixed warp
// layout (4x8 threads), STM/STN subtiles per thread, As padded [BK][BM+1]
// against bank conflicts. Storage dtype via Traits; FP32 accumulation.

template <typename Traits>
__global__ void convertWT(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits, int BM, int BN, int BK, int TM, int TN, int WARP_M, int WARP_N>
__global__ void matmulWarptileTunedKernel(const typename Traits::T *A,
                                          const typename Traits::T *B,
                                          float *C, int N) {
    __shared__ typename Traits::T As[BK][BM + 1];  // +1 pad: bank conflicts
    __shared__ typename Traits::T Bs[BK][BN];

    constexpr int WARP_SIZE = 32;
    constexpr int WARP_THREAD_M = 4;
    constexpr int WARP_THREAD_N = 8;
    static_assert(WARP_THREAD_M * WARP_THREAD_N == WARP_SIZE);

    constexpr int WARPS_X = BN / WARP_N;
    constexpr int WARPS_Y = BM / WARP_M;
    constexpr int NUM_WARPS = WARPS_X * WARPS_Y;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

    constexpr int STM = WARP_M / (WARP_THREAD_M * TM);
    constexpr int STN = WARP_N / (WARP_THREAD_N * TN);
    static_assert(STM >= 1 && STN >= 1);

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

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        #pragma unroll
        for (int lo = 0; lo < BM; lo += strideA) {
            int row = innerRowA + lo;
            if (blockRow * BM + row < N && tileIdx + innerColA < N) {
                As[innerColA][row] = A[row * N + innerColA];
            } else {
                As[innerColA][row] = Traits::zero();
            }
        }
        #pragma unroll
        for (int lo = 0; lo < BK; lo += strideB) {
            int row = innerRowB + lo;
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
            for (int sm = 0; sm < STM; sm++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    int asRow = warpRow * WARP_M + sm * (WARP_M / STM) +
                                thrRow * TM + i;
                    regA[sm * TM + i] = Traits::to_float(As[dotIdx][asRow]);
                }
            }
            #pragma unroll
            for (int sn = 0; sn < STN; sn++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    int bsCol = warpCol * WARP_N + sn * (WARP_N / STN) +
                                thrCol * TN + j;
                    regB[sn * TN + j] = Traits::to_float(Bs[dotIdx][bsCol]);
                }
            }
            #pragma unroll
            for (int i = 0; i < STM * TM; i++) {
                #pragma unroll
                for (int j = 0; j < STN * TN; j++) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int sm = 0; sm < STM; sm++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int globalRow = blockRow * BM + warpRow * WARP_M +
                            sm * (WARP_M / STM) + thrRow * TM + i;
            if (globalRow < N) {
                #pragma unroll
                for (int sn = 0; sn < STN; sn++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        int globalCol = blockCol * BN + warpCol * WARP_N +
                                        sn * (WARP_N / STN) + thrCol * TN + j;
                        if (globalCol < N) {
                            C[(warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i) * N +
                              (warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j)] =
                                acc[sm * TM + i][sn * TN + j];
                        }
                    }
                }
            }
        }
    }
}

#include "matmul_warptile_configs.inc"

// Host-side dispatch (if-chain over generated branches).
template <typename Traits>
static bool launchWTuned(const typename Traits::T *A, const typename Traits::T *B,
                         float *C, int N, int BM, int BN, int BK, int TM, int TN, int WM, int WN) {
    int threads = (BM / WM) * (BN / WN) * 32;
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
#include "matmul_warptile_launch.inc"
    return false;
}

// ---------------------------------------------------------------------------
// Class implementation

template <typename Traits>
MatmulWarptileTuned<Traits>::MatmulWarptileTuned(int N, int blockDim, const CfgW &c)
    : N(N), blockDim(blockDim), cfg(c) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
MatmulWarptileTuned<Traits>::~MatmulWarptileTuned() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

template <typename Traits>
void MatmulWarptileTuned<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertWT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertWT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    launchWTuned<Traits>(d_A16, d_B16, d_C, N, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN, cfg.WM, cfg.WN);
}

template <typename Traits>
void MatmulWarptileTuned<Traits>::autotune(const float *d_A, const float *d_B, int N,
                                           int num_iterations) {
    {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertWT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertWT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    float *d_probe;
    cudaCheckError(cudaMalloc(&d_probe, (size_t)N * N * sizeof(float)));

    printf("# warptile autotune (%s storage, N=%d): %d configs\n",
           Traits::name, N, NUM_W_CONFIGS);

    double best = -1.0;
    for (int ci = 0; ci < NUM_W_CONFIGS; ci++) {
        const CfgWEnum &c = kConfigsW[ci];

        usleep(150000);  // per-config cool-down (power-state parity)

        bool ok = true;
        for (int w = 0; w < 10; w++)
            ok = launchWTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN) && ok;
        cudaDeviceSynchronize();
        cudaError_t lerr = cudaGetLastError();
        if (!ok || lerr != cudaSuccess) {
            printf("%d,%d,%d,%d,%d,%d,%d,%d,LAUNCH_FAIL(%s)\n", ci, c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN,
                   !ok ? "no dispatch" : cudaGetErrorString(lerr));
            fflush(stdout);
            continue;
        }

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));
        cudaCheckError(cudaEventRecord(start));
        for (int it = 0; it < num_iterations; it++)
            launchWTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(stop));

        double tflops = 2.0 * N * N * N * num_iterations / (ms * 1e-3) / 1e12;
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%.1f\n", ci, c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN, tflops * 1000);
        fflush(stdout);
        if (tflops > best) {
            best = tflops;
            cfg = CfgW{c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN};
        }
    }

    printf("# BEST %s BM=%d BN=%d BK=%d TM=%d TN=%d WM=%d WN=%d: %.1f GFLOPS (%.2f TFLOPS)\n",
           Traits::name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN, cfg.WM, cfg.WN, best * 1000, best);

    cudaFree(d_probe);
}

// Explicit instantiations: FP32 control + FP16.
template class MatmulWarptileTuned<DTypeTraitsFloat>;
template class MatmulWarptileTuned<DTypeTraitsHalf>;
