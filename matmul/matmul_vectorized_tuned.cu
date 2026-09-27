#include "matmul_vectorized_tuned.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

// Templatized vectorized kernel (rung 6, autotune space), dtype-templated
// 16-byte vector loads. Config space mirrors archive-cudakernels
// CANDIDATES_VEC (16 candidates; FP32 winner (128,128,8,8,16) at 34.9T).
//
// FP32 note: storage is a CONVERTED FP32 buffer (memcpy-identical values)
// rather than the caller's original float* — this keeps both dtypes on the
// exact same code path (same pointer types, same launch), matching the
// hard-coded FP32 baseline semantics bit-for-bit while sharing one template.

template <typename Traits>
__global__ void convertVecT(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits, int BM, int BN, int BK, int TM, int TN>
__global__ void matmulVecTunedKernel(const typename Traits::T *A,
                                     const typename Traits::T *B,
                                     float *C, int N) {
    constexpr int VE = 16 / sizeof(typename Traits::T);  // 4 (fp32) or 8 (fp16)
    using VecT = typename Traits::VecT;
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    __shared__ typename Traits::T As[BK][BM];  // transposed [k][m]
    __shared__ typename Traits::T Bs[BK][BN];

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

    // Vector-load mapping: A tile is BM*BK elems, VE elems per load.
    // Loads per tile = BM*BK/VE; threads iterate in strides of NUM_THREADS.
    // A row holds BK/VE vectors; row stride (in vectors) = BK/VE.
    const int aVecRow = threadIdx.x / (BK / VE);         // row this thread starts at
    const int aVecCol = threadIdx.x % (BK / VE);         // vector column
    const int aRowStride = NUM_THREADS / (BK / VE);      // row advance per stride step

    // B tile is BK*BN elems; row holds BN/VE vectors.
    const int bVecRow = threadIdx.x / (BN / VE);
    const int bVecCol = threadIdx.x % (BN / VE);
    const int bRowStride = NUM_THREADS / (BN / VE);

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        // --- load A tile (vectorized), store transposed As[k][m] ---
        #pragma unroll
        for (int rowOff = 0; rowOff < BM; rowOff += aRowStride) {
            const int row = aVecRow + rowOff;
            if (row < BM && blockRow * BM + row < N) {
                VecT tmp;
                Traits::zero_vec(&tmp);
                if (tileIdx + aVecCol * VE + (VE - 1) < N) {
                    tmp = *reinterpret_cast<const VecT *>(&A[row * N + aVecCol * VE]);
                } else {
                    for (int i = 0; i < VE; i++) {
                        if (tileIdx + aVecCol * VE + i < N)
                            Traits::set_vec_elem(&tmp, i, A[row * N + aVecCol * VE + i]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < VE; i++) {
                    As[aVecCol * VE + i][row] = Traits::get_vec_elem(tmp, i);
                }
            }
        }

        // --- load B tile (vectorized) ---
        #pragma unroll
        for (int rowOff = 0; rowOff < BK; rowOff += bRowStride) {
            const int row = bVecRow + rowOff;
            if (row < BK && tileIdx + row < N) {
                VecT tmp;
                Traits::zero_vec(&tmp);
                if (blockCol * BN + bVecCol * VE + (VE - 1) < N) {
                    tmp = *reinterpret_cast<const VecT *>(&B[row * N + bVecCol * VE]);
                } else {
                    for (int i = 0; i < VE; i++) {
                        if (blockCol * BN + bVecCol * VE + i < N)
                            Traits::set_vec_elem(&tmp, i, B[row * N + bVecCol * VE + i]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < VE; i++) {
                    Bs[row][bVecCol * VE + i] = Traits::get_vec_elem(tmp, i);
                }
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // --- compute (outer product; As transposed -> coalesced SMEM reads) ---
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = Traits::to_float(As[dotIdx][threadRow * TM + i]);
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

    // --- write C (float4; C stays FP32) ---
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalRow = blockRow * BM + threadRow * TM + i;
        if (globalRow < N) {
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int globalCol = blockCol * BN + threadCol * TN + j;
                if (globalCol + 3 < N) {
                    float4 tmp;
                    tmp.x = threadResults[i][j + 0];
                    tmp.y = threadResults[i][j + 1];
                    tmp.z = threadResults[i][j + 2];
                    tmp.w = threadResults[i][j + 3];
                    *reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j]) = tmp;
                } else {
                    for (int k = 0; k < 4 && globalCol + k < N; k++) {
                        C[(threadRow * TM + i) * N + threadCol * TN + j + k] = threadResults[i][j + k];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config space: 16 candidates mirroring archive-cudakernels CANDIDATES_VEC.
struct CfgVecEnum { int BM, BN, BK, TM, TN; };
static const CfgVecEnum kConfigsVec[] = {
    {128, 128,  8,  8,  8},   // [ 0] typed-kernel default
    {128, 128,  8, 16,  8},   // [ 1] bigger TM
    {128, 128,  8,  8, 16},   // [ 2] bigger TN (archive FP32 winner 34.9T)
    {128, 128, 16,  8,  8},   // [ 3] deeper BK
    {128, 128, 16, 16,  8},   // [ 4] BK=16 + big TM (2D winner)
    {128, 128, 16,  8, 16},   // [ 5] BK=16 + big TN
    {128,  64,  8, 16,  8},   // [ 6] tall
    { 64, 128,  8,  8,  8},   // [ 7] wide
    {256, 128,  8, 16,  8},   // [ 8] bigger block (tall)
    {128, 256,  8,  8, 16},   // [ 9] bigger block (wide)
    { 64,  64,  8,  8,  8},   // [10] smaller block
    { 64,  64, 16,  8,  8},   // [11] smaller + deeper BK
    {128, 128, 32, 16,  8},   // [12] deeper BK on winner
    {256, 128, 16, 16,  8},   // [13] big tall + BK=16
    {256, 256,  8, 16,  8},   // [14] big square (may spill)
    {128, 128, 16,  4,  4},   // [15] small thread tile — 1024 thr
};
static const int NUM_VEC_CONFIGS = sizeof(kConfigsVec) / sizeof(kConfigsVec[0]);

// Host-side dispatch.
template <typename Traits>
static bool launchVecTuned(const typename Traits::T *A, const typename Traits::T *B,
                           float *C, int N, int BM, int BN, int BK, int TM, int TN) {
    int threads = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define LVEC(_BM, _BN, _BK, _TM, _TN) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) { \
            matmulVecTunedKernel<Traits, _BM, _BN, _BK, _TM, _TN><<<grid, threads>>>(A, B, C, N); \
            return true; \
        }
    LVEC(128, 128,  8,  8,  8)
    LVEC(128, 128,  8, 16,  8)
    LVEC(128, 128,  8,  8, 16)
    LVEC(128, 128, 16,  8,  8)
    LVEC(128, 128, 16, 16,  8)
    LVEC(128, 128, 16,  8, 16)
    LVEC(128,  64,  8, 16,  8)
    LVEC( 64, 128,  8,  8,  8)
    LVEC(256, 128,  8, 16,  8)
    LVEC(128, 256,  8,  8, 16)
    LVEC( 64,  64,  8,  8,  8)
    LVEC( 64,  64, 16,  8,  8)
    LVEC(128, 128, 32, 16,  8)
    LVEC(256, 128, 16, 16,  8)
    LVEC(256, 256,  8, 16,  8)
    LVEC(128, 128, 16,  4,  4)
    #undef LVEC
    return false;
}

// ---------------------------------------------------------------------------
// Class implementation

template <typename Traits>
MatmulVectorizedTuned<Traits>::MatmulVectorizedTuned(int N, int blockDim, const CfgVec &c)
    : N(N), blockDim(blockDim), cfg(c) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
MatmulVectorizedTuned<Traits>::~MatmulVectorizedTuned() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

template <typename Traits>
void MatmulVectorizedTuned<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertVecT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertVecT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    launchVecTuned<Traits>(d_A16, d_B16, d_C, N, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN);
}

template <typename Traits>
void MatmulVectorizedTuned<Traits>::autotune(const float *d_A, const float *d_B, int N,
                                             int num_iterations) {
    {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertVecT<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertVecT<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    float *d_probe;
    cudaCheckError(cudaMalloc(&d_probe, (size_t)N * N * sizeof(float)));

    printf("# vectorized autotune (%s storage, N=%d): %d configs\n",
           Traits::name, N, NUM_VEC_CONFIGS);

    double best = -1.0;
    for (int ci = 0; ci < NUM_VEC_CONFIGS; ci++) {
        const CfgVecEnum &c = kConfigsVec[ci];

        usleep(150000);  // per-config cool-down (power-state parity)

        bool ok = true;
        for (int w = 0; w < 10; w++)
            ok = launchVecTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN) && ok;
        cudaDeviceSynchronize();
        cudaError_t lerr = cudaGetLastError();
        if (!ok || lerr != cudaSuccess) {
            printf("%d,%d,%d,%d,%d,%d,LAUNCH_FAIL(%s)\n", ci, c.BM, c.BN, c.BK, c.TM, c.TN,
                   !ok ? "no dispatch" : cudaGetErrorString(lerr));
            fflush(stdout);
            continue;
        }

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));
        cudaCheckError(cudaEventRecord(start));
        for (int it = 0; it < num_iterations; it++)
            launchVecTuned<Traits>(d_A16, d_B16, d_probe, N, c.BM, c.BN, c.BK, c.TM, c.TN);
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
            cfg = CfgVec{c.BM, c.BN, c.BK, c.TM, c.TN};
        }
    }

    printf("# BEST %s BM=%d BN=%d BK=%d TM=%d TN=%d: %.1f GFLOPS (%.2f TFLOPS)\n",
           Traits::name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN, best * 1000, best);

    cudaFree(d_probe);
}

// Explicit instantiations: FP32 control + FP16.
template class MatmulVectorizedTuned<DTypeTraitsFloat>;
template class MatmulVectorizedTuned<DTypeTraitsHalf>;
