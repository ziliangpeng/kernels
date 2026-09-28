// Warp Tiling + cp.async double buffering — FP16 storage / FP32 accumulation.
// Port of matmul_warptile_dbuf.cu (Simon rung 12) to 16-bit with two layout
// changes required by cp.async's 4/8/16B granularity (see header comment).
//
// Pipeline mechanics identical to the FP32 version: cp.async 4B (A, k-paired
// halves) + 16B (B, 8 halves along N) into double-buffered SMEM; commit per
// tile; wait_group 1 overlaps next-tile loads with current-tile compute.
// OOB handled by cp.async src-size zero-fill.

#include "matmul_warptile_dbuf_f16.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ void cp_async_4b(void *smem_dst, const void *gmem_src,
                                            int src_bytes) {
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr),
                 "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_16b(void *smem_dst, const void *gmem_src,
                                             int src_bytes) {
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr),
                 "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int Pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending));
}

template <int BK>
__device__ __forceinline__ int apad(int k) { return k + 2; }  // +2 halves padding

// Load one K-tile of A (natural layout, k-paired halves) and B (8-half chunks)
// into SMEM buffer `buf` via cp.async.
template <typename Traits, int BM, int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void
loadTileCpAsync16(const typename Traits::T *A, const typename Traits::T *B, int N,
                  int blockRow, int blockCol, int tileIdx, int buf,
                  typename Traits::T (*As)[2][BM][BK + 2],
                  typename Traits::T (*Bs)[2][BK][BN]) {
    using T = typename Traits::T;
    static_assert(sizeof(T) == 2, "16-bit only");
    // A tile: BM x BK, natural layout. Move 4B = 2 halves along k.
    // BK must be even. Flat-strided mapping: consecutive threads -> consecutive
    // k-pairs (coalesced along k), BM*BK/2 chunks.
    constexpr int K2 = BK / 2;
    for (int e = threadIdx.x; e < BM * K2; e += NUM_THREADS) {
        const int row = e / K2;      // m
        const int k2 = e % K2;       // k-pair index
        const int k = k2 * 2;
        const bool inBounds = (blockRow * BM + row) < N && (tileIdx + k + 1) < N;
        // src: A[row, tileIdx+k .. k+1] — 2 halves = 4B if both in bounds
        const T *src = A + (size_t)row * N + tileIdx + k;
        cp_async_4b(&(*As)[buf][row][k], inBounds ? src : A, inBounds ? 4 : 0);
    }
    // B tile: BK x BN, 16B = 8 halves along N.
    constexpr int N8 = BN / 8;
    for (int f = threadIdx.x; f < BK * N8; f += NUM_THREADS) {
        const int row = f / N8;      // k
        const int col8 = f % N8;     // 8-half column group
        const int col = col8 * 8;
        // B is pre-advanced by blockCol*BN at kernel entry (kernel line ~132);
        // do NOT add blockCol*BN again here (FP32 original computes src WITHOUT it).
        const T *src = B + (size_t)(tileIdx + row) * N + col;
        const bool rowOk = (tileIdx + row) < N;
        const int colGlobal = blockCol * BN + col;
        if (rowOk && colGlobal + 7 < N &&
            (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            cp_async_16b(&(*Bs)[buf][row][col], src, 16);
        } else {
            // tail: fall back to 4B pairs
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const bool ok = rowOk && (colGlobal + p * 2 + 1) < N;
                const T *psrc = src + p * 2;
                cp_async_4b(&(*Bs)[buf][row][col + p * 2], ok ? psrc : B,
                            ok ? 4 : 0);
            }
        }
    }
}

template <typename Traits, int BM, int BN, int BK, int WM, int WN, int WNITER,
          int TM, int TN, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmulWarptileDbufF16Kernel(const typename Traits::T * __restrict__ A,
                            const typename Traits::T * __restrict__ B,
                            float * __restrict__ C, int N) {
    using T = typename Traits::T;
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int THREADS_N = WSUBN / TN;

    static_assert(BM % WM == 0, "BM % WM");
    static_assert(BN % WN == 0, "BN % WN");
    static_assert(WN % WNITER == 0, "WN % WNITER");
    static_assert(WSUBN % TN == 0, "WSUBN % TN");
    static_assert(WSUBM % TM == 0, "WSUBM % TM");
    static_assert(32 % THREADS_N == 0, "32 % THREADS_N");
    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / 32, "warp cover");
    static_assert(WM * WN == 32 * TM * TN * WMITER * WNITER, "thread cover");
    static_assert(BK % 2 == 0, "BK must be even for k-paired A loads");

    __shared__ __align__(16) T As[2][BM][BK + 2];
    __shared__ __align__(16) T Bs[2][BK][BN];

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int numWarpsN = BN / WN;
    const int warpCol = warpId % numWarpsN;
    const int warpRow = warpId / numWarpsN;
    const int threadColInWarp = laneId % THREADS_N;
    const int threadRowInWarp = laneId / THREADS_N;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += (size_t)blockRow * BM * N;
    B += (size_t)blockCol * BN;
    C += (size_t)blockRow * BM * N + blockCol * BN;

    float threadResults[WMITER * TM][WNITER * TN] = {{0.0f}};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Prologue: K-tile 0 into buffer 0.
    loadTileCpAsync16<Traits, BM, BN, BK, NUM_THREADS>(A, B, N, blockRow,
                                                       blockCol, 0, 0, &As, &Bs);
    cp_async_commit();

    int bufIdx = 0;
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        const bool hasNext = (tileIdx + BK) < N;
        if (hasNext) {
            loadTileCpAsync16<Traits, BM, BN, BK, NUM_THREADS>(
                A, B, N, blockRow, blockCol, tileIdx + BK, 1 - bufIdx, &As, &Bs);
            cp_async_commit();
        }

        if (hasNext) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const T (*AsBuf)[BK + 2] = As[bufIdx];
        const T (*BsBuf)[BN] = Bs[bufIdx];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    regM[wSubRow * TM + i] = Traits::to_float(
                        AsBuf[warpRow * WM + wSubRow * WSUBM +
                              threadRowInWarp * TM + i][dotIdx]);
                }
            }
            #pragma unroll
            for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regN[wSubCol * TN + j] = Traits::to_float(
                        BsBuf[dotIdx][warpCol * WN + wSubCol * WSUBN +
                                      threadColInWarp * TN + j]);
                }
            }
            #pragma unroll
            for (int i = 0; i < WMITER * TM; i++) {
                #pragma unroll
                for (int j = 0; j < WNITER * TN; j++) {
                    threadResults[i][j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
        bufIdx = 1 - bufIdx;
    }

    #pragma unroll
    for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int globalRow = blockRow * BM + warpRow * WM +
                                  wSubRow * WSUBM + threadRowInWarp * TM + i;
            if (globalRow < N) {
                #pragma unroll
                for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        const int globalCol = blockCol * BN + warpCol * WN +
                                              wSubCol * WSUBN +
                                              threadColInWarp * TN + j;
                        if (globalCol < N) {
                            C[(size_t)(warpRow * WM + wSubRow * WSUBM +
                                       threadRowInWarp * TM + i) * N +
                              (warpCol * WN + wSubCol * WSUBN +
                               threadColInWarp * TN + j)] =
                                threadResults[wSubRow * TM + i][wSubCol * TN + j];
                        }
                    }
                }
            }
        }
    }
}

// device-side converter (float storage -> 16-bit storage)
template <typename Traits>
__global__ void convertKernelF16(const float * __restrict__ in,
                                 typename Traits::T * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
MatmulWarptileDbufF16<Traits>::MatmulWarptileDbufF16(int N, int blockDim,
                                                     int BM, int BN, int BK,
                                                     int WM, int WN, int WNITER,
                                                     int TM, int TN,
                                                     int NUM_THREADS)
    : N(N), blockDim(blockDim) {
    using T = typename Traits::T;
    cfg[0] = BM; cfg[1] = BN; cfg[2] = BK; cfg[3] = WM; cfg[4] = WN;
    cfg[5] = WNITER; cfg[6] = TM; cfg[7] = TN; cfg[8] = NUM_THREADS;
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(T)));
}

template <typename Traits>
MatmulWarptileDbufF16<Traits>::~MatmulWarptileDbufF16() {
    cudaFree(d_A16);
    cudaFree(d_B16);
}

template <typename Traits>
void MatmulWarptileDbufF16<Traits>::execute(const float *d_A, const float *d_B,
                                            float *d_C) {
    using T = typename Traits::T;
    // Convert FP32 storage -> 16-bit storage each execute() (N^2 elements vs
    // N^3 FLOPs = 0.02% of work at N=4096; kept inside execute for API purity,
    // matching how other typed kernels handle their input conversion).
    convertKernelF16<Traits><<<(N * N + 255) / 256, 256>>>(d_A, d_A16, N * N);
    convertKernelF16<Traits><<<(N * N + 255) / 256, 256>>>(d_B, d_B16, N * N);

    dim3 threads(cfg[8]);
    dim3 blocks((N + cfg[1] - 1) / cfg[1], (N + cfg[0] - 1) / cfg[0]);

    const T *A16 = d_A16;
    const T *B16 = d_B16;

#define LAUNCH_DBUF16(BM, BN, BK, WM, WN, WNITER, TM, TN, NT)                  \
    if (cfg[0] == BM && cfg[1] == BN && cfg[2] == BK && cfg[3] == WM &&        \
        cfg[4] == WN && cfg[5] == WNITER && cfg[6] == TM && cfg[7] == TN &&    \
        cfg[8] == NT) {                                                        \
        matmulWarptileDbufF16Kernel<Traits, BM, BN, BK, WM, WN, WNITER, TM,    \
                                    TN, NT>                                    \
            <<<blocks, threads>>>(A16, B16, d_C, N);                           \
        cudaCheckError(cudaGetLastError());                                    \
        return;                                                                \
    }
    // winner-family configs (FP32 winner + variants near it)
    LAUNCH_DBUF16(128, 256, 8, 64, 64, 2, 8, 4, 256)
    LAUNCH_DBUF16(128, 128, 8, 64, 64, 2, 8, 4, 128)
    LAUNCH_DBUF16(128, 128, 16, 64, 64, 2, 8, 4, 128)
    LAUNCH_DBUF16(128, 128, 16, 64, 32, 1, 8, 4, 256)
    LAUNCH_DBUF16(128, 128, 16, 64, 64, 1, 8, 4, 128)
    LAUNCH_DBUF16(128, 256, 16, 64, 64, 2, 8, 4, 256)
    LAUNCH_DBUF16(128, 256, 8, 64, 32, 1, 8, 4, 512)
    LAUNCH_DBUF16(128, 256, 8, 64, 64, 1, 8, 4, 256)
    LAUNCH_DBUF16(64, 128, 16, 64, 64, 1, 8, 4, 64)
    LAUNCH_DBUF16(128, 128, 8, 64, 32, 1, 8, 4, 256)
    LAUNCH_DBUF16(128, 128, 16, 64, 64, 2, 16, 4, 128)
    LAUNCH_DBUF16(128, 256, 16, 64, 64, 2, 16, 4, 256)
    LAUNCH_DBUF16(128, 128, 16, 64, 64, 2, 8, 8, 128)
    LAUNCH_DBUF16(128, 256, 8, 64, 64, 2, 16, 4, 256)
    LAUNCH_DBUF16(128, 128, 8, 64, 64, 2, 16, 4, 128)
    LAUNCH_DBUF16(64, 64, 16, 32, 64, 1, 8, 4, 64)
#undef LAUNCH_DBUF16

    fprintf(stderr,
            "dbuf_f16: unsupported config BM=%d BN=%d BK=%d WM=%d WN=%d "
            "WNITER=%d TM=%d TN=%d NT=%d\n",
            cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7],
            cfg[8]);
    exit(1);
}

template class MatmulWarptileDbufF16<DTypeTraitsHalf>;
