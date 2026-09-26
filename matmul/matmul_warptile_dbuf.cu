#include "matmul_warptile_dbuf.h"
#include "cuda_utils.h"
#include "dbuf_config.h"
#include <cuda_runtime.h>
#include <cstdint>

// Warp Tiling + cp.async double buffering (Simon rung 12, raw PTX flavor)
// — AUTOTUNABLE (templated) version.
//
// Algorithm identical to the original hardcoded kernel (verified correct at
// N=512/1000/2048/4096, +16.5% vs warptile on H100). Every tiling dimension
// is now a template parameter so the autotuner can sweep them:
//
//   BM, BN, BK  : thread-block tile
//   WM, WN      : warp tile inside the block tile
//   WNITER      : warp-subtile iterations along N
//   TM, TN      : per-thread register tile
//   NUM_THREADS : threads per block (must satisfy (BM/WM)*(BN/WN)*32)
//
// Derived (constexpr, Simon's parameterization):
//   WMITER = (WM*WN) / (32*TM*TN*WNITER)   warp-subtile iterations along M
//   WSUBM  = WM / WMITER,  WSUBN = WN / WNITER
//
// Pipeline mechanics (unchanged): cp.async.ca (A, 4B, transposed on store) +
// cp.async.cg (B, 16B float4) into double-buffered SMEM; commit_group per
// tile; wait_group 1 completes the current tile while the next tile's copies
// keep flying. OOB handled by cp.async src-size zero-fill.
//
// Load mapping is flat-strided (works for ANY valid config, coalesced):
//   A: consecutive threads -> consecutive k (within a row of the A tile)
//   B: consecutive threads -> consecutive float4 (within a k-row of B tile)

__device__ __forceinline__ void cp_async_4(float *smem_dst, const float *gmem_src,
                                           int src_bytes) {
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr),
                 "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_16(float *smem_dst, const float *gmem_src,
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

// Load one K-tile of A (transposed) and B into SMEM buffer `buf` via cp.async.
template <int BM, int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void
loadTileCpAsync(const float *A, const float *B, int N, int blockRow, int blockCol,
                int tileIdx, int buf, float (*As)[2][BK][BM], float (*Bs)[2][BK][BN]) {
    // A tile: BM x BK, transposed into As[buf][k][m].
    for (int e = threadIdx.x; e < BM * BK; e += NUM_THREADS) {
        const int row = e / BK;  // m
        const int k = e % BK;
        const bool inBounds = (blockRow * BM + row) < N && (tileIdx + k) < N;
        cp_async_4(&(*As)[buf][k][row],
                   inBounds ? &A[row * N + tileIdx + k] : A,
                   inBounds ? 4 : 0);
    }
    // B tile: BK x BN, 16B chunks into Bs[buf][k][n]. B is pre-advanced by
    // blockCol*BN; column indices are tile-local.
    constexpr int N4 = BN / 4;
    for (int f = threadIdx.x; f < BK * N4; f += NUM_THREADS) {
        const int row = f / N4;      // k
        const int col4 = f % N4;     // float4 column within the tile
        const int colGlobal = blockCol * BN + col4 * 4;
        const float *src = &B[(tileIdx + row) * N + col4 * 4];
        const bool rowOk = (tileIdx + row) < N;
        if (rowOk && colGlobal + 3 < N &&
            (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            cp_async_16(&(*Bs)[buf][row][col4 * 4], src, 16);
        } else {
            #pragma unroll
            for (int e4 = 0; e4 < 4; ++e4) {
                const bool ok = rowOk && (colGlobal + e4) < N;
                cp_async_4(&(*Bs)[buf][row][col4 * 4 + e4],
                           ok ? &src[e4] : B, ok ? 4 : 0);
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmulWarptileDbufKernelT(const float *A, const float *B, float *C, int N) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int THREADS_N = WSUBN / TN;     // warp threads along N

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(WN % WNITER == 0, "WN must be divisible by WNITER");
    static_assert(WSUBN % TN == 0, "WSUBN must be divisible by TN");
    static_assert(WSUBM % TM == 0, "WSUBM must be divisible by TM");
    static_assert(32 % THREADS_N == 0, "warp threads along N must divide 32");
    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / 32,
                  "warp tiles must exactly cover the block tile");
    static_assert(WM * WN == 32 * TM * TN * WMITER * WNITER,
                  "thread tiles must exactly cover the warp tile");

    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int numWarpsN = BN / WN;
    const int warpCol = warpId % numWarpsN;
    const int warpRow = warpId / numWarpsN;
    const int threadColInWarp = laneId % THREADS_N;
    const int threadRowInWarp = laneId / THREADS_N;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    float threadResults[WMITER * TM][WNITER * TN] = {{0.0f}};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Prologue: K-tile 0 into buffer 0.
    loadTileCpAsync<BM, BN, BK, NUM_THREADS>(A, B, N, blockRow, blockCol, 0, 0,
                                             &As, &Bs);
    cp_async_commit();

    int bufIdx = 0;
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        const bool hasNext = (tileIdx + BK) < N;
        if (hasNext) {
            loadTileCpAsync<BM, BN, BK, NUM_THREADS>(
                A, B, N, blockRow, blockCol, tileIdx + BK, 1 - bufIdx, &As, &Bs);
            cp_async_commit();
        }

        if (hasNext) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const float (*AsBuf)[BM] = As[bufIdx];
        const float (*BsBuf)[BN] = Bs[bufIdx];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    regM[wSubRow * TM + i] =
                        AsBuf[dotIdx][warpRow * WM + wSubRow * WSUBM +
                                      threadRowInWarp * TM + i];
                }
            }
            #pragma unroll
            for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regN[wSubCol * TN + j] =
                        BsBuf[dotIdx][warpCol * WN + wSubCol * WSUBN +
                                      threadColInWarp * TN + j];
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

    // Write results
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
                            C[(warpRow * WM + wSubRow * WSUBM +
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

// ---------------------------------------------------------------------------
// Host-side launch. Config comes from g_dbuf (set by CLI flags / defaults).
// The dispatch table over the sweep space is generated by
// scripts/gen_dbuf_dispatch.py into dbuf_dispatch.inc — the generator is the
// single source of truth for both the dispatch table and the sweep list.
// ---------------------------------------------------------------------------

void launchWarptileDbuf(const float *d_A, const float *d_B, float *d_C, int N,
                        int BM, int BN, int BK, int WM, int WN, int WNITER,
                        int TM, int TN, int NUM_THREADS) {
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);
#include "dbuf_dispatch.inc"
    fprintf(stderr,
            "warptile_dbuf: unsupported config BM=%d BN=%d BK=%d WM=%d WN=%d "
            "WNITER=%d TM=%d TN=%d NT=%d (regenerate dbuf_dispatch.inc)\n",
            BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS);
    exit(1);
}

// Default = the original hardcoded, verified config.
DbufConfig g_dbuf = {128, 128, 16, 64, 64, 2, 8, 4, 128};

MatmulWarptileDbuf::MatmulWarptileDbuf(int N, int blockDim)
    : N(N), blockDim(blockDim) {}

void MatmulWarptileDbuf::execute(const float *d_A, const float *d_B, float *d_C) {
    launchWarptileDbuf(d_A, d_B, d_C, N, g_dbuf.BM, g_dbuf.BN, g_dbuf.BK,
                       g_dbuf.WM, g_dbuf.WN, g_dbuf.WNITER, g_dbuf.TM,
                       g_dbuf.TN, g_dbuf.NUM_THREADS);
    cudaCheckError(cudaGetLastError());
}

MatmulWarptileDbuf::~MatmulWarptileDbuf() {}
