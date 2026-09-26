#include "matmul_warptile_dbuf.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdint>

// Warp Tiling + cp.async double buffering (Simon rung 12, raw PTX flavor)
//
// Same tiling, load mapping, and compute structure as matmul_warptile.cu.
// The ONLY change: GMEM->SMEM tile loads are issued with the cp.async PTX
// instruction into one of two SMEM buffers. While the block computes on
// buffer i, the loads for tile i+1 are already in flight into buffer 1-i.
// Load latency is hidden behind compute instead of serializing it.
//
// Pipeline mechanics (classic Ampere+ pattern, what CUTLASS uses pre-TMA):
//   - cp.async.ca / cp.async.cg issue async GMEM->SMEM copies that bypass
//     the register file; the optional src-size operand zero-fills the rest
//     of the copy, which handles matrix-boundary OOB for free.
//   - cp.async.commit_group batches a thread's pending copies into a group.
//   - cp.async.wait_group N waits until at most N groups are still pending:
//     with two groups in flight (current tile + next tile), wait_group 1
//     completes the current tile while the next tile's copies keep flying.
//
// Load widths: B tiles use 16-byte cp.async.cg (with 4-byte fallback when
// the global address is not 16B-aligned, e.g. odd N); A tiles use 4-byte
// cp.async.ca because the A tile is transposed on store (same as Simon).
//
// A/B vs matmul_warptile therefore isolates exactly one variable:
// synchronous loads -> cp.async double-buffered pipeline.

#define BM_DB 128
#define BN_DB 128
#define BK_DB 16
#define WM_DB 64
#define WN_DB 64
#define TM_DB 8
#define TN_DB 4
#define WARP_SIZE 32

#define WARPS_PER_BLOCK_X_DB (BN_DB / WN_DB)  // 2
#define WARPS_PER_BLOCK_Y_DB (BM_DB / WM_DB)  // 2
#define NUM_WARPS_DB (WARPS_PER_BLOCK_X_DB * WARPS_PER_BLOCK_Y_DB)  // 4

// Thread placement inside the warp tile (identical to matmul_warptile.cu)
#define WARP_THREAD_M_DB 4   // threads of a warp along M
#define WARP_THREAD_N_DB 8   // threads of a warp along N (4*8 = 32)
#define WARP_SUBTILE_M_DB 2  // TM-subtiles per thread along M
#define WARP_SUBTILE_N_DB 2  // TN-subtiles per thread along N

// Verify: WARP_THREAD_M * TM * WARP_SUBTILE_M = 4 * 8 * 2 = 64 = WM
// Verify: WARP_THREAD_N * TN * WARP_SUBTILE_N = 8 * 4 * 2 = 64 = WN

#define NUM_THREADS_DB (NUM_WARPS_DB * WARP_SIZE)  // 128

// ---------------------------------------------------------------------------
// cp.async wrappers
// ---------------------------------------------------------------------------

// 4-byte async copy; src_bytes=0 zero-fills the destination (OOB guard).
__device__ __forceinline__ void cp_async_4(float *smem_dst, const float *gmem_src,
                                           int src_bytes) {
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr),
                 "l"(gmem_src), "r"(src_bytes));
}

// 16-byte async copy (.cg: bypass L1); src_bytes<16 zero-fills the remainder.
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

// ---------------------------------------------------------------------------
// Tile loader
// ---------------------------------------------------------------------------

// Load one K-tile of A and B into SMEM buffer `buf` via cp.async.
// OOB elements are zero-filled by passing src_bytes = 0 (with the source
// pointer clamped to a valid in-bounds address for safety).
__device__ void loadTileCpAsync(const float *A, const float *B, int N,
                                int blockRow, int blockCol, int tileIdx, int buf,
                                float (*As)[2][BK_DB][BM_DB],
                                float (*Bs)[2][BK_DB][BN_DB]) {
    // ----- A tile: BM x BK, transposed into As[buf][k][m] -----
    // Mapping identical to matmul_warptile.cu: strideA = 8 rows per pass.
    const int strideA = NUM_THREADS_DB / BK_DB;  // 128 / 16 = 8
    const int innerRowA = threadIdx.x / BK_DB;
    const int innerColA = threadIdx.x % BK_DB;
    #pragma unroll
    for (int loadOffset = 0; loadOffset < BM_DB; loadOffset += strideA) {
        const int row = innerRowA + loadOffset;
        const bool inBounds = (blockRow * BM_DB + row) < N && (tileIdx + innerColA) < N;
        cp_async_4(&(*As)[buf][innerColA][row],
                   inBounds ? &A[row * N + tileIdx + innerColA] : A,
                   inBounds ? 4 : 0);
    }

    // ----- B tile: BK x BN, into Bs[buf][k][n] -----
    // NOTE: B was pre-advanced by blockCol * BN_DB at kernel start; column
    // indices below are TILE-LOCAL. Fast path: 16B cp.async.cg,
    // 128 threads / 32 float4-columns = 4 passes.
    const int innerRowB4 = threadIdx.x / (BN_DB / 4);   // 0..3
    const int innerColB4 = threadIdx.x % (BN_DB / 4);   // 0..31
    constexpr int ROW_STRIDE_B4 = NUM_THREADS_DB / (BN_DB / 4);  // 4
    #pragma unroll
    for (int pass = 0; pass < BK_DB / ROW_STRIDE_B4; ++pass) {
        const int row = innerRowB4 + pass * ROW_STRIDE_B4;   // k index, 0..15
        const int colLocal = innerColB4 * 4;                 // n within tile
        const int colGlobal = blockCol * BN_DB + colLocal;
        const float *src = &B[(tileIdx + row) * N + colLocal];
        const bool rowOk = (tileIdx + row) < N;
        if (rowOk && colGlobal + 3 < N &&
            (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            cp_async_16(&(*Bs)[buf][row][colLocal], src, 16);
        } else {
            // Boundary / unaligned path: per-element 4B copies.
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool ok = rowOk && (colGlobal + e) < N;
                cp_async_4(&(*Bs)[buf][row][colLocal + e],
                           ok ? &src[e] : B, ok ? 4 : 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void matmulWarptileDbufKernel(const float *A, const float *B, float *C, int N) {
    // Two SMEM buffers per operand; buffer i is computed while buffer 1-i fills.
    __shared__ __align__(16) float As[2][BK_DB][BM_DB];
    __shared__ __align__(16) float Bs[2][BK_DB][BN_DB];

    // Warp / thread placement (identical to matmul_warptile.cu)
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpRow = warpId / WARPS_PER_BLOCK_X_DB;       // 0-1
    const int warpCol = warpId % WARPS_PER_BLOCK_X_DB;       // 0-1
    const int threadRowInWarp = laneId / WARP_THREAD_N_DB;   // 0-3
    const int threadColInWarp = laneId % WARP_THREAD_N_DB;   // 0-7

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to the block tile origin
    A += blockRow * BM_DB * N;
    B += blockCol * BN_DB;
    C += blockRow * BM_DB * N + blockCol * BN_DB;

    float threadResults[WARP_SUBTILE_M_DB * TM_DB][WARP_SUBTILE_N_DB * TN_DB] = {{0.0f}};
    float regA[WARP_SUBTILE_M_DB * TM_DB];
    float regB[WARP_SUBTILE_N_DB * TN_DB];

    // Prologue: issue cp.async loads for K-tile 0 into buffer 0.
    loadTileCpAsync(A, B, N, blockRow, blockCol, 0, 0, &As, &Bs);
    cp_async_commit();

    int bufIdx = 0;
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_DB) {
        // Issue cp.async loads for the NEXT K-tile into the other buffer.
        // These copies fly while we wait for + compute the current tile.
        const bool hasNext = (tileIdx + BK_DB) < N;
        if (hasNext) {
            loadTileCpAsync(A, B, N, blockRow, blockCol, tileIdx + BK_DB, 1 - bufIdx,
                            &As, &Bs);
            cp_async_commit();
        }

        // Wait for the CURRENT tile's group to complete; the next tile's
        // group (issued just above) is allowed to keep flying.
        if (hasNext) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();  // block-wide visibility of the current tile

        const float (*AsBuf)[BM_DB] = As[bufIdx];
        const float (*BsBuf)[BN_DB] = Bs[bufIdx];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_DB; dotIdx++) {
            #pragma unroll
            for (int subtileM = 0; subtileM < WARP_SUBTILE_M_DB; subtileM++) {
                #pragma unroll
                for (int i = 0; i < TM_DB; i++) {
                    const int asRow = warpRow * WM_DB +
                                      subtileM * (WM_DB / WARP_SUBTILE_M_DB) +
                                      threadRowInWarp * TM_DB + i;
                    regA[subtileM * TM_DB + i] = AsBuf[dotIdx][asRow];
                }
            }
            #pragma unroll
            for (int subtileN = 0; subtileN < WARP_SUBTILE_N_DB; subtileN++) {
                #pragma unroll
                for (int j = 0; j < TN_DB; j++) {
                    const int bsCol = warpCol * WN_DB +
                                      subtileN * (WN_DB / WARP_SUBTILE_N_DB) +
                                      threadColInWarp * TN_DB + j;
                    regB[subtileN * TN_DB + j] = BsBuf[dotIdx][bsCol];
                }
            }
            #pragma unroll
            for (int i = 0; i < WARP_SUBTILE_M_DB * TM_DB; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_SUBTILE_N_DB * TN_DB; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        // All threads must finish computing on the current buffer before the
        // next iteration issues cp.async copies into it (buffer-reuse safety).
        __syncthreads();
        bufIdx = 1 - bufIdx;
    }

    // Write results (identical to matmul_warptile.cu)
    #pragma unroll
    for (int subtileM = 0; subtileM < WARP_SUBTILE_M_DB; subtileM++) {
        #pragma unroll
        for (int i = 0; i < TM_DB; i++) {
            const int globalRow = blockRow * BM_DB + warpRow * WM_DB +
                                  subtileM * (WM_DB / WARP_SUBTILE_M_DB) +
                                  threadRowInWarp * TM_DB + i;
            if (globalRow < N) {
                #pragma unroll
                for (int subtileN = 0; subtileN < WARP_SUBTILE_N_DB; subtileN++) {
                    #pragma unroll
                    for (int j = 0; j < TN_DB; j++) {
                        const int globalCol = blockCol * BN_DB + warpCol * WN_DB +
                                              subtileN * (WN_DB / WARP_SUBTILE_N_DB) +
                                              threadColInWarp * TN_DB + j;
                        if (globalCol < N) {
                            const int localRow = warpRow * WM_DB +
                                                 subtileM * (WM_DB / WARP_SUBTILE_M_DB) +
                                                 threadRowInWarp * TM_DB + i;
                            const int localCol = warpCol * WN_DB +
                                                 subtileN * (WN_DB / WARP_SUBTILE_N_DB) +
                                                 threadColInWarp * TN_DB + j;
                            C[localRow * N + localCol] =
                                threadResults[subtileM * TM_DB + i][subtileN * TN_DB + j];
                        }
                    }
                }
            }
        }
    }
}

MatmulWarptileDbuf::MatmulWarptileDbuf(int N, int blockDim) : N(N), blockDim(blockDim) {}

void MatmulWarptileDbuf::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_DB);
    dim3 blocks((N + BN_DB - 1) / BN_DB,
                (N + BM_DB - 1) / BM_DB);
    matmulWarptileDbufKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWarptileDbuf::~MatmulWarptileDbuf() {}
