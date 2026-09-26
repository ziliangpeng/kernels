#include "matmul_warptile_dbuf.h"
#include "cuda_utils.h"
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

// Warp Tiling + async double buffering (Simon rung 12 style)
//
// Same tiling, load mapping, and compute structure as matmul_warptile.cu.
// The ONLY change: GMEM->SMEM tile loads are issued asynchronously
// (cuda::memcpy_async) into one of two SMEM buffers, with two
// cuda::barrier objects tracking copy completion. While the block computes
// on buffer i, the loads for tile i+1 are already in flight into buffer
// 1-i. Load latency is hidden behind compute instead of serializing it.
//
// Load widths follow Simon's kernel 12: B tiles use 16-byte (float4) async
// copies; A tiles use 4-byte async copies because the A tile is transposed
// on store (a 16B load cannot land transposed, so it is split into 4
// scalar async copies, same as the reference).
//
// A/B vs matmul_warptile therefore isolates exactly one variable:
// synchronous loads -> async double-buffered pipeline.

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

// Load one K-tile of A and B into SMEM buffer `buf`, asynchronously.
// In-bounds elements go through cuda::memcpy_async (tracked by `barrier`);
// out-of-bounds slots are zero-filled with plain stores. Both become
// visible after the barrier's arrive_and_wait (block-scope release-acquire),
// so no extra __syncthreads is needed for correctness.
__device__ void loadTileAsync(const float *A, const float *B, int N,
                              int blockRow, int blockCol, int tileIdx, int buf,
                              float (*As)[2][BK_DB][BM_DB],
                              float (*Bs)[2][BK_DB][BN_DB],
                              cuda::barrier<cuda::thread_scope_block> &barrier) {
    // A tile: BM x BK elements, transposed into As[buf][k][m].
    // Mapping identical to matmul_warptile.cu: strideA = 8 rows per pass.
    const int strideA = NUM_THREADS_DB / BK_DB;  // 128 / 16 = 8
    const int innerRowA = threadIdx.x / BK_DB;
    const int innerColA = threadIdx.x % BK_DB;
    #pragma unroll
    for (int loadOffset = 0; loadOffset < BM_DB; loadOffset += strideA) {
        const int row = innerRowA + loadOffset;
        const bool rowOk = (blockRow * BM_DB + row) < N;
        const bool colOk = (tileIdx + innerColA) < N;
        if (rowOk && colOk) {
            cuda::memcpy_async(&(*As)[buf][innerColA][row],
                               &A[row * N + tileIdx + innerColA],
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                               barrier);
        } else {
            (*As)[buf][innerColA][row] = 0.0f;
        }
    }

    // B tile: BK x BN elements into Bs[buf][k][n], float4 (16B) async copies.
    // 128 threads / 32 float4-columns per row = 4 row passes per thread.
    const int innerRowB4 = threadIdx.x / (BN_DB / 4);   // 0..3
    const int innerColB4 = threadIdx.x % (BN_DB / 4);   // 0..31
    constexpr int ROW_STRIDE_B4 = NUM_THREADS_DB / (BN_DB / 4);  // 4
    #pragma unroll
    for (int pass = 0; pass < BK_DB / ROW_STRIDE_B4; ++pass) {
        const int row = innerRowB4 + pass * ROW_STRIDE_B4;
        const bool rowOk = (tileIdx + row) < N;
        const int colBase = blockCol * BN_DB + innerColB4 * 4;
        if (rowOk && colBase + 3 < N) {
            // Fast path: full float4 in bounds.
            cuda::memcpy_async(&(*Bs)[buf][row][innerColB4 * 4],
                               &B[(tileIdx + row) * N + colBase],
                               cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                               barrier);
        } else {
            // Boundary path: per-element copy or zero.
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (rowOk && colBase + e < N) {
                    cuda::memcpy_async(&(*Bs)[buf][row][innerColB4 * 4 + e],
                                       &B[(tileIdx + row) * N + colBase + e],
                                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                                       barrier);
                } else {
                    (*Bs)[buf][row][innerColB4 * 4 + e] = 0.0f;
                }
            }
        }
    }
}

__global__ void matmulWarptileDbufKernel(const float *A, const float *B, float *C, int N) {
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

    // Two SMEM buffers per operand + two barriers (front = buffer being
    // computed, back = buffer being filled). Barriers are swapped each
    // iteration, Simon rung-12 style.
    __shared__ __align__(16) float As[2][BK_DB][BM_DB];
    __shared__ __align__(16) float Bs[2][BK_DB][BN_DB];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrierStorage[2];

    if (block.thread_rank() == 0) {
        cuda::init(&barrierStorage[0], block.size());
        cuda::init(&barrierStorage[1], block.size());
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block> *frontBarrier = &barrierStorage[0];
    cuda::barrier<cuda::thread_scope_block> *backBarrier = &barrierStorage[1];

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

    // Prologue: issue async loads for K-tile 0 into buffer 0 (front barrier).
    loadTileAsync(A, B, N, blockRow, blockCol, 0, 0, &As, &Bs, *frontBarrier);

    int bufIdx = 0;
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_DB) {
        // Issue async loads for the NEXT K-tile into the other buffer,
        // tracked by the back barrier. Overlaps with the compute below.
        if (tileIdx + BK_DB < N) {
            loadTileAsync(A, B, N, blockRow, blockCol, tileIdx + BK_DB, 1 - bufIdx,
                          &As, &Bs, *backBarrier);
        }

        // Wait until the CURRENT tile's loads have landed, then compute.
        frontBarrier->arrive_and_wait();

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

        // Swap buffers and barriers for the next iteration.
        bufIdx = 1 - bufIdx;
        cuda::barrier<cuda::thread_scope_block> *tmp = frontBarrier;
        frontBarrier = backBarrier;
        backBarrier = tmp;

        // All threads must finish computing on the old buffer before anyone
        // issues new async copies into it (buffer-reuse safety).
        __syncthreads();
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
