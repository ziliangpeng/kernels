// Rung 9c: Hopper WGMMA — v1 correctness-first implementation.
//
// Instruction: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
//   - both operands from SMEM (SS), K-major, NO swizzle (mode 0)
//   - BK=16 keeps every tile start at a swizzle-pattern boundary (base
//     offset field = 0), LBO/SBO constants in element terms:
//       A_s[64][16] halfs, row pitch 32B -> LBO=16B (1 core matrix along K),
//       SBO=256B (8 rows of 32B = core-matrix step along M)
//   - B pre-transposed on device to [N][K] (K-major) so trans_b=0
//   - single buffer, cp.async not used in v1 (plain SMEM stores) —
//     feeding strategy is v2's problem; v1 asks: does the descriptor math
//     produce correct math and what does an unfed wgmma deliver?

#include "matmul_wgmma.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t make_smem_desc(const void *ptr, uint32_t lbo,
                                                   uint32_t sbo) {
    // No-swizzle descriptor. Fields (PTX ISA "GMMA descriptor"):
    //   [13:0]  smem addr >> 4       [29:16] LBO >> 4
    //   [45:32] SBO >> 4             [61:49] base offset (0 here)
    //   [63:62] swizzle mode: 0 = none
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    // bits 46-48 and 49-61 zero, swizzle 00
    return d;
}

// d[8][8] = 64 f32 accumulators (m64n128: 128 f32 across 128 threads? no —
// m64n128 accumulator = 64*128/128 = 64 f32 per thread, held as 64 regs).
// Layout used by PTX: 64 regs named {%0..%63}. We keep them in a float
// array of 64 and pass all 64 as "+r" constraints.

#define WGMMA_M64N128K16_F32F16F16(d, desc_a, desc_b, scale_d)                \
    asm volatile(                                                             \
        "{\n"                                                                 \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "                \
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                           \
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "                           \
        "%16, %17, %18, %19, %20, %21, %22, %23, "                            \
        "%24, %25, %26, %27, %28, %29, %30, %31, "                            \
        "%32, %33, %34, %35, %36, %37, %38, %39, "                            \
        "%40, %41, %42, %43, %44, %45, %46, %47, "                            \
        "%48, %49, %50, %51, %52, %53, %54, %55, "                            \
        "%56, %57, %58, %59, %60, %61, %62, %63},"                            \
        " %64,"                                                               \
        " %65,"                                                               \
        " %66, 1, 1, 0, 0;\n"                                                 \
        "}\n"                                                                 \
        : "+r"(d[0]),  "+r"(d[1]),  "+r"(d[2]),  "+r"(d[3]),                  \
          "+r"(d[4]),  "+r"(d[5]),  "+r"(d[6]),  "+r"(d[7]),                  \
          "+r"(d[8]),  "+r"(d[9]),  "+r"(d[10]), "+r"(d[11]),                 \
          "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),                 \
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),                 \
          "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),                 \
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),                 \
          "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),                 \
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),                 \
          "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),                 \
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]),                 \
          "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),                 \
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]),                 \
          "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),                 \
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),                 \
          "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])                  \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scale_d)));

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N));
}

// ---- converter kernels ----------------------------------------------------

__global__ void convertF32ToF16(const float * __restrict__ in,
                                __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// B: [K][N] row-major -> Bt: [N][K] (K-major), tiled coalesced-ish
__global__ void transposeB(const __half * __restrict__ B,
                           __half * __restrict__ Bt, int N) {
    __shared__ __half tile[32][33];
    int x = blockIdx.x * 32 + (threadIdx.x % 32);
    int y = blockIdx.y * 32 + (threadIdx.x / 32);
    if (x < N && y < N) tile[threadIdx.x % 32][threadIdx.x / 32] = B[y * N + x];
    __syncthreads();
    int xx = blockIdx.y * 32 + (threadIdx.x % 32);
    int yy = blockIdx.x * 32 + (threadIdx.x / 32);
    if (xx < N && yy < N) Bt[yy * N + xx] = tile[threadIdx.x % 32][threadIdx.x / 32];
}

// ---- main kernel ----------------------------------------------------------
//
// Grid: (N/128, N/64). One warpgroup per 64x128 output tile.
// SMEM: A_s[64][16] (2KB), B_s[128][16] (4KB) — single buffer, reloaded
// from GMEM each K-step via plain stores (v1; correctness first).

constexpr int WBM = 64;
constexpr int WBN = 128;
constexpr int WBK = 16;

__global__ __launch_bounds__(128) void matmulWgmmaKernel(
    const __half * __restrict__ A, const __half * __restrict__ Bt,
    float * __restrict__ C, int N) {

    __shared__ __align__(128) __half A_s[WBM][WBK];
    __shared__ __align__(128) __half B_s[WBN][WBK];

    const int blockM = blockIdx.y;  // 64-row tile
    const int blockN = blockIdx.x;  // 128-col tile

    const int tid = threadIdx.x;
    // load mapping: 128 threads move 64x16 + 128x16 halfs
    // A: 1024 halfs -> 8 per thread; B: 2048 halfs -> 16 per thread
    const int aRow = tid / 2;             // 0..63 (64 rows, 2 threads per row)
    const int aCol = (tid % 2) * 8;       // 0 or 8 (8 halfs each)
    const int bRow = tid;                 // 0..127 (one row each)
    const int bColBase = 0;               // 16 halfs per row -> loop below

    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    const __half *Atile = A + (size_t)(blockM * WBM) * N;
    const __half *Btile = Bt + (size_t)(blockN * WBN) * N;

    wgmma_fence();

    for (int k0 = 0; k0 < N; k0 += WBK) {
        // ---- load A_s: rows 0..63, cols k0..k0+15 (2 threads/row, 8 halfs)
        {
            const __half4 *src = reinterpret_cast<const __half4 *>(Atile + (size_t)aRow * N + k0 + aCol);
            __half4 *dst = reinterpret_cast<__half4 *>(&A_s[aRow][aCol]);
            dst[0] = src[0];
            dst[1] = src[1];
        }
        // ---- load B_s: 128 rows x 16 halfs (one row per thread, 4 half4)
        {
            const __half4 *src = reinterpret_cast<const __half4 *>(Btile + (size_t)bRow * N + k0);
            __half4 *dst = reinterpret_cast<__half4 *>(&B_s[bRow][0]);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
        }
        __syncthreads();

        // ---- one wgmma over this K-slice
        uint64_t descA = make_smem_desc(&A_s[0][0], 16, WBM * 2 * 8);
        uint64_t descB = make_smem_desc(&B_s[0][0], 16, WBN * 2 * 8);
        wgmma_fence();
        WGMMA_M64N128K16_F32F16F16(acc, descA, descB, 1);
        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // ---- write C: m64n128 fragment layout: thread t of the warpgroup owns
    // rows/cols per PTX spec: warp w = tid/32 owns rows w*16..w*16+15;
    // within a warp, lane l: row = (l/4) + w*16 base... standard mapping:
    //   acc index i in [0,64): 8 groups of 8: i = g*8 + j
    //   row = w*16 + (l/4) + (j/2)*8? — use the documented m64nN mapping:
    //   each thread holds 64 values: for g in 0..7 (n-blocks of 16 cols):
    //     row = w*16 + l/4,  col = g*16 + (l%4)*2 + (0/1)
    //     plus second row block: row += 8 for j in {4..7}
    const int w = tid / 32;
    const int l = tid % 32;
    const int rowBase = blockM * WBM + w * 16 + (l / 4);
    const int colBase = blockN * WBN + (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const int r = rowBase + ((j / 4) ? 8 : 0);
            const int c = colBase + g * 16 + (j % 4);
            C[(size_t)r * N + c]           = acc[g * 8 + j];
            C[(size_t)r * N + c + 1]       = acc[g * 8 + j + 1];
        }
    }
}

// ---- host wrapper ---------------------------------------------------------

MatmulWgmma::MatmulWgmma(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmma::~MatmulWgmma() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmma::execute(const float *d_A, const float *d_B, float *d_C) {
    int n = N * N;
    convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);

    dim3 grid(N / WBN, N / WBM);
    matmulWgmmaKernel<<<grid, 128>>>(d_A16, d_Bt16, d_C, N);
    cudaCheckError(cudaGetLastError());
}
