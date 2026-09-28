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
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),                  \
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),                  \
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),                 \
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),                 \
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),                 \
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),                 \
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),                 \
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),                 \
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),                 \
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),                 \
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),                 \
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),                 \
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),                 \
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),                 \
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),                 \
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])                  \
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
    if (xx < N && yy < N) Bt[yy * N + xx] = tile[threadIdx.x / 32][threadIdx.x % 32];
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

    // Canonical no-swizzle GMMA layout: core matrix = 8x8 halfs contiguous
    // (128B). Atom (mi, ki) at mi*128 + ki*1024 bytes. LBO=1024 (K step),
    // SBO=128 (M/N step). A tile: 8x2 atoms = 2KB; B tile: 16x2 atoms = 4KB.
    __shared__ __align__(128) unsigned char A_s[WBM * WBK * 2];
    __shared__ __align__(128) unsigned char B_s[WBN * WBK * 2];

    const int blockM = blockIdx.y;  // 64-row tile
    const int blockN = blockIdx.x;  // 128-col tile

    const int tid = threadIdx.x;
    // load mapping: 128 threads move 64x16 + 128x16 halfs
    // A: 1024 halfs -> 8 per thread; B: 2048 halfs -> 16 per thread
    const int aRow = tid / 2;             // 0..63 (64 rows, 2 threads per row)
    const int aCol = (tid % 2) * 8;       // 0 or 8 (8 halfs each)
    const int bRow = tid;                 // 0..127 (one Bt row each)

    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    const __half *Atile = A + (size_t)(blockM * WBM) * N;
    const __half *Btile = Bt + (size_t)(blockN * WBN) * N;

    wgmma_fence();

    for (int k0 = 0; k0 < N; k0 += WBK) {
        // ---- load A_s: thread t -> (row = t/2, half-chunk = t%2 of 8)
        // destination byte offset: (row/8)*128 + (t%2)*1024 + (row%8)*16
        {
            const uint4 *src = reinterpret_cast<const uint4 *>(Atile + (size_t)aRow * N + k0 + aCol);
            uint4 *dst = reinterpret_cast<uint4 *>(A_s + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
            *dst = *src;
        }
        // ---- load B_s: thread t -> Bt row t; two 16B chunks (k 0-7, 8-15)
        {
            const uint4 *src = reinterpret_cast<const uint4 *>(Btile + (size_t)bRow * N + k0);
            uint4 *dst0 = reinterpret_cast<uint4 *>(B_s + (bRow / 8) * 128 + (bRow % 8) * 16);
            uint4 *dst1 = reinterpret_cast<uint4 *>(B_s + (bRow / 8) * 128 + 1024 + (bRow % 8) * 16);
            dst0[0] = src[0];
            dst1[0] = src[1];
        }
        __syncthreads();

        // ---- one wgmma over this K-slice
        uint64_t descA = make_smem_desc(A_s, 1024, 128);  // LBO=K atom step, SBO=M atom step
        uint64_t descB = make_smem_desc(B_s, 1024, 128);
        wgmma_fence();
        WGMMA_M64N128K16_F32F16F16(acc, descA, descB, 1);
        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // ---- write C: PTX m64nN accumulator fragment mapping (pyptx/PTX spec):
    //   frag_row = w*16 + lane/4, frag_col = (lane%4)*2
    //   per column group g (8 cols wide): 4 regs
    //     acc[g*4+0] -> (frag_row,   frag_col + g*8)
    //     acc[g*4+1] -> (frag_row,   frag_col + g*8 + 1)
    //     acc[g*4+2] -> (frag_row+8, frag_col + g*8)
    //     acc[g*4+3] -> (frag_row+8, frag_col + g*8 + 1)
    const int w = tid / 32;
    const int l = tid % 32;
    const int rowBase = blockM * WBM + w * 16 + (l / 4);
    const int colBase = blockN * WBN + (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 16; g++) {
        C[(size_t)(rowBase) * N + colBase + g * 8]      = acc[g * 4 + 0];
        C[(size_t)(rowBase) * N + colBase + g * 8 + 1]  = acc[g * 4 + 1];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8]  = acc[g * 4 + 2];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8 + 1] = acc[g * 4 + 3];
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
