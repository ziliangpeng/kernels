// Rung 9c: Hopper WGMMA — v4, CTA tile 128x128, 4 warpgroups (2x2).
//
// Instruction: wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 (same
// sweep-verified macro as v2/v3). Each warpgroup computes a 64x64 quadrant;
// wgM = wg/2 picks the A half, wgN = wg%2 the B half.
//
// Key layout fact (v4): the no-swizzle interleave is 64-row periodic —
// row r+8 shifts the atom offset by exactly +1024B = one LBO unit — so the
// quadrant descriptors are simply base+1024 with the SAME LBO=1024/SBO=128.
//
// Pipeline (from v3, now cross-warpgroup safe): stage k+1 loads while the
// 4 warpgroups' wgmma run on stage k. wait_group 1 is per-warpgroup, so a
// __syncthreads() after it guarantees ALL warpgroups' previous reads are
// done before the double-buffer flip is overwritten.
//
// Roofline: AI = 2*128*128*16 / ((128+128)*16*2) = 64 FLOP/B -> ~230T cap
// at 3.35TB/s HBM (v3 measured ~91% of its own 115T cap: bandwidth-bound).

#include "matmul_wgmma.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t make_smem_desc(const void *ptr, uint32_t lbo,
                                                   uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    return d;
}

#define WGMMA_M64N64K16(d, desc_a, desc_b, scale_d)                            \
    asm volatile(                                                             \
        "{\n"                                                                 \
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "                 \
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}," \
        " %32,"                                                               \
        " %33,"                                                               \
        " %34, 1, 1, 0, 0;\n"                                                 \
        "}\n"                                                                 \
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),                  \
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),                  \
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),                 \
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),                 \
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),                 \
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),                 \
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),                 \
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])                  \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scale_d)));

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void fence_proxy_async() {
    // Orders prior SMEM writes (generic proxy) against subsequent wgmma
    // async-proxy reads of that SMEM. Required by PTX; __syncthreads alone
    // does NOT order the async proxy.
    asm volatile("fence.proxy.async.shared::cta;\n");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
// wait until at most N wgmma groups are pending (N=1: previous group done,
// its SMEM buffers free to overwrite while the current group still runs)
template <int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}
// ---- cp.async (Ampere-style 16B copies, v5) -------------------------------
//
// cp.async writes participate in the GENERIC proxy (unlike TMA); wgmma reads
// SMEM via the async proxy — so the v4-verified fence.proxy.async ordering
// is kept between copy completion and wgmma issue.
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---- converter kernels ----------------------------------------------------

__global__ void convertF32ToF16(const float * __restrict__ in,
                                __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// B: [K][N] row-major -> Bt: [N][K] (K-major)
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
// Grid: (N/128, N/128). 512 threads = 4 warpgroups, 2x2 quadrant split.

constexpr int WBM = 128;
constexpr int WBN = 128;
constexpr int WBK = 16;

__global__ __launch_bounds__(512) void matmulWgmmaKernel(
    const __half * __restrict__ A, const __half * __restrict__ Bt,
    float * __restrict__ C, int N) {

    // Interleave atoms: (row/8)*128 + chunk*1024 + (row%8)*16, chunk = k/8.
    // A 128x16 tile = 2 quadrants of 64x16; quadrant footprint 2048B.
    // v5.1: STAGE-deep pipeline (per-stage 4KB per matrix; 4 stages = 32KB).
    constexpr int STAGES = 4;
    __shared__ __align__(128) unsigned char A_s[STAGES][WBM * WBK * 2];
    __shared__ __align__(128) unsigned char B_s[STAGES][WBN * WBK * 2];

    const int blockM = blockIdx.y;
    const int blockN = blockIdx.x;
    const int tid = threadIdx.x;
    const int wg = tid / 128;          // warpgroup 0..3
    const int wgM = wg / 2, wgN = wg % 2;

    // v5: cp.async staged loads. Each thread issues ONE 16B cp.async (256
    // threads for A + 256 for B cover each 128x16 half-tile: row = e/2,
    // chunk = e%2 gives 128 rows x 2 k-atoms x 16B).
    const int lrow = (tid % 256) / 2;
    const int lchunk = (tid % 256) % 2;
    const unsigned loff = (lrow / 64) * 2048 + ((lrow % 64) / 8) * 128 +
                          lchunk * 1024 + (lrow % 8) * 16;
    const bool ldA = tid < 256;
    const __half *Atile = A + (size_t)(blockM * WBM) * N;
    const __half *Btile = Bt + (size_t)(blockN * WBN) * N;
    const __half *Gsrc = ldA ? Atile : Btile;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    // prologue: cp.async tiles 0..STAGES-2 -> stages 0..STAGES-2, one group
    // per stage (so wait_group<N> counts STAGES-1-N pending groups).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        cp_async16(ldA ? A_s[s] + loff : B_s[s] + loff,
                   Gsrc + (size_t)lrow * N + s * WBK + lchunk * 8);
        cp_async_commit();
    }

    for (int k0 = 0; k0 < N; k0 += WBK) {
        const int buf = k0 / WBK % STAGES;
        // Quadrant base: a 64x16 quadrant's atoms span 2048B (2 K-atoms x
        // LBO 1024); rows 64-127's quadrant starts at +2048.
        const unsigned char *Aq = A_s[buf] + wgM * 2048;
        const unsigned char *Bq = B_s[buf] + wgN * 2048;
        uint64_t descA = make_smem_desc(Aq, 1024, 128);
        uint64_t descB = make_smem_desc(Bq, 1024, 128);

        // Wait until THIS stage's cp.async group has landed. Groups more
        // recent than tile i: tiles i+1..i+2 (tile i+3 is issued AFTER this
        // wait) — so the allowed-pending depth is min(2, tilesLeft), NOT
        // STAGES-1 (that over-permits and reads unlanded tiles: v5.1 bug).
        // Tail iterations have fewer newer groups -> clamp to wait<0>.
        const int tilesLeft = (N - k0) / WBK - 1;
        if (tilesLeft >= 2)      cp_async_wait<2>();
        else if (tilesLeft == 1) cp_async_wait<1>();
        else                     cp_async_wait<0>();
        __syncthreads();
        fence_proxy_async();

        wgmma_fence();
        WGMMA_M64N64K16(acc, descA, descB, 1);
        wgmma_commit();

        // issue cp.async for stage k+STAGES-1 while the pipeline drains —
        // keeps STAGES-1 copies in flight at all times.
        const int ks = k0 + (STAGES - 1) * WBK;
        if (ks < N) {
            wgmma_wait<1>();          // wgmma from 2 iterations ago done -> its stage reusable
            __syncthreads();          // ALL wgs done reading that stage
            cp_async16(ldA ? A_s[ks / WBK % STAGES] + loff : B_s[ks / WBK % STAGES] + loff,
                       Gsrc + (size_t)lrow * N + ks + lchunk * 8);
            cp_async_commit();
        }
    }
    wgmma_wait0();
    cp_async_wait<0>();

    // epilogue (per-quadrant, sweep-verified m64n64 mapping)
    const int w = (tid % 128) / 32;
    const int l = tid % 32;
    const int rowBase = blockM * WBM + wgM * 64 + w * 16 + (l / 4);
    const int colBase = blockN * WBN + wgN * 64 + (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
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
    matmulWgmmaKernel<<<grid, 512>>>(d_A16, d_Bt16, d_C, N);
    cudaCheckError(cudaGetLastError());
}
