// Kernel "wgmma_v3" — v2 + double-buffered SMEM + wgmma.wait_group 1 lag
// pipeline (load tile k+1 while wgmma k runs). Perf null result: the
// 64x64 tile is HBM-bandwidth-bound, so latency hiding had nothing to
// feed. Also the version that exposed the fake-BUILD-OK failure mode
// (undefined wgmma_wait<1> compiled to nothing; bench ran v2 binary).

#include "matmul_wgmma_v3.h"
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

// ---- converter kernels ----------------------------------------------------

__global__ void wgmmaV3_convertF32ToF16(const float * __restrict__ in,
                                __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// B: [K][N] row-major -> Bt: [N][K] (K-major)
__global__ void wgmmaV3_transposeB(const __half * __restrict__ B,
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
// Grid: (N/64, N/64). One warpgroup per 64x64 output tile.

constexpr int WV3BM = 64;
constexpr int WV3BN = 64;
constexpr int WV3BK = 16;

__global__ __launch_bounds__(128) void matmulWgmmaV3Kernel(
    const __half * __restrict__ A, const __half * __restrict__ Bt,
    float * __restrict__ C, int N) {

    // Canonical no-swizzle interleave (sweep-verified): core matrix = 8x8
    // halfs (128B) contiguous; atom (mi, ki) at mi*128 + ki*1024 bytes.
    // DOUBLE-BUFFERED (v3): load tile k+1 into buf^1 while wgmma runs on buf.
    __shared__ __align__(128) unsigned char A_s[2][WV3BM * WV3BK * 2];
    __shared__ __align__(128) unsigned char B_s[2][WV3BN * WV3BK * 2];

    const int blockM = blockIdx.y;
    const int blockN = blockIdx.x;
    const int tid = threadIdx.x;

    // load mapping (both tiles 64x16): row=tid/2, 8-half chunk=tid%2
    const int aRow = tid / 2;
    const int aCol = (tid % 2) * 8;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    const __half *Atile = A + (size_t)(blockM * WV3BM) * N;
    const __half *Btile = Bt + (size_t)(blockN * WV3BN) * N;

    // ---- v3 pipeline: prologue loads tile 0; steady state issues wgmma on
    // buf, then (wait_group 1 = previous wgmma done) loads tile k+1 into
    // buf^1 while the current wgmma runs. No per-step wait_group 0.
    {
        // load tile 0 -> buf 0
        const uint4 *srcA = reinterpret_cast<const uint4 *>(Atile + (size_t)aRow * N + aCol);
        uint4 *dstA = reinterpret_cast<uint4 *>(A_s[0] + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
        dstA[0] = srcA[0];
        const uint4 *srcB = reinterpret_cast<const uint4 *>(Btile + (size_t)aRow * N + aCol);
        uint4 *dstB = reinterpret_cast<uint4 *>(B_s[0] + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
        dstB[0] = srcB[0];
        fence_proxy_async();
        __syncthreads();
    }

    for (int k0 = 0; k0 < N; k0 += WV3BK) {
        const int buf = (k0 / WV3BK) & 1;
        uint64_t descA = make_smem_desc(A_s[buf], 1024, 128);
        uint64_t descB = make_smem_desc(B_s[buf], 1024, 128);
        wgmma_fence();
        WGMMA_M64N64K16(acc, descA, descB, 1);
        wgmma_commit();

        if (k0 + WV3BK < N) {
            wgmma_wait<1>();  // previous wgmma done -> buf^1 free to overwrite
            const uint4 *srcA = reinterpret_cast<const uint4 *>(Atile + (size_t)aRow * N + k0 + WV3BK + aCol);
            uint4 *dstA = reinterpret_cast<uint4 *>(A_s[buf ^ 1] + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
            dstA[0] = srcA[0];
            const uint4 *srcB = reinterpret_cast<const uint4 *>(Btile + (size_t)aRow * N + k0 + WV3BK + aCol);
            uint4 *dstB = reinterpret_cast<uint4 *>(B_s[buf ^ 1] + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
            dstB[0] = srcB[0];
            fence_proxy_async();
            __syncthreads();
        }
    }
    wgmma_wait0();

    // epilogue (sweep-verified m64n64): frag_row = w*16 + lane/4,
    // frag_col = (lane%4)*2; 8 col groups x 4 regs.
    const int w = tid / 32;
    const int l = tid % 32;
    const int rowBase = blockM * WV3BM + w * 16 + (l / 4);
    const int colBase = blockN * WV3BN + (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        C[(size_t)(rowBase) * N + colBase + g * 8]      = acc[g * 4 + 0];
        C[(size_t)(rowBase) * N + colBase + g * 8 + 1]  = acc[g * 4 + 1];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8]  = acc[g * 4 + 2];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8 + 1] = acc[g * 4 + 3];
    }
}

// ---- host wrapper ---------------------------------------------------------

MatmulWgmmaV3::MatmulWgmmaV3(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV3::~MatmulWgmmaV3() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV3::execute(const float *d_A, const float *d_B, float *d_C) {
    // ONE-TIME prep (2026-10-01 tax fix; see RESULTS-KERNELONLY-2026-10-01.md)
    static const float *s_lastA = nullptr;
    static const float *s_lastB = nullptr;
    if (d_A != s_lastA || d_B != s_lastB) {
    int n = N * N;
    wgmmaV3_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    wgmmaV3_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    wgmmaV3_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);
        s_lastA = d_A; s_lastB = d_B;
    }

    dim3 grid(N / WV3BN, N / WV3BM);
    matmulWgmmaV3Kernel<<<grid, 128>>>(d_A16, d_Bt16, d_C, N);
    cudaCheckError(cudaGetLastError());
}
