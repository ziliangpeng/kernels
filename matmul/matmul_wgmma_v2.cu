// Kernel "wgmma_v2" — Hopper WGMMA, m64n64k16, single warpgroup.
// Full docs in matmul_wgmma_v2.h. Snapshot of the v2 lineage (commits
// bbf2d68..1c6ee60); see matmul/RESULTS.md rung 9c for the measurement
// narrative.

#include "matmul_wgmma_v2.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t wgmma2_make_smem_desc(const void *ptr, uint32_t lbo,
                                                          uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    return d;
}

#define WGMMA2_M64N64K16(d, desc_a, desc_b, scale_d)                            \
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

__device__ __forceinline__ void wgmma2_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma2_fence_proxy_async() {
    // Orders prior SMEM writes (generic proxy) against subsequent wgmma
    // async-proxy reads. Required by PTX; __syncthreads does NOT order
    // the async proxy.
    asm volatile("fence.proxy.async.shared::cta;\n");
}
__device__ __forceinline__ void wgmma2_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma2_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

// ---- converter kernels ----------------------------------------------------

__global__ void wgmma2_convertF32ToF16(const float * __restrict__ in,
                                       __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// B: [K][N] row-major -> Bt: [N][K] (K-major)
__global__ void wgmma2_transposeB(const __half * __restrict__ B,
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
// Grid: (N/64, N/64). One warpgroup (128 threads) per 64x64 output tile.

constexpr int W2BM = 64;
constexpr int W2BN = 64;
constexpr int W2BK = 16;

__global__ __launch_bounds__(128) void matmulWgmmaV2Kernel(
    const __half * __restrict__ A, const __half * __restrict__ Bt,
    float * __restrict__ C, int N) {

    // Canonical no-swizzle interleave (sweep-verified): core matrix = 8x8
    // halfs (128B) contiguous; atom (mi, ki) at mi*128 + ki*1024 bytes.
    __shared__ __align__(128) unsigned char A_s[W2BM * W2BK * 2];
    __shared__ __align__(128) unsigned char B_s[W2BN * W2BK * 2];

    const int blockM = blockIdx.y;
    const int blockN = blockIdx.x;
    const int tid = threadIdx.x;

    // load mapping (both tiles 64x16): row=tid/2, 8-half chunk=tid%2
    const int aRow = tid / 2;
    const int aCol = (tid % 2) * 8;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    const __half *Atile = A + (size_t)(blockM * W2BM) * N;
    const __half *Btile = Bt + (size_t)(blockN * W2BN) * N;

    for (int k0 = 0; k0 < N; k0 += W2BK) {
        {
            const uint4 *src = reinterpret_cast<const uint4 *>(Atile + (size_t)aRow * N + k0 + aCol);
            uint4 *dst = reinterpret_cast<uint4 *>(A_s + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
            *dst = *src;
        }
        {
            const uint4 *src = reinterpret_cast<const uint4 *>(Btile + (size_t)aRow * N + k0 + aCol);
            uint4 *dst = reinterpret_cast<uint4 *>(B_s + (aRow / 8) * 128 + (tid % 2) * 1024 + (aRow % 8) * 16);
            *dst = *src;
        }
        __syncthreads();
        wgmma2_fence_proxy_async();

        uint64_t descA = wgmma2_make_smem_desc(A_s, 1024, 128);
        uint64_t descB = wgmma2_make_smem_desc(B_s, 1024, 128);
        wgmma2_fence();
        WGMMA2_M64N64K16(acc, descA, descB, 1);
        wgmma2_commit();
        wgmma2_wait0();

        __syncthreads();
    }

    // epilogue (sweep-verified m64n64): frag_row = w*16 + lane/4,
    // frag_col = (lane%4)*2; 8 col groups x 4 regs.
    const int w = tid / 32;
    const int l = tid % 32;
    const int rowBase = blockM * W2BM + w * 16 + (l / 4);
    const int colBase = blockN * W2BN + (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        C[(size_t)(rowBase) * N + colBase + g * 8]      = acc[g * 4 + 0];
        C[(size_t)(rowBase) * N + colBase + g * 8 + 1]  = acc[g * 4 + 1];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8]  = acc[g * 4 + 2];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8 + 1] = acc[g * 4 + 3];
    }
}

// ---- host wrapper ---------------------------------------------------------

MatmulWgmmaV2::MatmulWgmmaV2(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV2::~MatmulWgmmaV2() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV2::execute(const float *d_A, const float *d_B, float *d_C) {
    int n = N * N;
    wgmma2_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    wgmma2_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    wgmma2_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);

    dim3 grid(N / W2BN, N / W2BM);
    matmulWgmmaV2Kernel<<<grid, 128>>>(d_A16, d_Bt16, d_C, N);
    cudaCheckError(cudaGetLastError());
}
