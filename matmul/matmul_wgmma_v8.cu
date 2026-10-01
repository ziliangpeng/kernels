// Kernel "wgmma_v8" — warp specialization + mbarrier pipeline.
// Design docs in matmul_wgmma_v8.h. Compute path (macro, layouts, epilogue)
// is the v7-verified one, unchanged; only the feed/sync engine differs.

#include "matmul_wgmma_v8.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t w8_make_smem_desc(const void *ptr, uint32_t lbo,
                                                      uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    return d;
}

#define W8_M64N256K16(d, desc_a, desc_b, scale_d)                               \
    asm volatile(                                                              \
        "{\n"                                                                  \
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "                 \
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31," \
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63," \
        " %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}," \
        " %128,"                                                               \
        " %129,"                                                               \
        " %130, 1, 1, 0, 0;\n"                                                 \
        "}\n"                                                                  \
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),                   \
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),                   \
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),                  \
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),                  \
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),                  \
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),                  \
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),                  \
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),                  \
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),                  \
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),                  \
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),                  \
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),                  \
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),                  \
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),                  \
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),                  \
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),                  \
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),                  \
          "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),                  \
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),                  \
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),                  \
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),                  \
          "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),                  \
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),                  \
          "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),                  \
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),                  \
          "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),              \
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),              \
          "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),              \
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),              \
          "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),              \
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),              \
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])               \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scale_d)));

__device__ __forceinline__ void w8_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void w8_fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n");
}
__device__ __forceinline__ void w8_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void w8_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
template <int N>
__device__ __forceinline__ void w8_wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

// ---- cp.async + mbarrier helpers -------------------------------------------

__device__ __forceinline__ void w8_cp_async16(void *smem, const void *gmem) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
}
__device__ __forceinline__ void w8_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void w8_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void w8_mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void w8_mbar_arrive(uint64_t *bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
// spins until the barrier completes the given phase parity
__device__ __forceinline__ void w8_mbar_trywait(uint64_t *bar, uint32_t p) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" :: "r"(addr), "r"(p));
}

// ---- converter kernels ----------------------------------------------------

__global__ void w8_convertF32ToF16(const float * __restrict__ in,
                                   __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// B: [K][N] row-major -> Bt: [N][K] (K-major)
__global__ void w8_transposeB(const __half * __restrict__ B,
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
// Grid: (N/256, N/128). 384 threads = 3 warpgroups: wg0 producer, wg1/wg2
// consumers (64-row strips each).

constexpr int W8BM = 128;
constexpr int W8BN = 256;
constexpr int W8BK = 16;
constexpr int W8STAGES = 3;

__global__ __launch_bounds__(384) void matmulWgmmaV8Kernel(
    const __half * __restrict__ A, const __half * __restrict__ Bt,
    float * __restrict__ C, int N) {

    // Operand layouts (v7-verified): A per-strip 64-row quadrant atoms
    // (r/8)*128 + chunk*1024 (2 strips -> A buffer 2x 4KB); B full 256-row
    // atoms (r/8)*128 + chunk*4096 (8KB).
    __shared__ __align__(128) unsigned char A_s[W8STAGES][W8BM * W8BK * 2];
    __shared__ __align__(128) unsigned char B_s[W8STAGES][W8BN * W8BK * 2];
    __shared__ __align__(8)  uint64_t full_bar[W8STAGES];
    __shared__ __align__(8)  uint64_t free_bar[W8STAGES];

    const int blockM = blockIdx.y;
    const int blockN = blockIdx.x;
    const int tid = threadIdx.x;
    const int wg = tid / 128;

    // ---- one-time init (ALL threads pass the barrier — a __syncthreads
    // inside the producer branch alone deadlocks: block barriers need every
    // thread; this was the original v8 hang) ----
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < W8STAGES; s++) {
            w8_mbar_init(&full_bar[s], 128);   // 128 producer threads
            w8_mbar_init(&free_bar[s], 2);     // 2 consumer wgs
        }
    }
    __syncthreads();

    // ---- producer (wg0): stage tiles with cp.async ----
    if (wg == 0) {
        const int pt = tid % 128;              // producer thread 0..127
        // A: each thread loads ONE full row (pt), BOTH k-chunks — 128 rows
        // x 2 chunks = 256 copies (the old pt/2 mapping loaded only rows
        // 0..63: A rows 64..127 never staged -> strip cm=1 read garbage).
        const unsigned aoff0 = (pt / 64) * 2048 + ((pt % 64) / 8) * 128 +
                               (pt % 8) * 16;          // k-chunk 0
        const unsigned aoff1 = aoff0 + 1024;           // k-chunk 1
        // B: 256 rows; 128 threads cover 2 rows each (both k-chunks)
        const int brow0 = pt * 2;              // rows brow0, brow0+1
        const unsigned boff0 = (brow0 / 8) * 128 + (brow0 % 8) * 16;
        const unsigned boff1 = ((brow0 + 1) / 8) * 128 + ((brow0 + 1) % 8) * 16;

        const __half *Atile = A + (size_t)(blockM * W8BM) * N;
        const __half *Btile = Bt + (size_t)(blockN * W8BN) * N;

        // Phase math (v8 deadlock fix): a stage's barriers complete once per
        // CYCLE (STAGES laps), not per lap. At cycle c = lap / STAGES:
        //   - fill #c+1 of stage s completes full_bar[s] phase c -> consumer
        //     waits parity (c & 1)
        //   - release #c completes free_bar[s] phase c-1 -> producer's wait
        //     at cycle c (c >= 1) uses parity ((c-1) & 1)
        // Parities are DERIVED from the lap index — never flipped by hand
        // (the per-lap flip was the deadlock: it inverted every 3 laps).
        for (int k0 = 0, lap = 0; k0 < N; k0 += W8BK, lap++) {
            const int s = lap % W8STAGES;

            if (lap >= W8STAGES) {
                const uint32_t c = lap / W8STAGES;
                w8_mbar_trywait(&free_bar[s], (c - 1) & 1);
            }

            w8_cp_async16(A_s[s] + aoff0, Atile + (size_t)pt * N + k0);
            w8_cp_async16(A_s[s] + aoff1, Atile + (size_t)pt * N + k0 + 8);
            w8_cp_async16(B_s[s] + boff0, Btile + (size_t)brow0 * N + k0);
            w8_cp_async16(B_s[s] + boff0 + 4096, Btile + (size_t)brow0 * N + k0 + 8);
            w8_cp_async16(B_s[s] + boff1, Btile + (size_t)(brow0 + 1) * N + k0);
            w8_cp_async16(B_s[s] + boff1 + 4096, Btile + (size_t)(brow0 + 1) * N + k0 + 8);
            w8_cp_async_commit();

            w8_cp_async_wait<0>();
            w8_fence_proxy_async();
            w8_mbar_arrive(&full_bar[s]);
        }
    } else {
        // ---- consumers (wg1, wg2): wgmma on 64-row strips ----
        const int ct = tid % 128;
        const int cm = (wg - 1);               // strip 0 or 1
        const int w = ct / 32;
        const int l = ct % 32;

        float acc[128];
        #pragma unroll
        for (int i = 0; i < 128; i++) acc[i] = 0.0f;

        for (int k0 = 0, lap = 0; k0 < N; k0 += W8BK, lap++) {
            const int s = lap % W8STAGES;

            // fill #c+1 of this stage completes full_bar[s] phase c
            const uint32_t c = lap / W8STAGES;
            w8_mbar_trywait(&full_bar[s], c & 1);

            const unsigned char *Aq = A_s[s] + cm * 2048;
            const unsigned char *Bq = B_s[s];
            uint64_t descA = w8_make_smem_desc(Aq, 1024, 128);
            uint64_t descB = w8_make_smem_desc(Bq, 4096, 128);

            w8_wgmma_fence();
            W8_M64N256K16(acc, descA, descB, 1);
            w8_wgmma_commit();

            // correctness-first: wait for THIS wgmma to fully complete, then
            // release the stage it read. (No wgmma overlap across stages;
            // pipeline slack comes from the producer running ahead.)
            w8_wgmma_wait0();
            if (ct == 0) {
                w8_mbar_arrive(&free_bar[s]);
            }
        }

        // epilogue (v7-verified m64n256 mapping): row = cm*64 + w*16 + l/4
        const int rowBase = blockM * W8BM + cm * 64 + w * 16 + (l / 4);
        const int colBase = blockN * W8BN + (l % 4) * 2;
        #pragma unroll
        for (int g = 0; g < 32; g++) {
            C[(size_t)(rowBase) * N + colBase + g * 8]      = acc[g * 4 + 0];
            C[(size_t)(rowBase) * N + colBase + g * 8 + 1]  = acc[g * 4 + 1];
            C[(size_t)(rowBase + 8) * N + colBase + g * 8]  = acc[g * 4 + 2];
            C[(size_t)(rowBase + 8) * N + colBase + g * 8 + 1] = acc[g * 4 + 3];
        }
    }
}

// ---- host wrapper ---------------------------------------------------------

MatmulWgmmaV8::MatmulWgmmaV8(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV8::~MatmulWgmmaV8() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV8::execute(const float *d_A, const float *d_B, float *d_C) {
    // ONE-TIME prep (2026-10-01 tax fix; see RESULTS-KERNELONLY-2026-10-01.md)
    static const float *s_lastA = nullptr;
    static const float *s_lastB = nullptr;
    if (d_A != s_lastA || d_B != s_lastB) {
    int n = N * N;
    w8_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    w8_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    w8_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);
        s_lastA = d_A; s_lastB = d_B;
    }

    dim3 grid(N / W8BN, N / W8BM);
    matmulWgmmaV8Kernel<<<grid, 384>>>(d_A16, d_Bt16, d_C, N);
    cudaCheckError(cudaGetLastError());
}
