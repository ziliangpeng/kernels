// TMA + 128B-swizzle descriptor sweep for wgmma k16 slices (v9 groundwork).
//
// Question: with TMA loading a 64x64 f16 K-major tile via SWIZZLE_128B
// (one bulk instruction), what (LBO, SBO, phase/base-offset) descriptor
// values make m64n64k16 wgmma correct for each of the four k16 slices?
//
// Method: same as wgmma_desc_sweep.cu (empiricism over spec-reading):
// CPU-verified N=64 single tile, one combo per process invocation.
//
// TMA facts used:
//  - cp.async.bulk.tensor writes the ASYNC proxy; mbarrier completion
//    orders it against wgmma reads (async-to-async) — no fence.proxy.async.
//  - SWIZZLE_128B pattern: box row = 128B (64 f16); row r's 16B chunk c
//    lands at c XOR (r % 8).
//  - CUtensorMap is passed by value as __grid_constant__ param.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA ERR %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define DCHK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(r, &s); printf("DRV ERR %s at %d\n", s, __LINE__); exit(1);} } while (0)

// ---- wgmma helpers (m64n64k16, 32 accs) -----------------------------------

__device__ __forceinline__ uint64_t make_desc_sw(const void *ptr, uint32_t lbo,
                                                 uint32_t sbo, uint32_t swizzle) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    d |= ((uint64_t)(swizzle & 0x3) << 62);   // bits 63-62: 1 = 128B swizzle
    return d;
}

#define WGMMA_M64N64K16(d, da, db)                                             \
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
        : "l"(da), "l"(db), "n"(1));

__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P;\n"
        "W:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D;\nbra W;\nD:\n}\n" :: "r"(addr), "r"(parity));
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap *map, void *dst,
                                            uint64_t *bar, int x, int y) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}

// ---- sweep kernel ----------------------------------------------------------
// One combo: TMA-load 64x64 A and Bt tiles (K-major f16, SWIZZLE_128B,
// box {64 k, 64 rows}), then 4x m64n64k16 wgmma over k slices 0/16/32/48
// with (LBO, SBO) from params and the slice start address advanced by
// k_slice*32B (2 chunks). 128 threads = 1 warpgroup.

__global__ __launch_bounds__(128) void sweepKernel(
    const __grid_constant__ CUtensorMap tmA,
    const __grid_constant__ CUtensorMap tmB,
    float *C, uint32_t lbo, uint32_t sbo) {

    // 64 rows x 64 k x 2B = 8KB per operand; SW128 box
    __shared__ __align__(1024) unsigned char sA[64 * 64 * 2];
    __shared__ __align__(1024) unsigned char sB[64 * 64 * 2];
    __shared__ __align__(8) uint64_t bar;

    const int tid = threadIdx.x;
    if (tid == 0) {
        mbar_init(&bar, 1);
        // A box (k=64, rows=64) at (x=0, y=0); B box same
        asm volatile("" ::: "memory");
        mbar_arrive_expect_tx(&bar, 2 * 64 * 128);
        tma_load_2d(&tmA, sA, &bar, 0, 0);
        tma_load_2d(&tmB, sB, &bar, 0, 0);
    }
    __syncthreads();
    mbar_wait(&bar, 0);

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    // 4 k16 slices; slice j starts 32B into each 128B swizzled row pattern.
    // The descriptor start address for slice j: base + j*32 bytes.
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint64_t da = make_desc_sw(sA + j * 32, lbo, sbo, 1);
        uint64_t db = make_desc_sw(sB + j * 32, lbo, sbo, 1);
        asm volatile("wgmma.fence.sync.aligned;\n");
        WGMMA_M64N64K16(acc, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
    }
    __syncthreads();

    // epilogue (v2-verified m64n64 mapping)
    const int w = tid / 32, l = tid % 32;
    const int rowBase = w * 16 + (l / 4);
    const int colBase = (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        C[(size_t)rowBase * 64 + colBase + g * 8]      = acc[g * 4 + 0];
        C[(size_t)rowBase * 64 + colBase + g * 8 + 1]  = acc[g * 4 + 1];
        C[(size_t)(rowBase + 8) * 64 + colBase + g * 8]  = acc[g * 4 + 2];
        C[(size_t)(rowBase + 8) * 64 + colBase + g * 8 + 1] = acc[g * 4 + 3];
    }
}

// ---- host -----------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: %s LBO SBO [N=64]\n", argv[0]); return 2; }
    const uint32_t lbo = atoi(argv[1]);
    const uint32_t sbo = atoi(argv[2]);
    const int N = argc > 3 ? atoi(argv[3]) : 64;

    std::vector<float> hA(N * N), hB(N * N), ref(N * N, 0.f);
    for (int i = 0; i < N * N; i++) { hA[i] = (float)((i * 37 + 11) % 97) / 32.f;
                                       hB[i] = (float)((i * 53 + 7) % 89) / 32.f; }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = hA[i * N + k];
            for (int j = 0; j < N; j++) ref[i * N + j] += a * hB[k * N + j];
        }

    // device fp16 copies (A row-major [m][k]; B row-major [k][n] -> we need
    // Bt [n][k] K-major for the B operand: transpose on host for the sweep)
    std::vector<__half> hA16(N * N), hBt16(N * N);
    for (int i = 0; i < N * N; i++) hA16[i] = __float2half(hA[i]);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < N; k++) hBt16[n * N + k] = __float2half(hB[k * N + n]);

    __half *dA, *dBt; float *dC;
    CHK(cudaMalloc(&dA, N * N * 2));
    CHK(cudaMalloc(&dBt, N * N * 2));
    CHK(cudaMalloc(&dC, N * N * 4));
    CHK(cudaMemcpy(dA, hA16.data(), N * N * 2, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dBt, hBt16.data(), N * N * 2, cudaMemcpyHostToDevice));

    // tensor maps: 2D, global dims {k-major inner = N, outer = N}
    // A is [m][k]: inner dim = k (N elements), outer = m
    CUtensorMap tmA, tmB;
    cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)N};
    cuuint64_t strides[1] = {(cuuint64_t)N * 2};   // row pitch
    cuuint32_t box[2] = {64, 64};                  // 64 k (128B) x 64 rows
    cuuint32_t estr[2] = {1, 1};
    DCHK(cuTensorMapEncodeTiled(&tmA, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                dA, dims, strides, box, estr,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DCHK(cuTensorMapEncodeTiled(&tmB, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                dBt, dims, strides, box, estr,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CHK(cudaMemset(dC, 0, N * N * 4));
    sweepKernel<<<1, 128>>>(tmA, tmB, dC, lbo, sbo);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("LBO=%u SBO=%u : KERNEL ERR %s\n", lbo, sbo, cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> got(N * N);
    CHK(cudaMemcpy(got.data(), dC, N * N * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (int i = 0; i < N * N; i++) {
        double d = std::fabs(got[i] - ref[i]) / std::max(1.0, (double)std::fabs(ref[i]));
        if (d > maxrel) maxrel = d;
    }
    printf("LBO=%u SBO=%u : maxrel=%.3e %s\n", lbo, sbo, maxrel,
           maxrel < 2e-3 ? "PASS" : "");
    return 0;
}
