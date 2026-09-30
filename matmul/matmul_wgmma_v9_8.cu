// Kernel "wgmma_v9_8" — persistent CTAs (132, stride over tiles) on the
// tensor stores. Single variable vs v9.3: the C write path only.
// variable vs v9.2: CTA traversal order only).
// consumer overlap (wait<1>). Single-variable vs v9: pipeline depth only.
// Design in matmul_wgmma_v9.h. Descriptor: swizzle=128B, LBO=1, SBO=1024.

#include "matmul_wgmma_v9_8.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t w98_make_desc(const void *ptr, uint32_t sbo) {
    // swizzle mode 128B (bits 63-62 = 01), LBO assumed 1, SBO = 1024
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((1u >> 4) & 0x3FFF) << 16);          // LBO = 1 (assumed)
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    d |= ((uint64_t)1 << 62);                              // 128B swizzle
    return d;
}

// m64n256k16, 128 f32 accumulators — same as v7/v8 (programmatic list)
#define W98_M64N256K16(d, desc_a, desc_b)                                        \
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
        : "l"(desc_a), "l"(desc_b), "n"(1));

__device__ __forceinline__ void w98_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void w98_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void w98_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
__device__ __forceinline__ void w98_wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n");
}

// ---- mbarrier + TMA --------------------------------------------------------

__device__ __forceinline__ void w98_mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void w98_mbar_arrive(uint64_t *bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
__device__ __forceinline__ void w98_mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void w98_mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P;\n"
        "W:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D;\nbra W;\nD:\n}\n" :: "r"(addr), "r"(parity));
}
// one elected thread issues all TMA copies for a stage, tracked by one barrier
__device__ __forceinline__ void w98_tma_2d(const CUtensorMap *map, void *dst,
                                          uint64_t *bar, int x, int y) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}

// v9.5: bulk tensor STORE (SMEM -> GMEM), bulk-group completion
__device__ __forceinline__ void w98_tma_store_2d(const CUtensorMap *map,
                                                 const void *src, int x, int y) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(map), "r"(x), "r"(y), "r"(s) : "memory");
}

// ---- converter kernels ----------------------------------------------------

__global__ void w98_convertF32ToF16(const float * __restrict__ in,
                                   __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void w98_transposeB(const __half * __restrict__ B,
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
// Grid: (N/256, N/128). 384 threads: wg0 producer (TMA), wg1/wg2 consumers
// (m64n256k16 on 64-row strips). STAGES=3; A tile = 2 TMA boxes, B = 4.

constexpr int W9BM = 128;
constexpr int W9BN = 256;
constexpr int W9STAGES = 3;
// dynamic SMEM: A[3][128*64], B[3][256*64] halfs = 3*(16KB + 32KB) = 144KB
// too big -> stage A/B in ONE 64-k buffer each per stage but with 2-stage
// depth for B? Simplify: STAGES=2 (A 2x16KB + B 2x32KB = 96KB <= 100KB ok).

// Dynamic SMEM base is only 16B-aligned by default; SWIZZLE_128B TMA
// destinations must be 1024B-aligned — align manually (v9 bug #1: verify
// FAIL 0.55 with 327T — swizzle pattern misaligned vs the data).
extern __shared__ __align__(16) unsigned char w98_raw[];
__device__ unsigned char *w98_smem_ptr() {
    return (unsigned char *)(((uintptr_t)w98_raw + 1023) & ~(uintptr_t)1023);
}
#define w98_smem w98_smem_ptr()

__global__ __launch_bounds__(384) void matmulWgmmaV98Kernel(
    const __grid_constant__ CUtensorMap tmA,
    const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmC,
    float * __restrict__ C, int N, int G) {

    constexpr int STAGES = 4;
    unsigned char *A_s = w98_smem;                          // [4][128*128B]
    unsigned char *B_s = w98_smem + STAGES * (128 * 128);   // [4][256*128B]
    uint64_t *full_bar = reinterpret_cast<uint64_t *>(w98_smem + STAGES * (128 * 128 + 256 * 128));
    uint64_t *free_bar = full_bar + STAGES;

    const int tid = threadIdx.x;
    const int wg = tid / 128;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            w98_mbar_init(&full_bar[s], 1);   // one elected producer thread
            w98_mbar_init(&free_bar[s], 2);   // 2 consumer wgs
        }
    }
    __syncthreads();

    // v9.8: persistent CTAs. grid = min(132, tiles); each CTA strides over
    // tiles. Barrier parity uses a GLOBAL lap counter across tiles (i*(N/64)
    // + kIdx) so phase math survives tile boundaries. The epilogue staging
    // reuses the pipeline SMEM, so a DOUBLE __syncthreads brackets the bulk
    // store: the second one blocks the producer from refilling stage 0
    // (the staging region) until the store has retired.
    const int NM = N / 128;
    const int NN = N / 256;
    const int tiles = NM * NN;
    const int klaps = N / 64;

    for (int t = blockIdx.x, i = 0; t < tiles; t += gridDim.x, i++) {
        // band swizzle (v9.3 mapping, runtime G)
        const int group_size = NN * G;
        const int group_id = t / group_size;
        const int first_m = group_id * G;
        const int gsize = min(NM - first_m, G);
        const int blockM = first_m + (t % gsize);
        const int blockN = (t % group_size) / gsize;

        if (wg == 0) {
            // ---- producer: 6 TMA boxes per 64-k lap, global-lap parity ----
            if (tid == 0) {
                for (int k0 = 0; k0 < N; k0 += 64) {
                    const long lap = (long)i * klaps + k0 / 64;
                    const int s = (int)(lap % STAGES);
                    if (lap >= STAGES) {
                        const uint32_t c = (uint32_t)(lap / STAGES);
                        w98_mbar_wait(&free_bar[s], (c - 1) & 1);
                    }
                    w98_mbar_expect_tx(&full_bar[s], 6 * 64 * 128);
                    w98_tma_2d(&tmA, A_s + s * 128 * 128, &full_bar[s], k0, blockM * 128);
                    w98_tma_2d(&tmA, A_s + s * 128 * 128 + 64 * 128, &full_bar[s], k0, blockM * 128 + 64);
                    w98_tma_2d(&tmB, B_s + s * 256 * 128, &full_bar[s], k0, blockN * 256);
                    w98_tma_2d(&tmB, B_s + s * 256 * 128 + 64 * 128, &full_bar[s], k0, blockN * 256 + 64);
                    w98_tma_2d(&tmB, B_s + s * 256 * 128 + 128 * 128, &full_bar[s], k0, blockN * 256 + 128);
                    w98_tma_2d(&tmB, B_s + s * 256 * 128 + 192 * 128, &full_bar[s], k0, blockN * 256 + 192);
                }
            }
        } else {
            // ---- consumers: 4x m64n256k16 per 64-k lap ----
            const int ct = tid % 128;
            const int cm = wg - 1;
            const int w = ct / 32;
            const int l = ct % 32;

            float acc[64];
            #pragma unroll
            for (int x = 0; x < 64; x++) acc[x] = 0.0f;

            for (int k0 = 0; k0 < N; k0 += 64) {
                const long lap = (long)i * klaps + k0 / 64;
                const int s = (int)(lap % STAGES);
                const uint32_t c = (uint32_t)(lap / STAGES);
                w98_mbar_wait(&full_bar[s], c & 1);

                const unsigned char *Aq = A_s + s * 128 * 128 + cm * 64 * 128;
                const unsigned char *Bq = B_s + s * 256 * 128;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint64_t da = w98_make_desc(Aq + j * 32, 1024);
                    uint64_t db = w98_make_desc(Bq + j * 32, 1024);
                    w98_wgmma_fence();
                    W98_M64N256K16(acc, da, db);
                    w98_wgmma_commit();
                }
                w98_wgmma_wait1();
                if (lap > 0) {
                    const int prev = (int)((lap - 1) % STAGES);
                    if (ct == 0) w98_mbar_arrive(&free_bar[prev]);
                }
            }
            w98_wgmma_wait0();

            // stage acc to SMEM (aliases the pipeline region; guarded by the
            // double syncthreads below)
            float *stage = reinterpret_cast<float *>(w98_smem);
            const int rowL = cm * 64 + w * 16 + (l / 4);
            const int colL = (l % 4) * 2;
            #pragma unroll
            for (int g = 0; g < 32; g++) {
                stage[(size_t)rowL * 256 + colL + g * 8]           = acc[g * 4 + 0];
                stage[(size_t)rowL * 256 + colL + g * 8 + 1]       = acc[g * 4 + 1];
                stage[(size_t)(rowL + 8) * 256 + colL + g * 8]     = acc[g * 4 + 2];
                stage[(size_t)(rowL + 8) * 256 + colL + g * 8 + 1] = acc[g * 4 + 3];
            }
        }
        __syncthreads();                       // staging complete
        if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n");
            float *stg = reinterpret_cast<float *>(w98_smem);
            w98_tma_store_2d(&tmC, stg, blockN * 256, blockM * 128);
            w98_tma_store_2d(&tmC, stg + 64 * 256, blockN * 256, blockM * 128 + 64);
            asm volatile("cp.async.bulk.commit_group;\n");
            asm volatile("cp.async.bulk.wait_group 0;\n");
        }
        __syncthreads();                       // store retired -> producer may refill
    }
}

// ---- host wrapper ---------------------------------------------------------

void MatmulWgmmaV98::makeMaps() {
    // A: [m][k] row-major global (d_A16), inner = k. Box {64 k, 64 rows}.
    cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)N};
    cuuint64_t strides[1] = {(cuuint64_t)N * 2};
    cuuint32_t box[2] = {64, 64};
    cuuint32_t estr[2] = {1, 1};
    CUresult r1 = cuTensorMapEncodeTiled(&tmA, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                         d_A16, dims, strides, box, estr,
                                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                                         CU_TENSOR_MAP_SWIZZLE_128B,
                                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    // Bt: [n][k] row-major (d_Bt16), inner = k. Box {64 k, 64 rows}.
    CUresult r2 = cuTensorMapEncodeTiled(&tmB, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                         d_Bt16, dims, strides, box, estr,
                                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                                         CU_TENSOR_MAP_SWIZZLE_128B,
                                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    mapsReady = (r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS);
    if (!mapsReady) printf("TMA map encode FAILED: %d %d\n", (int)r1, (int)r2);
}

MatmulWgmmaV98::MatmulWgmmaV98(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV98::~MatmulWgmmaV98() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV98::execute(const float *d_A, const float *d_B, float *d_C) {
    int n = N * N;
    w98_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    w98_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    w98_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);

    if (!mapsReady) makeMaps();

    // tmC must be encoded per-execution (d_C is the runtime parameter)
    static CUtensorMap tmClocal;
    {
        cuuint64_t dimsC[2] = {(cuuint64_t)N, (cuuint64_t)N};
        cuuint64_t stridesC[1] = {(cuuint64_t)N * 4};
        cuuint32_t boxC[2] = {256, 64};
        cuuint32_t estrC[2] = {1, 1};
        CUresult r3 = cuTensorMapEncodeTiled(&tmClocal, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
                                             d_C, dimsC, stridesC, boxC, estrC,
                                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                                             CU_TENSOR_MAP_SWIZZLE_NONE,
                                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r3 != CUDA_SUCCESS) printf("tmC encode FAILED: %d\n", (int)r3);
    }

    static bool smemSet = false;
    if (!smemSet) {
        // v9.7: 4 stages x 48KB = 192KB + TMA epilogue; opt-in max 227KB
        cudaFuncSetAttribute(matmulWgmmaV98Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        smemSet = true;
    }

    // persistent: one CTA per SM (192KB SMEM each), stride over tiles
    const int tiles = (N / 128) * (N / 256);
    const int nctas = tiles < 132 ? tiles : 132;
    static int gG = -1;
    if (gG < 0) {
        const char *e = getenv("WG_G");
        gG = e ? atoi(e) : 8;
        if (gG < 1) gG = 1;
    }
    dim3 grid(nctas);
    const int smem = 4 * (128 * 64 * 2 + 256 * 64 * 2) + 4 * 2 * 8 + 1024;
    matmulWgmmaV98Kernel<<<grid, 384, smem>>>(tmA, tmB, tmClocal, d_C, N, gG);
    cudaCheckError(cudaGetLastError());
}
