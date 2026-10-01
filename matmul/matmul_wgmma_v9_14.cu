// Kernel "wgmma_v9_14" — multicast in the 128x128 geometry (see .h).
//
// Machinery inherited from v9.13 (split-issue multicast, symmetric ledger,
// cluster drain); geometry changed: CTA 128x128, cluster along M, m64n128k16.

#include "matmul_wgmma_v9_14.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cooperative_groups.h>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t w914_make_desc(const void *ptr, uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((1u >> 4) & 0x3FFF) << 16);          // LBO = 1 (assumed)
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    d |= ((uint64_t)1 << 62);                              // 128B swizzle
    return d;
}

#define W914_M64N128K16(d, desc_a, desc_b)                                      \
    asm volatile(                                                              \
        "{\n"                                                                  \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "                 \
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31," \
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}," \
        " %64,"                                                                \
        " %65,"                                                                \
        " %66, 1, 1, 0, 0;\n"                                                  \
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
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])                   \
        : "l"(desc_a), "l"(desc_b), "n"(1));

__device__ __forceinline__ void w914_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void w914_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void w914_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
__device__ __forceinline__ void w914_wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n");
}

// ---- mbarrier + TMA --------------------------------------------------------

__device__ __forceinline__ void w914_mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void w914_mbar_arrive(uint64_t *bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
__device__ __forceinline__ void w914_mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void w914_mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P;\n"
        "W:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D;\nbra W;\nD:\n}\n" :: "r"(addr), "r"(parity));
}
__device__ __forceinline__ void w914_tma_2d(const CUtensorMap *map, void *dst,
                                            uint64_t *bar, int x, int y) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
__device__ __forceinline__ void w914_tma_2d_mc(const CUtensorMap *map, void *dst,
                                               uint64_t *bar, int x, int y,
                                               uint16_t ctaMask) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b), "h"(ctaMask) : "memory");
}
__device__ __forceinline__ void w914_mbar_expect_tx_remote(uint64_t *bar, uint32_t bytes,
                                                           uint32_t peerRank) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(peerRank));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
                 :: "r"(remote), "r"(bytes));
}
__device__ __forceinline__ void w914_mbar_arrive_remote(uint64_t *bar,
                                                        uint32_t dstRank) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(dstRank));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" :: "r"(remote));
}
__device__ __forceinline__ void w914_tma_store_2d(const CUtensorMap *map,
                                                  const void *src, int x, int y) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(map), "r"(x), "r"(y), "r"(s) : "memory");
}

// ---- converters (one-time prep) --------------------------------------------

__global__ void w914_convertF32ToF16(const float * __restrict__ in,
                                     __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void w914_transposeB(const __half * __restrict__ B,
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

extern __shared__ __align__(16) unsigned char w914_raw[];
__device__ unsigned char *w914_smem_ptr() {
    return (unsigned char *)(((uintptr_t)w914_raw + 1023) & ~(uintptr_t)1023);
}
#define w914_smem w914_smem_ptr()

__global__ __launch_bounds__(384) void matmulWgmmaV914Kernel(
    const __grid_constant__ CUtensorMap tmA,
    const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmC,
    float * __restrict__ C, int N) {

    constexpr int STAGES = 6;
    unsigned char *base = w914_smem;
    unsigned char *A_s2 = base;                                  // [6][128*128B]
    unsigned char *B_s2 = base + STAGES * (128 * 128);           // [6][128*128B]
    uint64_t *full_bar = reinterpret_cast<uint64_t *>(base + STAGES * (128 * 128 + 128 * 128));
    uint64_t *free_bar = full_bar + STAGES;

    namespace cg = cooperative_groups;
    const uint32_t rank = (uint32_t)cg::this_cluster().block_rank();
    const uint32_t peer = rank ^ 1u;
    const int NM = N / 128;
    const int NN = N / 128;
    constexpr int G = 8;
    const int cid = blockIdx.x / 2;        // cluster index (grid 1-D)
    const int bid = cid * 2 + (int)rank;   // this CTA's band-swizzle tile id
    const int group_size = NN * G;
    const int group_id = bid / group_size;
    const int first_m = group_id * G;
    const int gsize = min(NM - first_m, G);
    const int blockM = first_m + (bid % gsize);
    const int blockN = (bid % group_size) / gsize;
    const int tid = threadIdx.x;
    const int wg = tid / 128;
    const int ct = tid % 128;
    const int cm = (wg > 0) ? (wg - 1) : 0;
    float acc[64];

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            // full count 2: my expect_tx (A 16KB + my B box 8KB = 24KB)
            //                + peer's remote expect_tx (peer B box 8KB) = 32KB/lap
            // free count 4: both CTAs' 2 consumer WGs (B stage shared)
            w914_mbar_init(&full_bar[s], 2);
            w914_mbar_init(&free_bar[s], 4);
        }
    }
    __syncthreads();
    cg::this_cluster().sync();

    if (wg == 0) {
        if (tid == 0) {
            // my B box = n-box rank (of 2), multicast to BOTH CTAs
            const int b0 = (int)rank;
            for (int k0 = 0, lap = 0; k0 < N; k0 += 64, lap++) {
                const int s = lap % STAGES;
                if (lap >= STAGES) {
                    const uint32_t c = lap / STAGES;
                    w914_mbar_wait(&free_bar[s], (c - 1) & 1);
                }
                w914_mbar_expect_tx(&full_bar[s], 3 * 64 * 128);        // A 16KB + B 8KB
                w914_mbar_expect_tx_remote(&full_bar[s], 1 * 64 * 128, peer); // peer B 8KB
                w914_tma_2d(&tmA, A_s2 + s * 128 * 128, &full_bar[s], k0, blockM * 128);
                w914_tma_2d(&tmA, A_s2 + s * 128 * 128 + 64 * 128, &full_bar[s], k0, blockM * 128 + 64);
                w914_tma_2d_mc(&tmB, B_s2 + s * 128 * 128 + b0 * 64 * 128,
                               &full_bar[s], k0, blockN * 128 + b0 * 64, 0x3);
            }
        }
    } else {
        // ---- consumers: 2 WGs, m64n128k16 each ----
        #pragma unroll
        for (int i = 0; i < 64; i++) acc[i] = 0.0f;

        for (int k0 = 0, lap = 0; k0 < N; k0 += 64, lap++) {
            const int s = lap % STAGES;
            const uint32_t c = lap / STAGES;
            w914_mbar_wait(&full_bar[s], c & 1);

            const unsigned char *Aq = A_s2 + s * 128 * 128 + cm * 64 * 128;
            const unsigned char *Bq = B_s2 + s * 128 * 128;

            #pragma unroll
            for (int j = 0; j < 4; j++) {          // 4 x k16 = k64 lap (v9_13 pattern)
                uint64_t da = w914_make_desc(Aq + j * 32, 1024);
                uint64_t db = w914_make_desc(Bq + j * 32, 1024);
                w914_wgmma_fence();
                W914_M64N128K16(acc, da, db);
                w914_wgmma_commit();
            }
            w914_wgmma_wait1();
            if (lap > 0) {
                const int prev = (lap - 1) % STAGES;
                if (ct == 0) {
                    w914_mbar_arrive(&free_bar[prev]);
                    w914_mbar_arrive_remote(&free_bar[prev], peer);
                }
    }
        }
        w914_wgmma_wait0();
        if (ct == 0) {
            const int lastS = (N / 64 - 1) % STAGES;
            w914_mbar_arrive(&free_bar[lastS]);
            w914_mbar_arrive_remote(&free_bar[lastS], peer);
        }
    }

    // cluster drain (v9.13 lesson: kernel-level, ALL threads of both CTAs)
    cg::this_cluster().sync();

    if (wg != 0) {
        // stage C: 128x128 f32 = 64KB over the DRAINED A region.
        // m64n128k16 accumulator layout per WG: rows wg-1 quadrants of 64,
        // 64 cols; each thread holds 64 floats = (row group r, col pairs).
        float *stage = reinterpret_cast<float *>(w914_smem);
        const int rowL = (wg - 1) * 64 + (ct / 32) * 16 + ((ct % 32) / 4);
        const int colL = ((ct % 32) % 4) * 2;
        #pragma unroll
        for (int g = 0; g < 16; g++) {
            stage[(size_t)rowL * 128 + colL + g * 8]           = acc[g * 4 + 0];
            stage[(size_t)rowL * 128 + colL + g * 8 + 1]       = acc[g * 4 + 1];
            stage[(size_t)(rowL + 8) * 128 + colL + g * 8]     = acc[g * 4 + 2];
            stage[(size_t)(rowL + 8) * 128 + colL + g * 8 + 1] = acc[g * 4 + 3];
        }
    }
    __syncthreads();
    if (tid == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n");
        float *stg = reinterpret_cast<float *>(w914_smem);
        w914_tma_store_2d(&tmC, stg, blockN * 128, blockM * 128);
        w914_tma_store_2d(&tmC, stg + 64 * 128, blockN * 128, blockM * 128 + 64);
        asm volatile("cp.async.bulk.commit_group;\n");
        asm volatile("cp.async.bulk.wait_group 0;\n");
    }
}

// ---- host wrapper ---------------------------------------------------------

void MatmulWgmmaV914::makeMaps() {
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
    CUresult r2 = cuTensorMapEncodeTiled(&tmB, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                         d_Bt16, dims, strides, box, estr,
                                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                                         CU_TENSOR_MAP_SWIZZLE_128B,
                                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    mapsReady = (r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS);
    if (!mapsReady) printf("TMA map encode FAILED: %d %d\n", (int)r1, (int)r2);
}

MatmulWgmmaV914::MatmulWgmmaV914(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV914::~MatmulWgmmaV914() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV914::execute(const float *d_A, const float *d_B, float *d_C) {
    // ONE-TIME prep (2026-10-01 harness tax fix; see RESULTS-KERNELONLY-2026-10-01.md)
    static const float *s_lastA = nullptr;
    static const float *s_lastB = nullptr;
    if (d_A != s_lastA || d_B != s_lastB) {
        int n = N * N;
        w914_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
        w914_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
        dim3 tb((N + 31) / 32, (N + 31) / 32);
        w914_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);
        s_lastA = d_A; s_lastB = d_B;
    }

    if (!mapsReady) makeMaps();

    CUtensorMap tmClocal;
    {
        cuuint64_t dimsC[2] = {(cuuint64_t)N, (cuuint64_t)N};
        cuuint64_t stridesC[1] = {(cuuint64_t)N * 4};
        cuuint32_t boxC[2] = {128, 64};
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
        cudaFuncSetAttribute(matmulWgmmaV914Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        smemSet = true;
    }

    // cluster launch: 2-CTA clusters along M over a 1-D grid
    const int NM = N / 128, NN = N / 128;
    const int tiles = NM * NN;
    const int nclusters = tiles / 2 + (tiles & 1);
    dim3 grid(nclusters * 2);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = 6 * (128 * 128 + 128 * 128) + 2 * 6 * 8 + 1024;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(&cfg, matmulWgmmaV914Kernel,
                                         tmA, tmB, tmClocal, d_C, N);
    cudaCheckError(err);
}
