// Kernel "wgmma_v9_10" — persistent cooperative + async epilogue (Fable #2).
//
// v9.8's persistent skeleton but WITHOUT the full-CTA double-__syncthreads
// drain (its negative-result cause). Instead: per-consumer-wg 32KB staging
// slots OUTSIDE the pipeline region; epilogue stages a 32-row chunk, issues
// a bulk tensor store, waits that one store, stages the next chunk — the
// producer never stops (pipeline stays full across tile boundaries), and
// the wave-quantization tail disappears (132 persistent CTAs).
//
// Stage ring: 3 x 48KB = 144KB. Slots: 2 x 32KB = 64KB. Total 208KB <= 227.
// Barrier parity uses the global lap counter across tiles (v9.8-proven).
// free_bar count = 2 (both consumer wgs arrive per stage, cooperative).

#include "matmul_wgmma_v9_10.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t w910_make_desc(const void *ptr, uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((1u >> 4) & 0x3FFF) << 16);          // LBO = 1 (assumed)
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    d |= ((uint64_t)1 << 62);                              // 128B swizzle
    return d;
}

#define W910_M64N256K16(d, desc_a, desc_b)                                      \
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

__device__ __forceinline__ void w910_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void w910_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void w910_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
__device__ __forceinline__ void w910_wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n");
}

// ---- mbarrier + TMA --------------------------------------------------------

__device__ __forceinline__ void w910_mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void w910_mbar_arrive(uint64_t *bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
__device__ __forceinline__ void w910_mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void w910_mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P;\n"
        "W:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D;\nbra W;\nD:\n}\n" :: "r"(addr), "r"(parity));
}
__device__ __forceinline__ void w910_tma_2d(const CUtensorMap *map, void *dst,
                                            uint64_t *bar, int x, int y) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
__device__ __forceinline__ void w910_tma_store_2d(const CUtensorMap *map,
                                                  const void *src, int x, int y) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(map), "r"(x), "r"(y), "r"(s) : "memory");
}

// ---- converter kernels ----------------------------------------------------

__global__ void w910_convertF32ToF16(const float * __restrict__ in,
                                     __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void w910_transposeB(const __half * __restrict__ B,
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

__device__ int w910_use_boundary = 0;

extern __shared__ __align__(16) unsigned char w910_raw[];
__device__ unsigned char *w910_smem_ptr() {
    return (unsigned char *)(((uintptr_t)w910_raw + 1023) & ~(uintptr_t)1023);
}
#define w910_smem w910_smem_ptr()

__global__ __launch_bounds__(384, 1) void matmulWgmmaV910Kernel(
    const __grid_constant__ CUtensorMap tmA,
    const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmC,
    float * __restrict__ C, int N, int G) {

    constexpr int STAGES = 3;
    // layout: [3 stages x (A 16KB + B 32KB)] [slot wg1 32KB] [slot wg2 32KB] [bars]
    unsigned char *A_s = w910_smem;                            // [3][128*128B]
    unsigned char *B_s = w910_smem + STAGES * (128 * 128);     // [3][256*128B]
    float *slots = reinterpret_cast<float *>(w910_smem + STAGES * (128 * 128 + 256 * 128));
    uint64_t *full_bar = reinterpret_cast<uint64_t *>(
        w910_smem + STAGES * (128 * 128 + 256 * 128) + 2 * 32 * 1024);
    uint64_t *free_bar = full_bar + STAGES;

    const int tid = threadIdx.x;
    const int wg = tid / 128;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            w910_mbar_init(&full_bar[s], 1);
            w910_mbar_init(&free_bar[s], 2);
        }
    }
    __syncthreads();

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
            // ---- producer: global lap counter across tiles ----
            if (tid == 0) {
                for (int k0 = 0; k0 < N; k0 += 64) {
                    const long lap = (long)i * klaps + k0 / 64;
                    const int s = (int)(lap % STAGES);
                    if (lap >= STAGES) {
                        const uint32_t c = (uint32_t)(lap / STAGES);
                        w910_mbar_wait(&free_bar[s], (c - 1) & 1);
                    }
                    w910_mbar_expect_tx(&full_bar[s], 6 * 64 * 128);
                    w910_tma_2d(&tmA, A_s + s * 128 * 128, &full_bar[s], k0, blockM * 128);
                    w910_tma_2d(&tmA, A_s + s * 128 * 128 + 64 * 128, &full_bar[s], k0, blockM * 128 + 64);
                    w910_tma_2d(&tmB, B_s + s * 256 * 128, &full_bar[s], k0, blockN * 256);
                    w910_tma_2d(&tmB, B_s + s * 256 * 128 + 64 * 128, &full_bar[s], k0, blockN * 256 + 64);
                    w910_tma_2d(&tmB, B_s + s * 256 * 128 + 128 * 128, &full_bar[s], k0, blockN * 256 + 128);
                    w910_tma_2d(&tmB, B_s + s * 256 * 128 + 192 * 128, &full_bar[s], k0, blockN * 256 + 192);
                }
            }
        } else {
            // ---- consumers: cooperative on this tile's laps ----
            // DISCRIMINATOR (Fable follow-up, experiment 2): consumers-only
            // named barrier (id 4, both consumer wgs = 256 threads) at each
            // tile boundary. If the N=3072 deadlock is the cross-WG
            // phase-merge ABA (wg racing into tile i+1's mainloop while the
            // peer wg still owes arrives for tile i), this barrier removes
            // it. Runtime-switched via W910_BOUNDARY=1 (A/B in one binary).
            if (i > 0 && w910_use_boundary) {
                asm volatile("bar.sync 4, 256;\n");
            }
            const int ct = tid % 128;
            const int cm = wg - 1;
            const int w = ct / 32;
            const int l = ct % 32;

            float acc[128];
            #pragma unroll
            for (int x = 0; x < 128; x++) acc[x] = 0.0f;

            for (int k0 = 0; k0 < N; k0 += 64) {
                const long lap = (long)i * klaps + k0 / 64;
                const int s = (int)(lap % STAGES);
                const uint32_t c = (uint32_t)(lap / STAGES);
                w910_mbar_wait(&full_bar[s], c & 1);

                const unsigned char *Aq = A_s + s * 128 * 128 + cm * 64 * 128;
                const unsigned char *Bq = B_s + s * 256 * 128;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint64_t da = w910_make_desc(Aq + j * 32, 1024);
                    uint64_t db = w910_make_desc(Bq + j * 32, 1024);
                    w910_wgmma_fence();
                    W910_M64N256K16(acc, da, db);
                    w910_wgmma_commit();
                }
                w910_wgmma_wait1();
                // release the PREVIOUS lap's stage. For lap == i*klaps (the
                // first lap of tile i>0) this releases the previous tile's
                // last stage — exactly-once across tiles (the old per-tile
                // tail release double-arrived and corrupted the barrier).
                if (lap > 0 || i > 0) {
                    const int prev = (int)((lap - 1) % STAGES);
                    if (ct == 0) w910_mbar_arrive(&free_bar[prev]);
                }
            }
            w910_wgmma_wait0();

            // ---- async epilogue into this wg's own 32KB slot ----
            // m64n256 acc mapping (v7-verified): thread (w,l) owns rows
            // r0 = w*16 + l/4 and r0+8, cols (l%4)*2 + g*8, g=0..31.
            // Warps 0,1 own chunk 0 (strip rows 0..31), warps 2,3 chunk 1
            // (rows 32..63). The 32KB slot is REUSED sequentially: stage
            // chunk -> wg barrier -> leader stores + waits -> wg barrier.
            // Producer and the other wg never wait on this (no CTA drain).
            float *slot = slots + cm * (32 * 1024 / 4);
            for (int ch = 0; ch < 2; ch++) {
                if (w / 2 == ch) {
                    const int r0 = w * 16 + (l / 4) - ch * 32;
                    const int colL = (l % 4) * 2;
                    #pragma unroll
                    for (int g = 0; g < 32; g++) {
                        slot[(size_t)r0 * 256 + colL + g * 8]           = acc[g * 4 + 0];
                        slot[(size_t)r0 * 256 + colL + g * 8 + 1]       = acc[g * 4 + 1];
                        slot[(size_t)(r0 + 8) * 256 + colL + g * 8]     = acc[g * 4 + 2];
                        slot[(size_t)(r0 + 8) * 256 + colL + g * 8 + 1] = acc[g * 4 + 3];
                    }
                }
                // wg-scoped named barrier (id = wg, 128 threads)
                asm volatile("bar.sync %0, 128;\n" :: "r"(wg));
                if (l == 0 && w == 2 * ch) {   // leader of this chunk's first warp
                    asm volatile("fence.proxy.async.shared::cta;\n");
                    w910_tma_store_2d(&tmC, slot, blockN * 256,
                                      blockM * 128 + cm * 64 + ch * 32);
                    asm volatile("cp.async.bulk.commit_group;\n");
                    asm volatile("cp.async.bulk.wait_group 0;\n");
                }
                asm volatile("bar.sync %0, 128;\n" :: "r"(wg));
            }

        }
    }
    // after this CTA's LAST tile: release its final lap's stage (the
    // in-loop release only covers laps 1..n of each tile plus each tile's
    // first lap; the very last lap of the very last tile is released here —
    // exactly once, global-lap indexed)
    if (wg != 0 && (tid % 128) == 0 && blockIdx.x < tiles) {
        const long nTilesMine = (long)((tiles - 1 - blockIdx.x) / gridDim.x) + 1;
        const long lastLap = (nTilesMine - 1) * klaps + (klaps - 1);
        w910_mbar_arrive(&free_bar[(int)(lastLap % STAGES)]);
    }
}

// ---- host wrapper ---------------------------------------------------------

void MatmulWgmmaV910::makeMaps() {
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

MatmulWgmmaV910::MatmulWgmmaV910(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV910::~MatmulWgmmaV910() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV910::execute(const float *d_A, const float *d_B, float *d_C) {
    static int boundarySet = -1;
    if (boundarySet < 0) {
        int b = getenv("W910_BOUNDARY") ? atoi(getenv("W910_BOUNDARY")) : 0;
        cudaMemcpyToSymbol(w910_use_boundary, &b, sizeof(int));
        boundarySet = b;
        printf("w910: W910_BOUNDARY=%d\n", b);
    }
    int n = N * N;
    w910_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    w910_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    w910_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);

    if (!mapsReady) makeMaps();

    // tmC: [m][n] row-major f32. Box {256 cols, 32 rows} (32-row chunks).
    CUtensorMap tmClocal;
    {
        cuuint64_t dimsC[2] = {(cuuint64_t)N, (cuuint64_t)N};
        cuuint64_t stridesC[1] = {(cuuint64_t)N * 4};
        cuuint32_t boxC[2] = {256, 32};
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
        cudaFuncSetAttribute(matmulWgmmaV910Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        smemSet = true;
    }

    const int tiles = (N / 128) * (N / 256);
    const int nctas = tiles < 132 ? tiles : 132;
    static int gG = -1;
    if (gG < 0) {
        const char *e = getenv("WG_G");
        gG = e ? atoi(e) : 8;
        if (gG < 1) gG = 1;
    }
    dim3 grid(nctas);
    // 3 x (16KB A + 32KB B) + 2 x 32KB slots + barriers + align slack
    const int smem = 3 * (128 * 128 + 256 * 128) + 2 * 32 * 1024 + 6 * 8 + 1024;
    matmulWgmmaV910Kernel<<<grid, 384, smem>>>(tmA, tmB, tmClocal, d_C, N, gG);
    cudaCheckError(cudaGetLastError());
}
