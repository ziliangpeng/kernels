// Kernel "wgmma_v9_12" — 2-CTA cluster + TMA multicast on B (Fable #3).
//
// Cluster pairs = adjacent band-swizzle t (even t0, odd t0+1) share the
// same blockN (band order is n-major: consecutive t differ in m), so both
// CTAs need the SAME B tile every lap. The even CTA's producer issues B
// loads ONCE with multicast::cluster (both CTAs' SMEM written in one HBM
// transaction); each CTA loads its own A tiles unicast.
//
// Barrier protocol (no cluster-wide barrier): the MULTICAST CTA's producer
// does mbarrier.arrive.expect_tx on BOTH CTAs' full_bar[s] (remote arrive
// via mapa); consumers wait locally as in v9.7. Each CTA's free_bar[s] must
// be released by its OWN consumers only (count 2) — but the producing CTA
// waits on BOTH free_bars before reusing a B stage; the non-producing CTA
// waits only on its own (its A is loaded by... itself? No — A is unicast:
// each CTA's OWN producer thread loads its own A. But only one producer
// thread exists per CTA (wg0 tid0 of that CTA). Design:
//   - CTA0 (even) producer: A0+B multicast + remote expect_tx on both.
//   - CTA1 (odd)  producer: A1 only, local expect_tx for A bytes.
//   - CTA0 free_bar[s] arrival: CTA0 consumers (count 2) — but CTA1's
//     producer ALSO must know CTA1's B stage is free: CTA1 consumers
//     arrive CTA1's free_bar (count 2). CTA0's producer waits CTA0 free_bar
//     AND CTA1 free_bar (remote parity wait) before issuing the next
//     multicast into stage s.
//   - CTA1's producer waits its OWN free_bar for A-stage reuse (A and B
//     share the stage ring, so one wait covers both).
// Ahunt: expect_tx bytes on CTA1's full_bar = A bytes only (B arrives via
// multicast complete_tx from CTA0's transaction — multicast completion
// signals BOTH barriers? NO: the multicast instruction's completion signals
// the mbarrier in EACH destination CTA (the barrier operand is translated
// per destination). So CTA1's expect_tx must ALSO count B bytes.

#include "matmul_wgmma_v9_12.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cooperative_groups.h>

// ---- device helpers -------------------------------------------------------

__device__ __forceinline__ uint64_t w912_make_desc(const void *ptr, uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((1u >> 4) & 0x3FFF) << 16);          // LBO = 1 (assumed)
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    d |= ((uint64_t)1 << 62);                              // 128B swizzle
    return d;
}

#define W912_M64N256K16(d, desc_a, desc_b)                                      \
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

__device__ __forceinline__ void w912_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void w912_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void w912_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
__device__ __forceinline__ void w912_wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n");
}

// ---- mbarrier + TMA --------------------------------------------------------

__device__ __forceinline__ void w912_mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void w912_mbar_arrive(uint64_t *bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
__device__ __forceinline__ void w912_mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void w912_mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P;\n"
        "W:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D;\nbra W;\nD:\n}\n" :: "r"(addr), "r"(parity));
}
__device__ __forceinline__ void w912_tma_2d(const CUtensorMap *map, void *dst,
                                            uint64_t *bar, int x, int y) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
// multicast: one transaction, both CTAs of the pair written
__device__ __forceinline__ void w912_tma_2d_mc(const CUtensorMap *map, void *dst,
                                               uint64_t *bar, int x, int y,
                                               uint16_t ctaMask) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;\n"
        :: "r"(d), "l"(map), "r"(x), "r"(y), "r"(b), "h"(ctaMask) : "memory");
}
// remote arrive on a peer CTA's barrier (mapa translates the address)
__device__ __forceinline__ void w912_mbar_expect_tx_remote(uint64_t *bar, uint32_t bytes,
                                                           uint32_t peerRank) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(peerRank));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
                 :: "r"(remote), "r"(bytes));
}
__device__ __forceinline__ void w912_mbar_arrive_remote(uint64_t *bar,
                                                       uint32_t dstRank) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(dstRank));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" :: "r"(remote));
}
__device__ __forceinline__ void w912_tma_store_2d(const CUtensorMap *map,
                                                  const void *src, int x, int y) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(map), "r"(x), "r"(y), "r"(s) : "memory");
}

// ---- converter kernels ----------------------------------------------------

__global__ void w912_convertF32ToF16(const float * __restrict__ in,
                                     __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void w912_transposeB(const __half * __restrict__ B,
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

extern __shared__ __align__(16) unsigned char w912_raw[];
__device__ unsigned char *w912_smem_ptr() {
    return (unsigned char *)(((uintptr_t)w912_raw + 1023) & ~(uintptr_t)1023);
}
#define w912_smem w912_smem_ptr()

__global__ __launch_bounds__(384) void matmulWgmmaV912Kernel(
    const __grid_constant__ CUtensorMap tmA,
    const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmC,
    float * __restrict__ C, int N) {

    constexpr int STAGES = 3;
    unsigned char *A_s = w912_smem;                       // [4][128*128B]
    unsigned char *B_s = w912_smem + STAGES * (128 * 128);  // [4][256*128B]
    uint64_t *full_bar = reinterpret_cast<uint64_t *>(w912_smem + STAGES * (128 * 128 + 256 * 128));
    uint64_t *free_bar = full_bar + STAGES;

    // cluster: 2 CTAs; rank 0 = even t (multicast leader), rank 1 = odd t
    namespace cg = cooperative_groups;
    const uint32_t rank = (uint32_t)cg::this_cluster().block_rank();
    const uint32_t peer = rank ^ 1u;
    // band swizzle on the CLUSTER id: pairs (2c, 2c+1) share blockN
    const int NM = N / 128;
    const int NN = N / 256;
    constexpr int G = 8;
    const int cid = blockIdx.x / 2;        // cluster index (grid is 1-D now)
    const int bid = cid * 2 + (int)rank;   // this CTA's band-swizzle t
    const int group_size = NN * G;
    const int group_id = bid / group_size;
    const int first_m = group_id * G;
    const int gsize = min(NM - first_m, G);
    const int blockM = first_m + (bid % gsize);
    const int blockN = (bid % group_size) / gsize;
    const int tid = threadIdx.x;
    const int wg = tid / 128;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            // leader's free_bar: released by BOTH CTAs' consumers (B stages
            // are shared); follower's free_bar: its own consumers (A only)
            w912_mbar_init(&full_bar[s], 1);
            w912_mbar_init(&free_bar[s], rank == 0 ? 4 : 2);
        }
    }
    __syncthreads();
    // CLUSTER-wide handshake: no CTA may TMA/expect_tx into a peer's SMEM
    // before the peer has initialized its barriers (the launch-failure bug:
    // remote arrive/expect_tx racing peer mbarrier init = illegal instruction)
    cg::this_cluster().sync();

    if (wg == 0) {
        if (tid == 0) {
            for (int k0 = 0, lap = 0; k0 < N; k0 += 64, lap++) {
                const int s = lap % STAGES;
                if (lap >= STAGES) {
                    const uint32_t c = lap / STAGES;
                    // stage ring reuse: local free_bar only. On the leader it
                    // is arrived by both CTAs' consumers (B shared); on the
                    // follower by its own consumers (A).
                    w912_mbar_wait(&free_bar[s], (c - 1) & 1);
                }
                if (rank == 0) {
                    // leader: expect_tx on BOTH barriers (A+B local, A+B remote)
                    w912_mbar_expect_tx(&full_bar[s], 6 * 64 * 128);
                    w912_mbar_expect_tx_remote(&full_bar[s], 6 * 64 * 128, peer);
                } else {
                    // follower: expect_tx for its A only (2 boxes)
                    w912_mbar_expect_tx(&full_bar[s], 2 * 64 * 128);
                }
                // A tiles: each CTA its own (unicast, own barrier)
                w912_tma_2d(&tmA, A_s + s * 128 * 128, &full_bar[s], k0, blockM * 128);
                w912_tma_2d(&tmA, A_s + s * 128 * 128 + 64 * 128, &full_bar[s], k0, blockM * 128 + 64);
                if (rank == 0) {
                    // B tiles: ONE multicast transaction, mask = both CTAs
                    w912_tma_2d_mc(&tmB, B_s + s * 256 * 128, &full_bar[s], k0, blockN * 256, 0x3);
                    w912_tma_2d_mc(&tmB, B_s + s * 256 * 128 + 64 * 128, &full_bar[s], k0, blockN * 256 + 64, 0x3);
                    w912_tma_2d_mc(&tmB, B_s + s * 256 * 128 + 128 * 128, &full_bar[s], k0, blockN * 256 + 128, 0x3);
                    w912_tma_2d_mc(&tmB, B_s + s * 256 * 128 + 192 * 128, &full_bar[s], k0, blockN * 256 + 192, 0x3);
                }
            }
        }
    } else {
        // ---- consumers: identical to v9.7 ----
        const int ct = tid % 128;
        const int cm = wg - 1;
        const int w = ct / 32;
        const int l = ct % 32;

        float acc[128];
        #pragma unroll
        for (int i = 0; i < 128; i++) acc[i] = 0.0f;

        for (int k0 = 0, lap = 0; k0 < N; k0 += 64, lap++) {
            const int s = lap % STAGES;
            const uint32_t c = lap / STAGES;
            w912_mbar_wait(&full_bar[s], c & 1);

            const unsigned char *Aq = A_s + s * 128 * 128 + cm * 64 * 128;
            const unsigned char *Bq = B_s + s * 256 * 128;

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t da = w912_make_desc(Aq + j * 32, 1024);
                uint64_t db = w912_make_desc(Bq + j * 32, 1024);
                w912_wgmma_fence();
                W912_M64N256K16(acc, da, db);
                w912_wgmma_commit();
            }
            w912_wgmma_wait1();
            if (lap > 0) {
                const int prev = (lap - 1) % STAGES;
                if (ct == 0) {
                    if (rank == 0) {
                        // own ring (A+B, arrived by both CTAs -> count 4)
                        w912_mbar_arrive(&free_bar[prev]);
                    } else {
                        // leader's free_bar (remote arrive for B) + own (A)
                        w912_mbar_arrive_remote(&free_bar[prev], 0);
                        w912_mbar_arrive(&free_bar[prev]);
                    }
                }
            }
        }
        w912_wgmma_wait0();
        if (ct == 0) {
            const int lastS = (N / 64 - 1) % STAGES;
            if (rank == 0) {
                w912_mbar_arrive(&free_bar[lastS]);
            } else {
                w912_mbar_arrive_remote(&free_bar[lastS], 0);
                w912_mbar_arrive(&free_bar[lastS]);
            }
        }

        // epilogue: identical to v9.7 (SMEM staging + 2 bulk stores)
        float *stage = reinterpret_cast<float *>(w912_smem);
        const int rowL = cm * 64 + w * 16 + (l / 4);
        const int colL = (l % 4) * 2;
        #pragma unroll
        for (int g = 0; g < 32; g++) {
            stage[(size_t)rowL * 256 + colL + g * 8]           = acc[g * 4 + 0];
            stage[(size_t)rowL * 256 + colL + g * 8 + 1]       = acc[g * 4 + 1];
            stage[(size_t)(rowL + 8) * 256 + colL + g * 8]     = acc[g * 4 + 2];
            stage[(size_t)(rowL + 8) * 256 + colL + g * 8 + 1] = acc[g * 4 + 3];
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n");
            float *stg = reinterpret_cast<float *>(w912_smem);
            w912_tma_store_2d(&tmC, stg, blockN * 256, blockM * 128);
            w912_tma_store_2d(&tmC, stg + 64 * 256, blockN * 256, blockM * 128 + 64);
            asm volatile("cp.async.bulk.commit_group;\n");
            asm volatile("cp.async.bulk.wait_group 0;\n");
        }
    }
}


// ---- host wrapper ---------------------------------------------------------

void MatmulWgmmaV912::makeMaps() {
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

MatmulWgmmaV912::MatmulWgmmaV912(int N, int blockDim) : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(__half)));
    cudaCheckError(cudaMalloc(&d_Bt16, (size_t)N * N * sizeof(__half)));
}

MatmulWgmmaV912::~MatmulWgmmaV912() {
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_Bt16);
}

void MatmulWgmmaV912::execute(const float *d_A, const float *d_B, float *d_C) {
    int n = N * N;
    w912_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_A, d_A16, n);
    w912_convertF32ToF16<<<(n + 255) / 256, 256>>>(d_B, d_B16, n);
    dim3 tb((N + 31) / 32, (N + 31) / 32);
    w912_transposeB<<<tb, 1024>>>(d_B16, d_Bt16, N);

    if (!mapsReady) makeMaps();

    CUtensorMap tmClocal;
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
        cudaFuncSetAttribute(matmulWgmmaV912Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        smemSet = true;
    }

    // cluster launch: 2-CTA clusters over a 1-D grid of band-swizzle t
    const int NM = N / 128, NN = N / 256;
    const int tiles = NM * NN;
    const int nclusters = tiles / 2 + (tiles & 1);
    dim3 grid(nclusters * 2);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = 3 * (128 * 128 + 256 * 128) + 2 * 3 * 8 + 1024;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(&cfg, matmulWgmmaV912Kernel,
                                         tmA, tmB, tmClocal, d_C, N);
    cudaCheckError(err);
}
