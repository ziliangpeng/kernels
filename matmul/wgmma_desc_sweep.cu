// wgmma_desc_sweep.cu — empirically find the correct (LBO, SBO, layout) combo
// for K-major no-swizzle wgmma A/B descriptors. N=64 (single wgmma tile),
// CPU-verified per candidate. Prints PASS/FAIL per combo.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

__device__ __forceinline__ uint64_t make_desc(const void *ptr, uint32_t lbo, uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo >> 4) & 0x3FFF) << 16);
    d |= ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
    return d;
}

#define WGMMA_CALL(d, da, db, sd)                                              \
    asm volatile(                                                             \
        "{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "               \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}," \
        " %32, %33, %34, 1, 1, 0, 0;\n}\n"                                    \
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]), \
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]), \
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]), \
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) \
        : "l"(da), "l"(db), "n"(int32_t(sd)))

// layout L0: "interleave128" — atom(mi,ko) at mi*128 + ko*1024 bytes (A[64][16] tile)
// layout L1: "rowpitch32"  — plain A_s[m][k] with 32B row pitch (as [64][16] halfs)
// layout L2: "kblocks"     — ko-major: k0-half block (64x8) then k1-half block: atom(mi,ko) at ko*1024 + mi*16
template <int LAYOUT>
__global__ void wgmmaTest(const __half *A, const __half *Bt, float *C, int N,
                          uint32_t lboA, uint32_t sboA, uint32_t lboB, uint32_t sboB) {
    // K-LOOP version: single 64x16 buffer, reloaded per k-tile, wgmma per
    // step, accumulate — mirrors the main kernel's structure exactly.
    __shared__ __align__(128) unsigned char sA[64 * 16 * 2];
    __shared__ __align__(128) unsigned char sB[64 * 16 * 2];
    const int tid = threadIdx.x;
    asm volatile("wgmma.fence.sync.aligned;\n");
    float acc[32];
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    for (int k0 = 0; k0 < N; k0 += 16) {
        for (int e = tid; e < 64 * 16; e += 128) {
            int m = e / 16, k = e % 16;
            __half v = A[m * N + k0 + k];
            int off;
            if (LAYOUT == 0) off = (m / 8) * 128 + (k / 8) * 1024 + (m % 8) * 16 + (k % 8) * 2;
            else if (LAYOUT == 1) off = m * 32 + k * 2;
            else off = (k / 8) * 1024 + (m / 8) * 128 + (m % 8) * 16 + (k % 8) * 2;
            *reinterpret_cast<__half *>(sA + off) = v;
            __half w = Bt[m * N + k0 + k];
            if (LAYOUT == 0) off = (m / 8) * 128 + (k / 8) * 1024 + (m % 8) * 16 + (k % 8) * 2;
            else if (LAYOUT == 1) off = m * 32 + k * 2;
            else off = (k / 8) * 1024 + (m / 8) * 128 + (m % 8) * 16 + (k % 8) * 2;
            *reinterpret_cast<__half *>(sB + off) = w;
        }
        __syncthreads();
        uint64_t da = make_desc(sA, lboA, sboA);
        uint64_t db = make_desc(sB, lboB, sboB);
        asm volatile("wgmma.fence.sync.aligned;\n");
        WGMMA_CALL(acc, da, db, 1);
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
        __syncthreads();
    }
    // epilogue (m64n64: 8 col groups)
    const int w = tid / 32, l = tid % 32;
    const int rowBase = w * 16 + (l / 4);
    const int colBase = (l % 4) * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        C[(size_t)rowBase * N + colBase + g * 8]     = acc[g * 4 + 0];
        C[(size_t)rowBase * N + colBase + g * 8 + 1] = acc[g * 4 + 1];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8]     = acc[g * 4 + 2];
        C[(size_t)(rowBase + 8) * N + colBase + g * 8 + 1] = acc[g * 4 + 3];
    }
}

int main(int argc, char **argv) {
    const int N = 64;
    std::vector<__half> hA(N * N), hB(N * N), hBt(N * N);
    std::vector<float> A(N * N), B(N * N), ref(N * N, 0.f), got(N * N, 0.f);
    for (long long i = 0; i < (long long)N * N; i++) {
        A[i] = ((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        B[i] = ((i * 2654435761 + 987654321) % 1000) / 1000.0f - 0.5f;
        hA[i] = __float2half(A[i]);
        hB[i] = __float2half(B[i]);
    }
    for (int n = 0; n < N; n++)
        for (int k = 0; k < N; k++) hBt[n * N + k] = hB[k * N + n];  // Bt[n][k]
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) ref[i * N + j] += a * B[k * N + j];
        }
    __half *dA, *dBt; float *dC;
    cudaMalloc(&dA, N * N * 2); cudaMalloc(&dBt, N * N * 2); cudaMalloc(&dC, N * N * 4);
    cudaMemcpy(dA, hA.data(), N * N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBt, hBt.data(), N * N * 2, cudaMemcpyHostToDevice);

    // wait — kernel stages only the FIRST 16-k slice? No: our sweep kernel does
    // one wgmma on k=0..15 only. Restrict verify to k-slice 0..15 contribution.
    std::vector<float> ref16(N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 16; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) ref16[i * N + j] += a * B[k * N + j];
        }

    int lay = atoi(argv[1]);
    uint32_t la = atoi(argv[2]), sa = atoi(argv[3]), lb = atoi(argv[4]), sb = atoi(argv[5]);
    cudaMemset(dC, 0, N * N * 4);
    if (lay == 0) wgmmaTest<0><<<1, 128>>>(dA, dBt, dC, N, la, sa, lb, sb);
    else if (lay == 1) wgmmaTest<1><<<1, 128>>>(dA, dBt, dC, N, la, sa, lb, sb);
    else wgmmaTest<2><<<1, 128>>>(dA, dBt, dC, N, la, sa, lb, sb);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("L%d LBO=%u SBO=%u / LBO=%u SBO=%u : ERR %s\n", lay, la, sa, lb, sb, cudaGetErrorString(e)); return 1; }
    cudaMemcpy(got.data(), dC, N * N * 4, cudaMemcpyDeviceToHost);
    double maxrel = 0;
    for (int i = 0; i < N * N; i++) {
        double d = std::fabs(got[i] - ref16[i]) / std::max(1.0, (double)std::fabs(ref16[i]));
        if (d > maxrel) maxrel = d;
    }
    printf("L%d LBOa=%u SBOa=%u LBOb=%u SBOb=%u : maxrel=%.3e %s\n", lay, la, sa, lb, sb, maxrel, maxrel < 2e-3 ? "PASS" : "");
    return 0;
}
