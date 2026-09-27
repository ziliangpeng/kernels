#include "matmul_vectorized_typed.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>
#include <cstdio>

// Vectorized GEMM (rung 6) with dtype-templated 16-byte vector loads.
//
// Mirrors matmul_vectorized.cu (BM=128 BN=128 BK=8 TM=TN=8, As transposed in
// SMEM for coalesced reads) with the vector width scaled by element size:
// FP32 storage -> float4 (4 elems/load), FP16 storage -> uint4 (8 halves/load,
// 8x fewer load instructions AND half the GMEM bytes of FP32).
//
// The 16-byte unit is chosen because it is the widest single global-load
// instruction on H100; both dtypes saturate the same L1/L2 transaction size.

#define BM_VEC 128
#define BN_VEC 128
#define BK_VEC 8
#define TM_VEC 8
#define TN_VEC 8

#define NUM_THREADS_VEC ((BM_VEC / TM_VEC) * (BN_VEC / TN_VEC))
#define VEC_ELEMS (16 / sizeof(typename Traits::T))  // 4 (fp32) or 8 (fp16)

template <typename Traits>
__global__ void convertVec(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
__global__ void matmulVectorizedTypedKernel(const typename Traits::T *A,
                                            const typename Traits::T *B,
                                            float *C, int N) {
    constexpr int VE = 16 / sizeof(typename Traits::T);  // vector elems per 16B
    using VecT = typename Traits::VecT;                  // float4 or uint4

    __shared__ typename Traits::T As[BK_VEC][BM_VEC];  // transposed [k][m]
    __shared__ typename Traits::T Bs[BK_VEC][BN_VEC];

    const int threadCol = threadIdx.x % (BN_VEC / TN_VEC);
    const int threadRow = threadIdx.x / (BN_VEC / TN_VEC);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_VEC * N;
    B += blockCol * BN_VEC;
    C += blockRow * BM_VEC * N + blockCol * BN_VEC;

    float threadResults[TM_VEC][TN_VEC] = {{0.0f}};
    float regA[TM_VEC];
    float regB[TN_VEC];

    // A tile: BM*BK = 128*8 = 1024 elems / VE per thread
    const int innerRowA = threadIdx.x / (BK_VEC / VE);        // e.g. fp16: 256/1 = 128 rows... see guard
    const int innerColA = threadIdx.x % (BK_VEC / VE);        // which vector in the row

    // B tile: BK*BN = 8*128 = 1024 elems / VE per thread
    const int innerRowB = threadIdx.x / (BN_VEC / VE);
    const int innerColB = threadIdx.x % (BN_VEC / VE);

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_VEC) {
        // --- load A tile (vectorized), store transposed As[k][m] ---
        if (innerRowA < BM_VEC && blockRow * BM_VEC + innerRowA < N) {
            VecT tmp;
            Traits::zero_vec(&tmp);
            bool full = (tileIdx + innerColA * VE + (VE - 1)) < N;
            if (full) {
                tmp = *reinterpret_cast<const VecT *>(&A[innerRowA * N + innerColA * VE]);
            } else {
                for (int i = 0; i < VE; i++) {
                    if (tileIdx + innerColA * VE + i < N)
                        Traits::set_vec_elem(&tmp, i, A[innerRowA * N + innerColA * VE + i]);
                }
            }
            #pragma unroll
            for (int i = 0; i < VE; i++) {
                As[innerColA * VE + i][innerRowA] = Traits::get_vec_elem(tmp, i);
            }
        }

        // --- load B tile (vectorized) ---
        if (innerRowB < BK_VEC && tileIdx + innerRowB < N) {
            VecT tmp;
            Traits::zero_vec(&tmp);
            bool full = (blockCol * BN_VEC + innerColB * VE + (VE - 1)) < N;
            if (full) {
                tmp = *reinterpret_cast<const VecT *>(&B[innerRowB * N + innerColB * VE]);
            } else {
                for (int i = 0; i < VE; i++) {
                    if (blockCol * BN_VEC + innerColB * VE + i < N)
                        Traits::set_vec_elem(&tmp, i, B[innerRowB * N + innerColB * VE + i]);
                }
            }
            #pragma unroll
            for (int i = 0; i < VE; i++) {
                Bs[innerRowB][innerColB * VE + i] = Traits::get_vec_elem(tmp, i);
            }
        }

        __syncthreads();

        A += BK_VEC;
        B += BK_VEC * N;

        // --- compute (outer product; As transposed -> coalesced SMEM reads) ---
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_VEC; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                regA[i] = Traits::to_float(As[dotIdx][threadRow * TM_VEC + i]);
            }
            #pragma unroll
            for (int j = 0; j < TN_VEC; j++) {
                regB[j] = Traits::to_float(Bs[dotIdx][threadCol * TN_VEC + j]);
            }
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                #pragma unroll
                for (int j = 0; j < TN_VEC; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // --- write C (float4 on the output side; C stays FP32) ---
    #pragma unroll
    for (int i = 0; i < TM_VEC; i++) {
        int globalRow = blockRow * BM_VEC + threadRow * TM_VEC + i;
        if (globalRow < N) {
            #pragma unroll
            for (int j = 0; j < TN_VEC; j += 4) {
                int globalCol = blockCol * BN_VEC + threadCol * TN_VEC + j;
                if (globalCol + 3 < N) {
                    float4 tmp;
                    tmp.x = threadResults[i][j + 0];
                    tmp.y = threadResults[i][j + 1];
                    tmp.z = threadResults[i][j + 2];
                    tmp.w = threadResults[i][j + 3];
                    *reinterpret_cast<float4 *>(&C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j]) = tmp;
                } else {
                    for (int k = 0; k < 4 && globalCol + k < N; k++) {
                        C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j + k] = threadResults[i][j + k];
                    }
                }
            }
        }
    }
}

template <typename Traits>
MatmulVectorizedTyped<Traits>::MatmulVectorizedTyped(int N, int blockDim)
    : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
void MatmulVectorizedTyped<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertVec<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertVec<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }
    dim3 threads(NUM_THREADS_VEC);
    dim3 blocks((N + BN_VEC - 1) / BN_VEC,
                (N + BM_VEC - 1) / BM_VEC);
    matmulVectorizedTypedKernel<Traits><<<blocks, threads>>>(d_A16, d_B16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

template <typename Traits>
MatmulVectorizedTyped<Traits>::~MatmulVectorizedTyped() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiations: FP32 control + FP16.
template class MatmulVectorizedTyped<DTypeTraitsFloat>;
template class MatmulVectorizedTyped<DTypeTraitsHalf>;
