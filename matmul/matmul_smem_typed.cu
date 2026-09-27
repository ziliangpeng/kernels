#include "matmul_smem_typed.h"
#include "cuda_utils.h"
#include "dtype_traits.h"
#include <cuda_runtime.h>

// SMEM tiling matmul with 16-bit storage + FP32 accumulation (rung 3 of the
// multi-precision ladder).
//
// Same structure as matmul_smem.cu (SMEM_TILE=32, one thread per output
// element, K-loop over 32-wide tiles, two __syncthreads per iteration).
// Tiles of A and B are staged in shared memory in the 16-bit format —
// SMEM capacity per tile halves, and GMEM→SMEM traffic halves. The compute
// loop converts each SMEM element to float (cvt) before the FP32 FMA.

#define SMEM_TILE_T 32

template <typename Traits>
__global__ void convertFp32ToKernelSmem(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

template <typename Traits>
__global__ void matmulSmemTypedKernel(const typename Traits::T *A,
                                      const typename Traits::T *B,
                                      float *C, int N) {
    __shared__ typename Traits::T As[SMEM_TILE_T][SMEM_TILE_T];
    __shared__ typename Traits::T Bs[SMEM_TILE_T][SMEM_TILE_T];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * SMEM_TILE_T + ty;
    int col = blockIdx.x * SMEM_TILE_T + tx;

    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < N; tileIdx += SMEM_TILE_T) {
        int A_col = tileIdx + tx;
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = Traits::zero();
        }

        int B_row = tileIdx + ty;
        if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = Traits::zero();
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SMEM_TILE_T; k++) {
            sum += Traits::to_float(As[ty][k]) * Traits::to_float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

template <typename Traits>
MatmulSmemTyped<Traits>::MatmulSmemTyped(int N, int blockDim)
    : N(N), blockDim(blockDim) {
    cudaCheckError(cudaMalloc(&d_A16, (size_t)N * N * sizeof(typename Traits::T)));
    cudaCheckError(cudaMalloc(&d_B16, (size_t)N * N * sizeof(typename Traits::T)));
}

template <typename Traits>
void MatmulSmemTyped<Traits>::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!converted) {
        int threads = 256;
        int blocks = ((long long)N * N + threads - 1) / threads;
        convertFp32ToKernelSmem<Traits><<<blocks, threads>>>(d_A, d_A16, N * N);
        convertFp32ToKernelSmem<Traits><<<blocks, threads>>>(d_B, d_B16, N * N);
        cudaCheckError(cudaGetLastError());
        converted = true;
    }

    dim3 threads(SMEM_TILE_T, SMEM_TILE_T);
    dim3 blocks((N + SMEM_TILE_T - 1) / SMEM_TILE_T,
                (N + SMEM_TILE_T - 1) / SMEM_TILE_T);

    matmulSmemTypedKernel<Traits><<<blocks, threads>>>(d_A16, d_B16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

template <typename Traits>
MatmulSmemTyped<Traits>::~MatmulSmemTyped() {
    if (d_A16) cudaFree(d_A16);
    if (d_B16) cudaFree(d_B16);
}

// Explicit instantiation: FP16 only (16-bit format settled 2026-09-26).
template class MatmulSmemTyped<DTypeTraitsHalf>;
